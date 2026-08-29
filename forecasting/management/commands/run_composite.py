from datetime import datetime
from typing import Optional

from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe, User
from core.threading_utils import update_job_progress
from market_data.models import TrainedModel
from portfolio.services.portfolio_service import save_portfolio_to_db
from forecasting.services.ensemble import run_ensemble_layer
from forecasting.services.composite import (
    compute_composite_scores,
    build_composite_portfolio,
    compute_config_hash,
    save_composite_scores_to_db,
)
from forecasting.services.reports import save_portfolio_report_pdf, save_score_breakdown_excel
from portfolio_optimizer import rank_stocks, score_to_ranking_table, construct_portfolio

# Total stage count for the "Step X/5" progress display below. Kept as a
# module constant (not a magic number sprinkled through handle()) so the
# two save-report sub-stages (PDF, Excel) staying folded into a single
# "Step 5" — rather than becoming steps 6/7 — is a visible, deliberate
# choice, not an accident of wherever update_job_progress happened to be
# called last.
TOTAL_STAGES = 5


class Command(BaseCommand):
    help = "Runs composite/ensemble scoring over a universe of stocks."

    def add_arguments(self, parser):
        parser.add_argument('--models', nargs='+', required=True, help='List of TrainedModel names')
        parser.add_argument('--model-weights', nargs='+', type=float, help='Weights for ensemble models')
        parser.add_argument('--symbols', nargs='+', default=None, help='Specific tickers')
        parser.add_argument('--universe', type=str, default='nifty50', help='Universe slug')
        parser.add_argument('--as-of', type=str, default=None, help='Date to predict for (YYYY-MM-DD)')
        parser.add_argument('--top-n', type=int, default=10, help='Number of BUY-tier stocks')
        parser.add_argument('--amount', type=float, default=100000.0, help='Portfolio capital')
        parser.add_argument('--user-id', type=int, default=None, help='ID of the core.User who triggered this run (sets Portfolio.owner)')
        parser.add_argument(
            '--job-id', type=int, default=None,
            help='PK of the core.JobRun tracking this run, if launched via launch_tracked_command. '
                 'When supplied, structured progress is reported to that JobRun as each stage '
                 'completes (see core.threading_utils.update_job_progress) so the web UI can show '
                 'more than a bare "Running" spinner. Optional — omitted entirely when run from '
                 'the CLI directly, in which case progress reporting is simply skipped.',
        )
        # Layer toggles
        parser.add_argument('--w-transformer', type=float, default=0.80)
        parser.add_argument('--w-quality', type=float, default=0.12)
        parser.add_argument('--w-technical', type=float, default=0.08)
        parser.add_argument('--no-sentiment', action='store_true', help='Skip news sentiment')
        parser.add_argument('--plot', action='store_true', help='Save PDF and Excel reports')

    def _report_progress(self, job_id: Optional[int], stage: int, message: str) -> None:
        """Updates the tracking JobRun's progress fields, if this run has one.

        A thin wrapper around update_job_progress so every call site below
        reads as one line and doesn't need to repeat the "only if job_id is
        set" guard. Silently a no-op when job_id is None (CLI-only runs).

        Args:
            job_id: PK of the JobRun to update, or None to skip reporting.
            stage: Current stage number out of TOTAL_STAGES.
            message: Human-readable description of the stage just starting.
        """
        if job_id is None:
            return
        update_job_progress(
            job_id,
            message=f"Step {stage}/{TOTAL_STAGES}: {message}",
            current=stage,
            total=TOTAL_STAGES,
        )

    def handle(self, *args, **options):
        job_id = options.get('job_id')
        as_of_date = datetime.strptime(options['as_of'], "%Y-%m-%d").date() if options['as_of'] else datetime.today().date()

        # Resolve Models
        model_names = options['models']
        weights = options['model_weights'] or [1.0] * len(model_names)

        if len(model_names) != len(weights):
            raise CommandError("Number of models must match number of model-weights.")

        models_with_weights = {}
        for m_name, w in zip(model_names, weights):
            try:
                models_with_weights[TrainedModel.objects.get(name=m_name)] = w
            except TrainedModel.DoesNotExist:
                raise CommandError(f"Model '{m_name}' not found in DB.")

        # Resolve Symbols
        if options['symbols']:
            symbols = list(Symbol.objects.filter(ticker__in=options['symbols']))
        else:
            try:
                universe = Universe.objects.get(name=options['universe'])
                symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
            except Universe.DoesNotExist:
                raise CommandError(f"Universe '{options['universe']}' not found.")

        self.stdout.write(self.style.MIGRATE_HEADING("Running Composite Engine..."))

        # 1. Ensemble Layer
        self._report_progress(
            job_id, 1,
            f"Running {len(models_with_weights)} model(s) across {len(symbols)} symbols..."
        )
        # run_ensemble_layer now returns a 3rd value, contributing_models,
        # mapping each ticker to the list of model names that actually
        # scored it. A symbol scored by fewer than len(models_with_weights)
        # models is flagged downstream in compute_composite_scores'
        # data_quality dict, rather than silently blending in as if fully
        # covered.
        transformer_scores, per_model_scores, contributing_models = run_ensemble_layer(
            symbols=symbols,
            models_with_weights=models_with_weights,
            as_of=as_of_date,
            use_sentiment=not options['no_sentiment']
        )

        # 2. Composite Layer
        self._report_progress(
            job_id, 2,
            "Blending transformer, quality, and technical scores..."
        )
        weights_config = {
            "transformer": options['w_transformer'],
            "quality": options['w_quality'],
            "technical": options['w_technical']
        }

        composite_results = compute_composite_scores(
            symbols=symbols,
            as_of=as_of_date,
            transformer_scores=transformer_scores,
            weights_config=weights_config,
            per_model_scores=per_model_scores,
            contributing_models=contributing_models,
            n_models_requested=len(models_with_weights),
        )

        # 2b. Console visibility for any symbol with reduced data quality —
        # surfaced here (not just in logs) so a person running this
        # interactively sees it without needing to check the log file.
        low_quality_symbols = [
            sym for sym, d in composite_results.items()
            if d.get("data_quality", {}).get("quality_no_data")
            or d.get("data_quality", {}).get("technical_no_data")
            or d.get("data_quality", {}).get("partial_ensemble")
        ]
        if low_quality_symbols:
            self.stdout.write(self.style.WARNING(
                f"\n{len(low_quality_symbols)} of {len(composite_results)} symbols have reduced "
                f"data quality (missing quality/technical history, or scored by only a subset of "
                f"the ensemble): {', '.join(low_quality_symbols)}"
            ))

        # 3. Portfolio Ranking
        self._report_progress(job_id, 3, "Ranking symbols and constructing portfolio weights...")
        comp_scores_flat = {sym: d["composite_score"] for sym, d in composite_results.items()}
        ranking_df = score_to_ranking_table(comp_scores_flat)
        tiers = rank_stocks(comp_scores_flat, top_n=options['top_n'], hold_n=15)
        portfolio_weights = construct_portfolio(comp_scores_flat, top_n=options['top_n'], weighting="score")

        # 3b. Persist composite scores for this run. config_hash identifies the
        # scoring config so an identical re-run on the same as_of_date replaces
        # these rows instead of erroring on the unique_together constraint.
        config_hash = compute_config_hash(
            model_names=model_names,
            model_weights=weights,
            weights_config=weights_config,
            use_sentiment=not options['no_sentiment'],
        )
        composite_score_rows = list(save_composite_scores_to_db(
            composite_results=composite_results,
            tiers=tiers,
            as_of_date=as_of_date,
            config_hash=config_hash,
        ))

        # ReportArtifact.composite_run is a single FK, but CompositeScore is per-symbol —
        # link to the top-ranked symbol's row as a stand-in for "this run". The full
        # run is still queryable via CompositeScore.objects.filter(as_of_date=..., config_hash=...).
        representative_score = None
        if not ranking_df.empty:
            top_symbol = ranking_df.index[0]
            representative_score = next(
                (r for r in composite_score_rows if r.symbol.ticker == top_symbol), None
            )
            if representative_score is None:
                self.stdout.write(self.style.WARNING(
                    f"Top-ranked symbol '{top_symbol}' has no persisted CompositeScore row "
                    f"(likely missing from the Symbol table) — ReportArtifact.composite_run "
                    f"will be left unset for this run."
                ))

        # 4. Build structured holdings for Excel export. Pass as_of=as_of_date so
        # historical/backtested runs price holdings as of the run's actual
        # date rather than "today" (see build_composite_portfolio's as_of arg).
        self._report_progress(job_id, 4, "Pricing holdings and building portfolio allocations...")
        portfolio_holdings = build_composite_portfolio(
            composite_results=composite_results,
            top_n=options['top_n'],
            portfolio_amount=options['amount'],
            as_of=as_of_date,
            weighting="score"
        )

        # Console Output
        self.stdout.write("\n" + "=" * 72)
        self.stdout.write(f"  {'Rank':<5} {'Symbol':<20} {'Composite':>10}  {'Transformer':>12}  Tier")
        self.stdout.write("-" * 72)

        for rank, (sym, row) in enumerate(ranking_df.iterrows(), 1):
            tier = tiers.get(sym, "AVOID")
            comp = row["norm_score"]
            trans = composite_results[sym]["transformer_score"]
            icon = {"BUY": "▲", "HOLD": "─", "AVOID": "▼"}.get(tier, " ")
            flag = " *" if sym in low_quality_symbols else ""
            self.stdout.write(f"  {rank:<5} {sym:<20} {comp:>10.4f}  {trans:>12.5f}  {icon} {tier}{flag}")

        if low_quality_symbols:
            self.stdout.write("\n  * = reduced data quality (see warning above)")

        # 5. Save Portfolio to DB and Reports
        if options['plot']:
            self._report_progress(job_id, 5, "Saving portfolio and generating PDF/Excel reports...")
            owner = None
            user_id = options.get('user_id')
            if user_id:
                try:
                    owner = User.objects.get(pk=user_id)
                except User.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f"User id {user_id} not found; portfolio will be saved without an owner."))
            portfolio_name = f"composite_{'_'.join(model_names)[:10]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_portfolio = save_portfolio_to_db(
                name=portfolio_name,
                holdings=portfolio_holdings,
                total_amount=options['amount'],
                strategy_tag="composite_ai",
                owner=owner,
            )
            self.stdout.write(self.style.SUCCESS(f"Portfolio successfully saved to database as: '{saved_portfolio.name}'"))

            pdf_artifact = save_portfolio_report_pdf(
                ranking_df=ranking_df,
                weights=portfolio_weights,
                tiers=tiers,
                model_name=f"Composite ({(', '.join(model_names))[:15]}...)",
                portfolio=saved_portfolio,
                composite_run=representative_score,
            )
            self.stdout.write(self.style.SUCCESS(f"PDF Report Saved: {pdf_artifact.file.name}"))

            xlsx_artifact = save_score_breakdown_excel(
                composite_results=composite_results,
                portfolio_holdings=portfolio_holdings,
                tiers=tiers,
                model_names=model_names,
                model_weights=weights,
                weights_used=weights_config,
                portfolio_amount=options['amount'],
                portfolio=saved_portfolio,
                composite_run=representative_score,
            )
            if xlsx_artifact:
                self.stdout.write(self.style.SUCCESS(f"Excel Score Breakdown Saved: {xlsx_artifact.file.name}"))

        self._report_progress(job_id, TOTAL_STAGES, "Done.")