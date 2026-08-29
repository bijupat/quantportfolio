from typing import Optional

import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe
from core.threading_utils import update_job_progress
from market_data.models import TrainedModel
from forecasting.models import ReportArtifact
from forecasting.services.predictor import get_or_predict_bulk
from forecasting.services.reports import save_portfolio_report_pdf

# Import ranking logic from root
from portfolio_optimizer import rank_stocks, score_to_ranking_table, construct_portfolio

# Total stage count for the "Step X/4" progress display below.
TOTAL_STAGES = 4


class Command(BaseCommand):
    help = "Runs AI inference on a symbol list and outputs a ranked portfolio."

    def add_arguments(self, parser):
        parser.add_argument('--model-name', type=str, required=True, help='Name of TrainedModel to use')
        parser.add_argument('--symbols', nargs='+', default=None, help='Specific tickers to predict')
        parser.add_argument('--universe', type=str, default=None, help='Universe to predict (e.g., nifty50)')
        parser.add_argument('--as-of', type=str, default=None, help='Date to predict for (YYYY-MM-DD)')
        parser.add_argument('--top-n', type=int, default=10, help='Number of stocks in BUY tier')
        parser.add_argument('--no-sentiment', action='store_true', help='Skip news sentiment features')
        parser.add_argument('--plot', action='store_true', help='Save PDF report')
        parser.add_argument(
            '--job-id', type=int, default=None,
            help='PK of the core.JobRun tracking this run, if launched via launch_tracked_command. '
                 'When supplied, structured progress is reported to that JobRun as each stage '
                 'completes (see core.threading_utils.update_job_progress). Optional — omitted '
                 'entirely when run from the CLI directly, in which case progress reporting is '
                 'simply skipped.',
        )

    def _report_progress(self, job_id: Optional[int], stage: int, message: str) -> None:
        """Updates the tracking JobRun's progress fields, if this run has one.

        No-op when job_id is None (CLI-only runs) — see run_composite.py's
        identical helper for the full rationale; kept as a local copy here
        rather than a shared mixin since each command's TOTAL_STAGES and
        stage wording differ, and the method body itself is a single call.
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
        model_name = options['model_name']
        as_of_date = datetime.strptime(options['as_of'], "%Y-%m-%d").date() if options['as_of'] else datetime.today().date()

        try:
            trained_model = TrainedModel.objects.get(name=model_name)
        except TrainedModel.DoesNotExist:
            raise CommandError(f"Model '{model_name}' not found in database.")

        # Resolve Symbols
        self._report_progress(job_id, 1, "Resolving symbol universe...")
        universe_slug = options['universe']
        symbol_tickers = options['symbols']

        if symbol_tickers:
            symbols = list(Symbol.objects.filter(ticker__in=symbol_tickers))
        elif universe_slug:
            try:
                universe = Universe.objects.get(name=universe_slug)
                symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
            except Universe.DoesNotExist:
                raise CommandError(f"Universe '{universe_slug}' not found.")
        elif trained_model.trained_on:
            self.stdout.write(self.style.WARNING(f"No symbols provided. Defaulting to model's training universe: {trained_model.trained_on.name}"))
            symbols = [m.symbol for m in trained_model.trained_on.members.select_related('symbol').all()]
        else:
            raise CommandError("Please specify --universe, --symbols, or ensure the model has a linked trained_on universe.")

        if not symbols:
            raise CommandError("No symbols found.")

        self.stdout.write(self.style.MIGRATE_HEADING(f"Running inference for {len(symbols)} symbols using '{model_name}'..."))

        # Run DB-First Inference
        self._report_progress(job_id, 2, f"Running inference on {len(symbols)} symbols with '{model_name}'...")
        predictions_map = get_or_predict_bulk(
            symbols=symbols,
            trained_model=trained_model,
            as_of=as_of_date,
            use_sentiment=not options['no_sentiment']
        )

        if not predictions_map:
            raise CommandError("No predictions generated. Check data availability.")

        # Prepare for ranking
        self._report_progress(job_id, 3, "Ranking symbols and constructing portfolio weights...")
        raw_scores = {p.symbol.ticker: p.predicted_return for p in predictions_map.values()}

        ranking_df = score_to_ranking_table(raw_scores)
        tiers = rank_stocks(raw_scores, top_n=options['top_n'], hold_n=15)
        weights = construct_portfolio(raw_scores, top_n=options['top_n'], weighting="score")

        # Console Output
        self.stdout.write("\n" + "═" * 64)
        self.stdout.write("  QuantPortfolioAI  —  Predicted 30-Day Return Ranking")
        self.stdout.write("═" * 64)
        self.stdout.write(f"  {'Rank':<6} {'Symbol':<20} {'Score':>8}  {'Tier'}")
        self.stdout.write("─" * 64)

        for rank, (sym, row) in enumerate(ranking_df.iterrows(), 1):
            tier = tiers.get(sym, "AVOID")
            score = row["norm_score"]
            icon = {"BUY": "▲", "HOLD": "─", "AVOID": "▼"}.get(tier, " ")
            self.stdout.write(f"  {rank:<6} {sym:<20} {score:>8.4f}  {icon} {tier}")

        self.stdout.write("\n  Portfolio Weights (BUY tier):")
        self.stdout.write("─" * 42)
        for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 50)
            self.stdout.write(f"  {sym:<20} {w:>6.1%}  {bar}")

        # PDF Report
        if options['plot']:
            self._report_progress(job_id, 4, "Saving PDF report...")
            artifact = save_portfolio_report_pdf(
                ranking_df=ranking_df,
                weights=weights,
                tiers=tiers,
                raw_scores=raw_scores,
                model_name=model_name,
                source=ReportArtifact.Source.PREDICT,
            )
            self.stdout.write(self.style.SUCCESS(f"\nPDF Report Saved: {artifact.file.name}"))

        self._report_progress(job_id, TOTAL_STAGES, "Done.")