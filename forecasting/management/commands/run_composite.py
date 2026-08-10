from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe
from market_data.models import TrainedModel
from portfolio.services.portfolio_service import save_portfolio_to_db
from forecasting.services.ensemble import run_ensemble_layer
from forecasting.services.composite import compute_composite_scores
from forecasting.services.reports import save_portfolio_report_pdf, save_score_breakdown_excel
from portfolio_optimizer import rank_stocks, score_to_ranking_table, construct_portfolio
from portfolio.services.portfolio_service import save_portfolio_to_db
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
        
        # Layer toggles
        parser.add_argument('--w-transformer', type=float, default=0.80)
        parser.add_argument('--w-quality', type=float, default=0.12)
        parser.add_argument('--w-technical', type=float, default=0.08)
        parser.add_argument('--no-sentiment', action='store_true', help='Skip news sentiment')
        parser.add_argument('--plot', action='store_true', help='Save PDF and Excel reports')

    def handle(self, *args, **options):
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
        transformer_scores, per_model_scores = run_ensemble_layer(
            symbols=symbols,
            models_with_weights=models_with_weights,
            as_of=as_of_date,
            use_sentiment=not options['no_sentiment']
        )
        
        # 2. Composite Layer
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
            per_model_scores=per_model_scores
        )

        # 3. Portfolio Ranking
        comp_scores_flat = {sym: d["composite_score"] for sym, d in composite_results.items()}
        ranking_df = score_to_ranking_table(comp_scores_flat)
        tiers = rank_stocks(comp_scores_flat, top_n=options['top_n'], hold_n=15)
        portfolio_weights = construct_portfolio(comp_scores_flat, top_n=options['top_n'], weighting="score")
        
        # Build structured holdings for Excel export
        from forecasting.services.composite import build_composite_portfolio
        portfolio_holdings = build_composite_portfolio(
            composite_results=composite_results,
            top_n=options['top_n'],
            portfolio_amount=options['amount'],
            weighting="score"
        )

        # 4. Console Output
        self.stdout.write("\n" + "=" * 72)
        self.stdout.write(f"  {'Rank':<5} {'Symbol':<20} {'Composite':>10}  {'Transformer':>12}  Tier")
        self.stdout.write("-" * 72)
        
        for rank, (sym, row) in enumerate(ranking_df.iterrows(), 1):
            tier = tiers.get(sym, "AVOID")
            comp = row["norm_score"]
            trans = composite_results[sym]["transformer_score"]
            icon = {"BUY": "▲", "HOLD": "─", "AVOID": "▼"}.get(tier, " ")
            self.stdout.write(f"  {rank:<5} {sym:<20} {comp:>10.4f}  {trans:>12.5f}  {icon} {tier}")

        # 5. Save Portfolio to DB and Reports
        if options['plot']:
            portfolio_name = f"composite_{'_'.join(model_names)[:10]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_portfolio = save_portfolio_to_db(
                name=portfolio_name,
                holdings=portfolio_holdings,
                total_amount=options['amount'],
                strategy_tag="composite_ai"
            )
            self.stdout.write(self.style.SUCCESS(f"Portfolio successfully saved to database as: '{saved_portfolio.name}'"))

            pdf_artifact = save_portfolio_report_pdf(
                ranking_df=ranking_df,
                weights=portfolio_weights,
                tiers=tiers,
                model_name=f"Composite ({(', '.join(model_names))[:15]}...)"
            )
            self.stdout.write(self.style.SUCCESS(f"PDF Report Saved: {pdf_artifact.file.name}"))

            xlsx_artifact = save_score_breakdown_excel(
                composite_results=composite_results,
                portfolio_holdings=portfolio_holdings,
                tiers=tiers,
                model_names=model_names,
                model_weights=weights,
                weights_used=weights_config,
                portfolio_amount=options['amount']
            )
            if xlsx_artifact:
                self.stdout.write(self.style.SUCCESS(f"Excel Score Breakdown Saved: {xlsx_artifact.file.name}"))