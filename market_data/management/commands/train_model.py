from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe
from market_data.services.training import train_standard_model_service

class Command(BaseCommand):
    help = "Trains a QuantTransformer model on database data and saves it to TrainedModel."

    def add_arguments(self, parser):
        parser.add_argument('--model-name', type=str, required=True, help='Unique save name for model')
        parser.add_argument('--universe', type=str, default='nifty50', help='Universe slug (e.g., nifty50, banking)')
        parser.add_argument('--symbols', nargs='+', default=None, help='Specific tickers (overrides universe)')
        parser.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
        
        # Transformer & ML Params
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--d-model', type=int, default=64)
        parser.add_argument('--n-heads', type=int, default=4)
        parser.add_argument('--n-layers', type=int, default=2)
        parser.add_argument('--seq-len', type=int, default=60)
        parser.add_argument('--horizon', type=int, default=30)
        
        # Restored Legacy Params
        parser.add_argument('--top-n', type=int, default=10, help='Top N stocks for evaluation tier')
        parser.add_argument('--no-sentiment', action='store_true', help='Skip news sentiment')
        parser.add_argument('--no-backtest', action='store_true', help='Skip backtest after training')

    def handle(self, *args, **options):
        model_name = options['model_name']
        universe_slug = options['universe']
        symbol_tickers = options['symbols']
        
        # 1. Safer Date Parsing
        try:
            start_date = datetime.strptime(options['start'], "%Y-%m-%d").date()
            end_date = datetime.strptime(options['end'], "%Y-%m-%d").date() if options['end'] else datetime.today().date()
        except ValueError:
            raise CommandError("Dates must be in YYYY-MM-DD format.")

        # 2. Resolve Universe and Symbols safely
        universe = None
        symbols = []

        if symbol_tickers:
            # Catch typos or missing tickers in DB
            symbols = list(Symbol.objects.filter(ticker__in=symbol_tickers))
            found_tickers = [s.ticker for s in symbols]
            missing = set(symbol_tickers) - set(found_tickers)
            if missing:
                self.stdout.write(self.style.WARNING(f"Warning: Following symbols not found in DB and will be skipped: {missing}"))
        elif universe_slug:
            try:
                universe = Universe.objects.get(name=universe_slug)
                symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
            except Universe.DoesNotExist:
                raise CommandError(f"Universe '{universe_slug}' not found in database.")
        else:
            raise CommandError("Please specify either a valid --universe or --symbols.")

        if not symbols:
            raise CommandError("No valid symbols found to train on. Aborting.")

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Starting DB-first Transformer training for '{model_name}' on {len(symbols)} symbols..."
        ))

        # 3. Call service with restored parameters
        trained_model = train_standard_model_service(
            model_name=model_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
            seq_len=options['seq_len'],
            horizon=options['horizon'],
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            d_model=options['d_model'],
            n_heads=options['n_heads'],
            n_layers=options['n_layers'],
            use_sentiment=not options['no_sentiment'],
            top_n=options['top_n'],                      # Passed to service
            run_backtest=not options['no_backtest']      # Passed to service
        )

        self.stdout.write(self.style.SUCCESS(f"Training complete! Model registered in DB with ID #{trained_model.id}"))