from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe
from market_data.services.training import train_hybrid_model_service


class Command(BaseCommand):
    help = "Trains a TripleBranchHybridModel on database data and saves it to TrainedModel."

    def add_arguments(self, parser):
        parser.add_argument('--model-name', type=str, required=True, help='Unique save name for model')
        parser.add_argument('--universe', type=str, default='nifty50', help='Universe slug (e.g., nifty50, banking)')
        parser.add_argument('--symbols', nargs='+', default=None, help='Specific tickers (overrides universe)')
        parser.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
        parser.add_argument('--epochs', type=int, default=60)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--d-model', type=int, default=64)
        parser.add_argument('--n-heads', type=int, default=4)
        parser.add_argument('--n-layers', type=int, default=2)
        parser.add_argument('--seq-len', type=int, default=120)
        parser.add_argument('--horizon', type=int, default=30, help='Forward-return horizon in trading days')
        parser.add_argument('--lstm-units', type=int, default=64)
        parser.add_argument('--no-sentiment', action='store_true', help='Skip news sentiment branch')

    def handle(self, *args, **options):
        model_name = options['model_name']
        universe_slug = options['universe']
        symbol_tickers = options['symbols']

        start_date = datetime.strptime(options['start'], "%Y-%m-%d").date()
        end_date = datetime.strptime(options['end'], "%Y-%m-%d").date() if options['end'] else datetime.today().date()

        universe = None
        if universe_slug:
            try:
                universe = Universe.objects.get(name=universe_slug)
            except Universe.DoesNotExist:
                self.stdout.write(self.style.WARNING(f"Universe '{universe_slug}' not found."))

        if symbol_tickers:
            symbols = list(Symbol.objects.filter(ticker__in=symbol_tickers))
        elif universe:
            symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
        else:
            raise CommandError("Please specify either a valid --universe or --symbols.")

        if not symbols:
            raise CommandError("No valid symbols found to train on.")

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Starting DB-first Hybrid training for '{model_name}' on {len(symbols)} symbols..."
        ))

        trained_model = train_hybrid_model_service(
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
            lstm_units=options['lstm_units'],
            use_sentiment=not options['no_sentiment'],
        )

        self.stdout.write(self.style.SUCCESS(
            f"Hybrid training complete! Model registered in DB with ID #{trained_model.id}"
        ))