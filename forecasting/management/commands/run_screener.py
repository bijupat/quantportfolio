from datetime import datetime
from django.core.management.base import BaseCommand, CommandError
from core.models import Symbol, Universe
from forecasting.services.screener import get_or_screen
from forecasting.services.reports import save_screener_excel

class Command(BaseCommand):
    help = "Filters a symbol universe using price/volume logic."

    def add_arguments(self, parser):
        parser.add_argument('--universe', type=str, default='nifty50')
        parser.add_argument('--symbols', nargs='+', default=None)
        parser.add_argument('--as-of', type=str, default=None)
        parser.add_argument('--top-n', type=int, default=30)
        parser.add_argument('--min-score', type=int, default=3)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--save-csv', action='store_true', help="Save to Excel")

    def handle(self, *args, **options):
        universe_slug = options['universe']
        symbol_tickers = options['symbols']
        as_of_date = datetime.strptime(options['as_of'], "%Y-%m-%d").date() if options['as_of'] else datetime.today().date()

        if symbol_tickers:
            symbols = list(Symbol.objects.filter(ticker__in=symbol_tickers))
        elif universe_slug:
            try:
                universe = Universe.objects.get(name=universe_slug)
                symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
            except Universe.DoesNotExist:
                raise CommandError(f"Universe '{universe_slug}' not found.")
        else:
            raise CommandError("Please specify --universe or --symbols.")

        if not symbols:
            raise CommandError("No symbols found.")

        self.stdout.write(self.style.MIGRATE_HEADING(f"Screening {len(symbols)} symbols as of {as_of_date}..."))

        results_qs = get_or_screen(symbols, as_of_date, options['min_score'], options['top_n'])
        
        # Sort in Python to handle the status hierarchy cleanly
        order_map = {"PASS": 0, "PASS (not selected)": 1, "NEAR MISS": 2, "FAIL": 3, "NO DATA": 4}
        results = sorted(list(results_qs), key=lambda x: (order_map.get(x.status, 5), -x.score))

        passed = [r for r in results if r.status == "PASS"]
        near_miss = [r for r in results if r.status == "NEAR MISS"]
        failed = [r for r in results if r.status == "FAIL"]

        self.stdout.write(self.style.SUCCESS(f"\nPASSED ({len(passed)} symbols):"))
        for r in passed:
            self.stdout.write(f"  {r.symbol.ticker:15} | Score: {r.score}/6 | Close: {r.raw_values.get('close', 'N/A'):.2f}")

        if options['verbose']:
            self.stdout.write(self.style.WARNING(f"\nNEAR MISS ({len(near_miss)} symbols):"))
            for r in near_miss:
                self.stdout.write(f"  {r.symbol.ticker:15} | Score: {r.score}/6")
                
            self.stdout.write(self.style.ERROR(f"\nFAILED ({len(failed)} symbols):"))
            for r in failed:
                self.stdout.write(f"  {r.symbol.ticker:15} | Score: {r.score}/6")

        self.stdout.write("\n" + "="*50)
        
        if options['save_csv']:
            artifact = save_screener_excel(results, options['min_score'])
            if artifact:
                self.stdout.write(self.style.SUCCESS(f"Excel report saved to: {artifact.file.name}"))