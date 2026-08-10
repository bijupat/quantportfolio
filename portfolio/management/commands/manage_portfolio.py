from django.core.management.base import BaseCommand, CommandError
from portfolio.models import Portfolio
from portfolio.services.portfolio_service import load_portfolio_from_csv_to_db

class Command(BaseCommand):
    help = "Manages stored database portfolios."

    def add_arguments(self, parser):
        parser.add_argument('--list', action='store_true', help='List all saved portfolios')
        parser.add_argument('--load-csv', type=str, help='Path to legacy portfolio CSV to import')
        parser.add_argument('--name', type=str, default='imported_portfolio', help='Name for the imported portfolio')
        parser.add_argument('--view', type=str, help='View holdings of a specific portfolio name')

    def handle(self, *args, **options):
        if options['list']:
            portfolios = Portfolio.objects.all()
            self.stdout.write(self.style.MIGRATE_HEADING("Stored Portfolios in Database:"))
            for p in portfolios:
                self.stdout.write(f"  - {p.name} (Capital: Rs. {p.total_capital:,.2f}, Strategy: {p.strategy}, Created: {p.created_at.strftime('%Y-%m-%d')})")
            return

        if options['load_csv']:
            csv_path = options['load_csv']
            name = options['name']
            self.stdout.write(f"Importing portfolio from {csv_path} as '{name}'...")
            portfolio = load_portfolio_from_csv_to_db(csv_path, name)
            self.stdout.write(self.style.SUCCESS(f"Successfully imported portfolio ID #{portfolio.id} with {portfolio.items.count()} items!"))
            return

        if options['view']:
            name = options['view']
            try:
                p = Portfolio.objects.get(name=name)
            except Portfolio.DoesNotExist:
                raise CommandError(f"Portfolio '{name}' not found.")
                
            self.stdout.write(self.style.MIGRATE_HEADING(f"Holdings for Portfolio: {p.name}"))
            self.stdout.write(f"{'Symbol':<15} {'Qty':>8} {'Price':>12} {'Alloc %':>10} {'Value (Rs.)':>14}")
            self.stdout.write("-" * 63)
            for item in p.items.select_related('symbol').all():
                val = item.quantity * item.purchase_price
                self.stdout.write(f"{item.symbol.ticker:<15} {item.quantity:>8} {item.purchase_price:>12.2f} {item.allocation_pct:>9.1f}% {val:>14,.2f}")
            self.stdout.write("-" * 63)
            self.stdout.write(f"{'TOTAL CAPITAL':<35} Rs. {p.total_capital:,.2f}")
            return

        self.stdout.write("Please specify an action: --list, --load-csv, or --view.")