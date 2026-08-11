from datetime import datetime
from django.core.management.base import BaseCommand
import yfinance as yf
from core.models import Universe, Symbol, UniverseMember
from market_data.services.prices import get_price_bars

class Command(BaseCommand):
    help = "Fetches historical market data and populates symbols and price bars."

    def add_arguments(self, parser):
        parser.add_argument('--universe', type=str, required=True, help='Name of the universe')
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')

    def handle(self, *args, **options):
        universe_name = options['universe']
        start_date_str = options['start']
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.today().date()
        
        self.stdout.write(f"Processing universe '{universe_name}' starting from {start_date}...")

        # 1. Get or create the Universe
        universe_obj, _ = Universe.objects.get_or_create(name=universe_name)

        # (Optional) If you want default sample tickers for the universe if empty:
        default_tickers = ['RELIANCE', 'TCS', 'INFY']
        
        for ticker in default_tickers:
            yf_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
            
            # Fetch company metadata to populate empty name/sector fields
            try:
                info = yf.Ticker(yf_ticker).info
                company_name = info.get('longName') or info.get('shortName', ticker)
                sector_name = info.get('sector', 'Unknown')
            except Exception:
                company_name = ticker
                sector_name = 'Unknown'

            # Update or create symbol
            symbol_obj, _ = Symbol.objects.update_or_create(
                ticker=ticker,
                defaults={
                    'name': company_name,
                    'sector': sector_name,
                    'is_active': True
                }
            )

            # Link symbol to universe
            UniverseMember.objects.get_or_create(universe=universe_obj, symbol=symbol_obj)

            # 2. Fetch price bars using your robust service function
            self.stdout.write(f"Syncing price bars for {ticker}...")
            get_price_bars(symbol=symbol_obj, start=start_date, end=end_date)

        self.stdout.write(self.style.SUCCESS("Successfully fetched symbols, metadata, and price bars!"))