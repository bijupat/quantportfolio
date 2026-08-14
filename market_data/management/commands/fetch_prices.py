from datetime import date, datetime
from typing import Optional

import yfinance as yf
from django.core.management.base import BaseCommand, CommandError

from core.models import Symbol, Universe
from market_data.services.prices import get_price_bars, backfill_symbol_metadata


class Command(BaseCommand):
    """Fetches historical market data for all symbols belonging to a pre-seeded Universe."""

    help = "Fetches historical market data and populates price bars for an existing universe."

    def add_arguments(self, parser) -> None:
        parser.add_argument('--universe', type=str, required=True, help='Slug of a pre-seeded universe (e.g., nifty50)')
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')

    def handle(self, *args, **options) -> None:
        universe_name: str = options['universe']
        start_date_str: str = options['start']
        start_date: date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date: date = datetime.today().date()

        # 1. Resolve the Universe — do NOT get_or_create here.
        #    A missing universe means seed_universes hasn't been run for it,
        #    and silently creating an empty one is what caused the original bug.
        try:
            universe_obj = Universe.objects.get(name=universe_name)
        except Universe.DoesNotExist:
            raise CommandError(
                f"Universe '{universe_name}' not found. "
                f"Run `python manage.py seed_universes` first to populate it."
            )

        # 2. Pull tickers from the Universe's existing membership — this is the
        #    actual fix: the ticker list now comes from the DB, not a hardcoded list.
        symbols = [m.symbol for m in universe_obj.members.select_related('symbol').all()]

        if not symbols:
            raise CommandError(
                f"Universe '{universe_name}' has no members. "
                f"Run `python manage.py seed_universes` first to populate it."
            )

        self.stdout.write(
            self.style.MIGRATE_HEADING(
                f"Processing universe '{universe_name}' ({len(symbols)} symbols) starting from {start_date}..."
            )
        )

        # 3. Fetch metadata + price bars for each symbol already tied to this universe.
        for symbol_obj in symbols:
            ticker = symbol_obj.ticker
            yf_ticker = ticker if ticker.endswith(('.NS', '.BO')) else f"{ticker}.NS"

            # Backfill name/sector if missing (seed_universes doesn't set these)
            if backfill_symbol_metadata(symbol_obj):
                self.stdout.write(f"  Backfilled metadata for {ticker}: {symbol_obj.name} ({symbol_obj.sector})")
            elif not symbol_obj.name or not symbol_obj.sector:
                self.stdout.write(self.style.WARNING(f"  Could not backfill metadata for {ticker} — check logs"))

            self.stdout.write(f"  Syncing price bars for {ticker}...")
            get_price_bars(symbol=symbol_obj, start=start_date, end=end_date)

        self.stdout.write(self.style.SUCCESS(
            f"Successfully fetched price bars for {len(symbols)} symbols in universe '{universe_name}'!"
        ))