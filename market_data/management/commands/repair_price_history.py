"""market_data/management/commands/repair_price_history.py

One-off remediation for the NonTradingDay cache-poisoning bug (Aug 2026):
a newly-listed/demerged equity's pre-listing gap was mis-recorded as a
market-wide holiday, silently truncating other, older symbols' history.

  1. Wipes the untrustworthy NonTradingDay table.
  2. Rebuilds it cleanly from ^NSEI.
  3. Re-runs get_price_bars for every symbol in the given universe(s) so
     previously wrongly-skipped trading days get backfilled.

Idempotent — safe to re-run.
"""

from datetime import date, datetime

from django.core.management.base import BaseCommand, CommandError

from core.models import Universe
from market_data.models import NonTradingDay
from market_data.services.prices import get_price_bars, sync_market_calendar


class Command(BaseCommand):
    """Purges the corrupted NonTradingDay cache and backfills affected universes."""

    help = "Repairs price history truncated by the NonTradingDay cache-poisoning bug."

    def add_arguments(self, parser) -> None:
        parser.add_argument('--universe', action='append', required=True,
                             help='Universe slug to repair. Repeat flag for multiple universes.')
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD), defaults to today')

    def handle(self, *args, **options) -> None:
        start: date = datetime.strptime(options['start'], "%Y-%m-%d").date()
        end: date = datetime.strptime(options['end'], "%Y-%m-%d").date() if options['end'] else date.today()

        wiped, _ = NonTradingDay.objects.all().delete()
        self.stdout.write(self.style.WARNING(f"Wiped {wiped} NonTradingDay row(s)."))

        holiday_count = sync_market_calendar(start, end)
        self.stdout.write(self.style.SUCCESS(f"Rebuilt calendar: {holiday_count} confirmed holidays in range."))

        for universe_name in options['universe']:
            try:
                universe = Universe.objects.get(name=universe_name)
            except Universe.DoesNotExist:
                raise CommandError(f"Universe '{universe_name}' not found.")

            symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
            self.stdout.write(self.style.MIGRATE_HEADING(f"Repairing {len(symbols)} symbols in '{universe_name}'..."))
            for symbol_obj in symbols:
                before = symbol_obj.bars.filter(date__range=(start, end)).count()
                get_price_bars(symbol_obj, start, end)
                after = symbol_obj.bars.filter(date__range=(start, end)).count()
                if (delta := after - before):
                    self.stdout.write(f"  {symbol_obj.ticker:<16} +{delta} bars ({before} -> {after})")

        self.stdout.write(self.style.SUCCESS("Repair complete."))