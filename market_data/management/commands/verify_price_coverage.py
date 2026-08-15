"""market_data/management/commands/verify_price_coverage.py

Read-only diagnostic: reports per-symbol PriceBar row counts, date ranges,
and coverage gaps for a universe. Does not fetch or modify any data — use
this to confirm whether a fetch_prices run (especially one that returned
suspiciously fast because everything was already cached) actually covers
the requested range for every symbol.
"""

from datetime import date, datetime, timedelta

from django.core.management.base import BaseCommand, CommandError

from core.models import Universe
from market_data.models import PriceBar, NonTradingDay


def _count_weekdays(start: date, end: date) -> int:
    """Counts Monday-Friday calendar days in [start, end], inclusive."""
    days = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            days += 1
        current += timedelta(days=1)
    return days


class Command(BaseCommand):
    """Reports PriceBar coverage (row counts, date ranges, gaps) for every symbol in a universe."""

    help = "Verifies PriceBar coverage for a universe without fetching or modifying any data."

    def add_arguments(self, parser) -> None:
        parser.add_argument('--universe', type=str, required=True, help='Slug of a seeded universe (e.g., nifty100)')
        parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
        parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD), defaults to today')
        parser.add_argument('--min-coverage', type=float, default=0.90,
                             help='Flag symbols whose bar count / expected trading days falls below this ratio')

    def handle(self, *args, **options) -> None:
        universe_name: str = options['universe']
        start: date = datetime.strptime(options['start'], "%Y-%m-%d").date()
        end: date = datetime.strptime(options['end'], "%Y-%m-%d").date() if options['end'] else date.today()
        min_coverage: float = options['min_coverage']

        try:
            universe = Universe.objects.get(name=universe_name)
        except Universe.DoesNotExist:
            raise CommandError(f"Universe '{universe_name}' not found.")

        symbols = [m.symbol for m in universe.members.select_related('symbol').all()]
        if not symbols:
            raise CommandError(f"Universe '{universe_name}' has no members.")

        weekdays = _count_weekdays(start, end)
        known_holidays = NonTradingDay.objects.filter(date__range=(start, end)).count()
        expected_days = max(weekdays - known_holidays, 1)

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Checking {len(symbols)} symbols in '{universe_name}' ({start} to {end}, "
            f"~{expected_days} expected trading days)..."
        ))
        self.stdout.write(f"  {'Symbol':<16} {'Bars':>6} {'Coverage':>9} {'First':>12} {'Last':>12}")
        self.stdout.write("-" * 62)

        flagged = []
        for symbol in symbols:
            qs = PriceBar.objects.filter(symbol=symbol, date__range=(start, end)).order_by("date")
            count = qs.count()
            first = qs.first()
            last = qs.last()
            coverage = count / expected_days

            flag = ""
            if coverage < min_coverage:
                flag = "  ⚠"
                flagged.append((symbol.ticker, count, coverage, first.date if first else None))

            self.stdout.write(
                f"  {symbol.ticker:<16} {count:>6} {coverage:>8.1%} "
                f"{str(first.date) if first else '—':>12} {str(last.date) if last else '—':>12}{flag}"
            )

        self.stdout.write("-" * 62)
        if flagged:
            self.stdout.write(self.style.WARNING(f"\n{len(flagged)} symbol(s) below {min_coverage:.0%} coverage:"))
            for ticker, count, coverage, first_date in flagged:
                self.stdout.write(f"  - {ticker}: {count} bars ({coverage:.1%}), first bar {first_date or 'none'}")
            self.stdout.write(self.style.WARNING(
                "\nNote: low coverage is expected (not a bug) for symbols listed, demerged, or "
                "renamed after your --start date — check 'first bar' against the actual listing date."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(f"\nAll {len(symbols)} symbols meet the {min_coverage:.0%} threshold."))