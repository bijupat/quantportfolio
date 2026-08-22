"""forecasting/management/commands/purge_stale_predictions.py

One-off remediation for the Prediction cache-staleness bug: rows cached
under an older forecasting.services.predictor.PIPELINE_VERSION are
indistinguishable from fresh ones to any consumer of get_or_predict_bulk
except by that version tag. Rather than waiting for the lazy per-request
staleness check in get_or_predict_bulk to naturally recompute every stale
row one (model, as_of_date) at a time, this command finds and deletes
them in bulk so the next inference/composite run recomputes cleanly.

Deletes, rather than recomputes, on purpose: recomputing here would need
the same seq_len/horizon/use_sentiment context get_or_predict_bulk has at
call time, which this command doesn't have a natural source for across
arbitrary historical rows. Deleting is safe — any code path that needs a
prediction for that (symbol, model, as_of_date) will recompute it via
get_or_predict_bulk on next access, same as if it had never been cached.

Read-only by default (--dry-run) — always inspect the count before
deleting a potentially large chunk of your Prediction history.
"""

from django.core.management.base import BaseCommand

from forecasting.models import Prediction
from forecasting.services.predictor import PIPELINE_VERSION


class Command(BaseCommand):
    """Reports on and optionally purges Prediction rows from stale pipeline versions."""

    help = "Purges cached Prediction rows computed under an outdated pipeline_version."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            '--model', type=str, default=None,
            help='Restrict to a single TrainedModel name. Defaults to all models.',
        )
        parser.add_argument(
            '--execute', action='store_true',
            help='Actually delete stale rows. Without this flag, only reports the count (dry run).',
        )

    def handle(self, *args, **options) -> None:
        qs = Prediction.objects.exclude(pipeline_version=PIPELINE_VERSION)

        model_name = options['model']
        if model_name:
            qs = qs.filter(model__name=model_name)

        # Break down by the stale version tag actually present, so a person
        # running this can see whether they're looking at genuinely old
        # ("unknown", from before this field existed) rows vs. rows from an
        # intermediate version that was itself superseded.
        version_counts = (
            qs.values_list("pipeline_version", flat=True)
        )
        breakdown: dict[str, int] = {}
        for v in version_counts:
            breakdown[v] = breakdown.get(v, 0) + 1

        total = qs.count()

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Current pipeline version: {PIPELINE_VERSION}"
        ))
        if not total:
            self.stdout.write(self.style.SUCCESS("No stale Prediction rows found. Nothing to do."))
            return

        self.stdout.write(f"Found {total} stale Prediction row(s):")
        for version, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            self.stdout.write(f"  - pipeline_version={version!r}: {count} row(s)")

        if not options['execute']:
            self.stdout.write(self.style.WARNING(
                "\nDry run only — no rows deleted. Re-run with --execute to purge them."
            ))
            return

        deleted_count, _ = qs.delete()
        self.stdout.write(self.style.SUCCESS(
            f"\nDeleted {deleted_count} stale Prediction row(s). "
            f"They will be recomputed under pipeline_version={PIPELINE_VERSION!r} on next access."
        ))