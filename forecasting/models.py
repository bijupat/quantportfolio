from django.db import models


class Prediction(models.Model):
    """A cached inference result for one symbol/model/date.

    ``pipeline_version`` exists to solve cache staleness: predictions are
    cached indefinitely per (symbol, model, as_of_date), with no natural
    expiry, so a row computed by an old/buggy version of the inference
    pipeline (e.g. before forecasting.services.training.build_dataset_from_db
    gained its mode="inference" staleness fix) would otherwise sit in this
    table forever and be served from cache as if it were current, silently
    mixed in with predictions computed after the fix. There is no way to
    tell the two apart from any other field on this model — same
    as_of_date, same symbol, same model, same schema.

    forecasting.services.predictor.PIPELINE_VERSION is bumped whenever a
    change to the inference/feature-building logic would make previously
    cached Prediction rows numerically wrong or stale (not for cosmetic
    changes). get_or_predict_bulk only accepts a cached row as a genuine
    cache hit when its pipeline_version matches the current constant;
    otherwise it's treated as missing and recomputed, overwriting the old
    row via update_or_create. Existing rows from before this field existed
    default to "unknown", which can never equal a real version string, so
    they are correctly treated as stale on first read after this change
    rather than requiring a separate one-off data migration to flag them.
    """

    symbol            = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    model             = models.ForeignKey("market_data.TrainedModel", on_delete=models.CASCADE)
    as_of_date        = models.DateField(db_index=True)
    horizon_days      = models.IntegerField(default=30)
    predicted_return  = models.FloatField()
    confidence        = models.FloatField(null=True)
    pipeline_version  = models.CharField(
        max_length=32,
        default="unknown",
        db_index=True,
        help_text="Inference pipeline version that produced this row (see "
                   "forecasting.services.predictor.PIPELINE_VERSION). Rows "
                   "whose version doesn't match the current constant are "
                   "treated as stale and recomputed rather than cache-hit.",
    )
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("symbol", "model", "as_of_date")


class CompositeScore(models.Model):
    symbol            = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    as_of_date        = models.DateField(db_index=True)
    config_hash       = models.CharField(max_length=16, db_index=True)
    composite_score   = models.FloatField()
    transformer_score = models.FloatField()
    transformer_norm  = models.FloatField()
    quality_score     = models.FloatField()
    technical_score   = models.FloatField()
    per_model_scores  = models.JSONField()
    tier              = models.CharField(max_length=10)

    class Meta:
        unique_together = ("symbol", "as_of_date", "config_hash")


class EnsembleRun(models.Model):
    trained_models    = models.ManyToManyField("market_data.TrainedModel")
    weights           = models.JSONField()
    as_of_date        = models.DateField()
    created_at        = models.DateTimeField(auto_now_add=True)


class ScreenerResult(models.Model):
    symbol            = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    as_of_date        = models.DateField(db_index=True)
    s1_trend          = models.BooleanField()
    s2_momentum       = models.BooleanField()
    s3_macd           = models.BooleanField()
    s4_rsi            = models.BooleanField()
    s5_volume         = models.BooleanField()
    s6_drawdown       = models.BooleanField()
    score             = models.IntegerField()
    status            = models.CharField(max_length=20)
    raw_values        = models.JSONField()

    class Meta:
        unique_together = ("symbol", "as_of_date")


class ReportArtifact(models.Model):
    """A generated PDF/Excel/PNG report artifact from a prediction, composite, or screener run.

    ``portfolio`` intentionally references the live ``portfolio.Portfolio``
    model (via string reference to avoid a circular import between the
    ``forecasting`` and ``portfolio`` apps), NOT a local model in this file.

    This app previously defined its own ``Portfolio``/``PortfolioHolding``
    pair that nothing else in the codebase read or wrote — all real
    portfolio persistence has always gone through
    ``portfolio.services.portfolio_service.save_portfolio_to_db``, which
    returns a ``portfolio.models.Portfolio`` instance. ``ReportArtifact.portfolio``
    was declared as a FK to the local (dead) ``forecasting.Portfolio`` class,
    so every call site that did ``ReportArtifact.objects.create(portfolio=saved_portfolio,
    ...)`` with a real, live portfolio was assigning an instance of the wrong
    model type to the FK — Django raises ``ValueError: Cannot assign
    ".../...": "ReportArtifact.portfolio" must be a "Portfolio" instance.``
    the moment ``run_composite --plot`` tries to save its PDF/Excel report.

    The dead ``forecasting.Portfolio``/``PortfolioHolding`` models have been
    removed entirely (see README "Known Limitations & Roadmap") rather than
    left in place, since keeping them around was the direct cause of this bug:
    their mere existence let this FK silently type-check against the wrong
    class instead of failing at import time.

    ``source`` distinguishes which pipeline produced the artifact — the
    Composite Reports page (forecasting/templates/forecasting/reports.html)
    renders both the run_composite ("Run Composite Engine") and run_screener
    ("Run Screener") triggers on one page, feeding a single shared
    "Generated Reports" table. Before this field, a screener's saved Excel
    workbook and a composite's saved Excel workbook were both
    ``kind=XLSX`` with no ``portfolio``/``composite_run`` set on the
    screener's row (it has no portfolio to link), making the two
    indistinguishable in that table except by opening the file. This is a
    plain CharField with choices (not a ForeignKey) since it's a fixed,
    small, code-defined set of pipeline names, not a queryable related
    entity — matches the existing ``kind`` field's own pattern on this
    model. Defaults to COMPOSITE so existing rows (all of which predate
    this field, and were all produced by run_composite) are migrated
    correctly with no separate data migration needed.
    """

    class Source(models.TextChoices):
        COMPOSITE = "composite", "Composite Engine"
        SCREENER = "screener", "Screener"
        PREDICT = "predict", "Single-Model Predict"

    PDF, XLSX, PNG = "pdf", "xlsx", "png"
    kind              = models.CharField(max_length=10, choices=[(PDF, PDF), (XLSX, XLSX), (PNG, PNG)])
    source            = models.CharField(
        max_length=20,
        choices=Source.choices,
        default=Source.COMPOSITE,
        help_text="Which pipeline generated this artifact (composite engine, "
                   "screener, or standalone predict) — see class docstring.",
    )
    portfolio         = models.ForeignKey("portfolio.Portfolio", null=True, on_delete=models.SET_NULL)
    composite_run     = models.ForeignKey(CompositeScore, null=True, on_delete=models.SET_NULL)
    file              = models.FileField(upload_to="reports/")
    created_at        = models.DateTimeField(auto_now_add=True)