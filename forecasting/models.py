from django.db import models

class Prediction(models.Model):
    symbol            = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    model             = models.ForeignKey("market_data.TrainedModel", on_delete=models.CASCADE)
    as_of_date        = models.DateField(db_index=True)
    horizon_days      = models.IntegerField(default=30)
    predicted_return  = models.FloatField()
    confidence        = models.FloatField(null=True)
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

class Portfolio(models.Model):
    name              = models.CharField(max_length=100)
    created_at        = models.DateTimeField(auto_now_add=True)
    is_active         = models.BooleanField(default=True)

class PortfolioHolding(models.Model):
    portfolio         = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="holdings")
    symbol            = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    quantity          = models.IntegerField()
    purchase_price    = models.DecimalField(max_digits=12, decimal_places=2)
    purchase_date     = models.DateField()
    price_source      = models.CharField(max_length=20)

class ReportArtifact(models.Model):
    PDF, XLSX, PNG = "pdf", "xlsx", "png"
    kind              = models.CharField(max_length=10, choices=[(PDF,PDF),(XLSX,XLSX),(PNG,PNG)])
    portfolio         = models.ForeignKey(Portfolio, null=True, on_delete=models.SET_NULL)
    composite_run     = models.ForeignKey(CompositeScore, null=True, on_delete=models.SET_NULL)
    file              = models.FileField(upload_to="reports/")
    created_at        = models.DateTimeField(auto_now_add=True)