from django.db import models
import json
from django.db import models

class PriceBar(models.Model):
    symbol   = models.ForeignKey("core.Symbol", on_delete=models.CASCADE, related_name="bars")
    date     = models.DateField(db_index=True)
    open     = models.DecimalField(max_digits=12, decimal_places=4)
    high     = models.DecimalField(max_digits=12, decimal_places=4)
    low      = models.DecimalField(max_digits=12, decimal_places=4)
    close    = models.DecimalField(max_digits=12, decimal_places=4)
    volume   = models.BigIntegerField()

    class Meta:
        unique_together = ("symbol", "date")
        indexes = [models.Index(fields=["symbol", "date"])]

class TechnicalIndicatorSnapshot(models.Model):
    symbol   = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    date     = models.DateField(db_index=True)
    values   = models.JSONField()

    class Meta:
        unique_together = ("symbol", "date")

class NewsSentiment(models.Model):
    symbol         = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    date           = models.DateField(db_index=True)
    sentiment_score = models.FloatField()
    positive_count  = models.FloatField()
    negative_count  = models.FloatField()
    news_volume     = models.FloatField()
    backend_used   = models.CharField(max_length=20)
    fetched_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("symbol", "date")

class MarketContext(models.Model):
    index_ticker   = models.CharField(max_length=20)
    date           = models.DateField(db_index=True)
    values         = models.JSONField()

    class Meta:
        unique_together = ("index_ticker", "date")

class FundamentalSnapshot(models.Model):
    symbol         = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    fiscal_period  = models.CharField(max_length=10)
    roe            = models.FloatField(null=True)
    roa            = models.FloatField(null=True)
    pe             = models.FloatField(null=True)
    revenue_growth = models.FloatField(null=True)
    debt_equity    = models.FloatField(null=True)
    fcf            = models.FloatField(null=True)
    fetched_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("symbol", "fiscal_period")

class QualityScore(models.Model):
    symbol           = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    date             = models.DateField(db_index=True)
    piotroski_score  = models.IntegerField(null=True)
    altman_z         = models.FloatField(null=True)
    excluded_sector  = models.BooleanField(default=False)
    composite_score  = models.FloatField()

    class Meta:
        unique_together = ("symbol", "date")



class TrainedModel(models.Model):
    STANDARD, HYBRID = "standard", "triple_branch_hybrid"
    MODEL_TYPES = [(STANDARD, "QuantTransformer"), (HYBRID, "TripleBranchHybridModel")]

    name           = models.SlugField(unique=True)
    model_type     = models.CharField(max_length=30, choices=MODEL_TYPES, default=STANDARD)
    seq_len        = models.IntegerField()
    d_model        = models.IntegerField()
    n_heads        = models.IntegerField()
    n_layers       = models.IntegerField()
    n_features     = models.IntegerField()
    weights_file   = models.FileField(upload_to="model_weights/")
    arch_json      = models.JSONField()
    scalers_json   = models.JSONField()
    eval_json      = models.JSONField(null=True)
    trained_on     = models.ForeignKey("core.Universe", null=True, on_delete=models.SET_NULL)
    created_at     = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"

    def build_keras_model(self):
        """
        Instantiates the uncompiled Keras architecture and loads the weights.
        """
        from model_transformer import build_transformer_model, _load_weights_shape_matched
        from hybrid_main import build_hybrid_model

        if self.model_type == self.HYBRID:
            model = build_hybrid_model(
                seq_len=self.seq_len,
                n_ts_features=self.arch_json.get("n_ts_features", 50),
                n_financial_features=self.arch_json.get("n_financial_features", 8),
                n_sentiment_features=self.arch_json.get("n_sentiment_features", 4),
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                lstm_units=self.arch_json.get("lstm_units", 64),
            )
        else:
            model = build_transformer_model(
                seq_len=self.seq_len,
                n_features=self.n_features,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
            )

        _load_weights_shape_matched(model, self.weights_file.path)
        
        model.seq_len = self.seq_len
        model.n_features = self.n_features
        
        return model

class TrainingRun(models.Model):
    model          = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name="runs")
    status         = models.CharField(max_length=20, default="pending")
    config         = models.JSONField()
    started_at     = models.DateTimeField(null=True)
    finished_at    = models.DateTimeField(null=True)
    log_output     = models.TextField(blank=True)