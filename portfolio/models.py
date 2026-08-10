from django.db import models
from core.models import Symbol

class Portfolio(models.Model):
    name = models.CharField(max_length=100, unique=True)
    total_capital = models.FloatField(default=0.0)
    strategy = models.CharField(max_length=50, default="composite")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class PortfolioItem(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="items")
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=0)
    purchase_price = models.FloatField(default=0.0)
    allocation_pct = models.FloatField(default=0.0)
    allocation_rs = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.symbol.ticker} ({self.quantity}) in {self.portfolio.name}"