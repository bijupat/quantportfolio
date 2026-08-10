from django.db import models

class Symbol(models.Model):
    ticker        = models.CharField(max_length=20, unique=True)
    name          = models.CharField(max_length=200, blank=True)
    exchange      = models.CharField(max_length=10)
    sector        = models.CharField(max_length=100, blank=True)
    is_financial  = models.BooleanField(default=False)
    active        = models.BooleanField(default=True)

    def __str__(self):
        return self.ticker

class Universe(models.Model):
    name          = models.SlugField(unique=True)
    description   = models.CharField(max_length=200, blank=True)

    def __str__(self):
        return self.name

class UniverseMembership(models.Model):
    universe      = models.ForeignKey(Universe, on_delete=models.CASCADE, related_name="members")
    symbol        = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    
    class Meta:
        unique_together = ("universe", "symbol")

    def __str__(self):
        return f"{self.symbol.ticker} -> {self.universe.name}"