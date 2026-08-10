from django.db import models

class Symbol(models.Model):
    ticker = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    sector = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.ticker

class Universe(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    symbols = models.ManyToManyField(Symbol, through='UniverseMember', related_name='universes')

    def __str__(self):
        return self.name

class UniverseMember(models.Model):
    universe = models.ForeignKey(Universe, on_delete=models.CASCADE, related_name='members')
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('universe', 'symbol')

    def __str__(self):
        return f"{self.symbol.ticker} in {self.universe.name}"