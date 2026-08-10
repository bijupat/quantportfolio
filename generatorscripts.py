import os
import textwrap

def generate_django_project():
    base_dir = "quantportfolio_django"
    
    # Dictionary containing the file paths and their respective boilerplate code
    files_to_create = {
        "requirements.txt": "Django>=5.0\npsycopg2-binary\ncelery\nredis\n",
        "manage.py": textwrap.dedent("""\
            #!/usr/bin/env python
            import os
            import sys

            def main():
                os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
                try:
                    from django.core.management import execute_from_command_line
                except ImportError as exc:
                    raise ImportError("Couldn't import Django.") from exc
                execute_from_command_line(sys.argv)

            if __name__ == '__main__':
                main()
        """),
        "config/__init__.py": "",
        "config/settings.py": textwrap.dedent("""\
            import os
            from pathlib import Path

            BASE_DIR = Path(__file__).resolve().parent.parent

            SECRET_KEY = 'django-insecure-replace-this-in-production'
            DEBUG = True
            ALLOWED_HOSTS = []

            INSTALLED_APPS = [
                'django.contrib.admin',
                'django.contrib.auth',
                'django.contrib.contenttypes',
                'django.contrib.sessions',
                'django.contrib.messages',
                'django.contrib.staticfiles',
                
                # Custom Apps
                'core',
                'market_data',
                'forecasting',
            ]

            MIDDLEWARE = [
                'django.middleware.security.SecurityMiddleware',
                'django.contrib.sessions.middleware.SessionMiddleware',
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
                'django.contrib.auth.middleware.AuthenticationMiddleware',
                'django.contrib.messages.middleware.MessageMiddleware',
                'django.middleware.clickjacking.XFrameOptionsMiddleware',
            ]

            ROOT_URLCONF = 'config.urls'

            DATABASES = {
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': BASE_DIR / 'db.sqlite3',
                }
            }

            MEDIA_URL = '/media/'
            MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')
            STATIC_URL = '/static/'
        """),
        "config/urls.py": textwrap.dedent("""\
            from django.contrib import admin
            from django.urls import path
            
            urlpatterns = [
                path('admin/', admin.site.urls),
            ]
        """),
        "config/celery.py": "",
        "config/wsgi.py": "",
        "config/asgi.py": "",
        
        # Core App
        "core/__init__.py": "",
        "core/apps.py": textwrap.dedent("""\
            from django.apps import AppConfig
            
            class CoreConfig(AppConfig):
                default_auto_field = 'django.db.models.BigAutoField'
                name = 'core'
        """),
        "core/models.py": textwrap.dedent("""\
            from django.db import models

            class Symbol(models.Model):
                ticker        = models.CharField(max_length=20, unique=True)
                name          = models.CharField(max_length=200, blank=True)
                exchange      = models.CharField(max_length=10)
                sector        = models.CharField(max_length=100, blank=True)
                is_financial  = models.BooleanField(default=False)
                active        = models.BooleanField(default=True)

            class Universe(models.Model):
                name          = models.SlugField(unique=True)
                description   = models.CharField(max_length=200, blank=True)

            class UniverseMembership(models.Model):
                universe      = models.ForeignKey(Universe, on_delete=models.CASCADE, related_name="members")
                symbol        = models.ForeignKey(Symbol, on_delete=models.CASCADE)
                
                class Meta:
                    unique_together = ("universe", "symbol")
        """),
        "core/admin.py": "from django.contrib import admin\nfrom .models import Symbol, Universe\n\nadmin.site.register(Symbol)\nadmin.site.register(Universe)\n",
        "core/management/commands/__init__.py": "",
        "core/management/commands/seed_universes.py": "",
        
        # Market Data App
        "market_data/__init__.py": "",
        "market_data/apps.py": textwrap.dedent("""\
            from django.apps import AppConfig
            
            class MarketDataConfig(AppConfig):
                default_auto_field = 'django.db.models.BigAutoField'
                name = 'market_data'
        """),
        "market_data/models.py": textwrap.dedent("""\
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
        """),
        "market_data/admin.py": "from django.contrib import admin\nfrom .models import TrainedModel\n\nadmin.site.register(TrainedModel)\n",
        "market_data/tasks.py": "",
        "market_data/services/__init__.py": "",
        "market_data/services/prices.py": textwrap.dedent("""\
            from datetime import date
            from django.db.models import QuerySet
            from core.models import Symbol
            from market_data.models import PriceBar

            def get_price_bars(symbol: Symbol, start: date, end: date) -> QuerySet[PriceBar]:
                # Implement DB-first fetch logic here
                pass
        """),
        "market_data/services/indicators.py": "",
        "market_data/services/sentiment.py": "",
        "market_data/services/fundamentals.py": "",
        "market_data/services/quality.py": "",
        "market_data/services/market_context.py": "",
        "market_data/services/training.py": "",
        "market_data/management/commands/__init__.py": "",
        "market_data/management/commands/fetch_prices.py": "",
        "market_data/management/commands/fetch_sentiment.py": "",
        "market_data/management/commands/backfill_indicators.py": "",
        "market_data/management/commands/train_model.py": "",
        "market_data/management/commands/train_hybrid_model.py": "",
        "market_data/management/commands/migrate_legacy_model.py": "",
        
        # Forecasting App
        "forecasting/__init__.py": "",
        "forecasting/apps.py": textwrap.dedent("""\
            from django.apps import AppConfig
            
            class ForecastingConfig(AppConfig):
                default_auto_field = 'django.db.models.BigAutoField'
                name = 'forecasting'
        """),
        "forecasting/models.py": textwrap.dedent("""\
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
        """),
        "forecasting/admin.py": "from django.contrib import admin\nfrom .models import Prediction\n\nadmin.site.register(Prediction)\n",
        "forecasting/services/__init__.py": "",
        "forecasting/services/predictor.py": "",
        "forecasting/services/ensemble.py": "",
        "forecasting/services/composite.py": "",
        "forecasting/services/screener.py": "",
        "forecasting/services/portfolio.py": "",
        "forecasting/services/reports.py": "",
        "forecasting/management/commands/__init__.py": "",
        "forecasting/management/commands/predict.py": "",
        "forecasting/management/commands/run_ensemble.py": "",
        "forecasting/management/commands/run_composite.py": "",
        "forecasting/management/commands/run_screener.py": "",
        "forecasting/management/commands/rebalance_portfolio.py": "",
    }

    print(f"Initializing project build in directory: {base_dir}...")
    
    for file_path, content in files_to_create.items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {full_path}")

    print("\\nSuccess! Your QuantPortfolioAI Django project folder is ready for development.")

if __name__ == "__main__":
    generate_django_project()