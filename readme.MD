Here's the updated `readme.MD` reflecting the custom `User` model, role-based permissions, and app config changes made in this conversation.

````markdown
# QuantPortfolioAI — Django Project README

Welcome to **QuantPortfolioAI**, an institutional-grade quantitative portfolio management and deep learning forecasting platform built on Django. This project integrates market data pipelines, financial quality scoring (Piotroski & Altman), advanced technical indicators (TSI), transformer-based AI forecasting models, multi-model ensembling, relational portfolio storage, and role-based user management.

---

## 📁 Project Structure

```text
quantportfolio/
│
├── core/                        # Core app (Users, Symbols, Universes, base models)
│   ├── models.py                # User (AbstractUser), Symbol, Universe, UniverseMember
│   ├── forms.py                 # CustomUserCreationForm, CustomUserChangeForm (admin)
│   ├── permissions.py           # require_role() — gates role-restricted actions
│   ├── threading_utils.py       # Background thread / management command runner
│   ├── apps.py                  # CoreConfig
│   └── admin.py                 # User, Symbol, Universe admin registration
│
├── market_data/                 # Data ingestion, features, and model training
│   ├── management/
│   │   └── commands/            # Custom Django management commands
│   │       ├── train_model.py         # Trains standard Transformer models
│   │       ├── train_hybrid_model.py  # Trains TripleBranchHybridModel
│   │       ├── fetch_prices.py        # Fetches OHLCV market data for a seeded universe
│   │       ├── fetch_sentiment.py     # Fetches/scores news sentiment
│   │       ├── backfill_indicators.py # Backfills technical indicator snapshots
│   │       └── migrate_legacy_model.py# Imports legacy .h5 weights into TrainedModel
│   ├── services/
│   │   ├── prices.py            # Price bar management & dataframes (DB-first, yfinance fallback)
│   │   ├── indicators.py        # Technical indicator computation (RSI, MACD, ADX, etc.)
│   │   ├── quality.py           # Piotroski F-Score & Altman Z-Score DB-first fetcher
│   │   ├── sentiment.py         # DB-first news sentiment fetcher
│   │   ├── market_context.py    # Index-level market context features
│   │   ├── fundamentals.py      # Raw fundamental snapshot placeholder
│   │   └── training.py          # Dataset builder & training orchestrators (standard + hybrid)
│   ├── models.py                # PriceBar, TechnicalIndicatorSnapshot, NewsSentiment,
│   │                             # MarketContext, FundamentalSnapshot, QualityScore,
│   │                             # TrainedModel, TrainingRun
│   ├── apps.py                  # MarketDataConfig
│   └── admin.py
│
├── forecasting/                 # AI inference, ensembling, and composite scoring
│   ├── management/
│   │   └── commands/
│   │       ├── run_composite.py # Executes ensemble predictions & composite ranking
│   │       ├── run_ensemble.py  # Multi-model ensemble inference only
│   │       ├── run_screener.py  # Price/volume technical screener
│   │       └── predict.py       # Single-model inference & ranked portfolio output
│   ├── services/
│   │   ├── predictor.py         # DB-cached single-model inference
│   │   ├── ensemble.py          # Multi-model weighted ensemble service
│   │   ├── composite.py         # Blends AI, Quality, and Technical layers
│   │   ├── screener.py          # Technical signal screening (trend/momentum/volume)
│   │   └── reports.py           # ReportLab PDF & openpyxl Excel exporters
│   ├── models.py                # Prediction, CompositeScore, EnsembleRun, ScreenerResult,
│   │                             # ReportArtifact
│   ├── apps.py                  # ForecastingConfig
│   └── admin.py
│
├── portfolio/                   # Portfolio execution and database storage
│   ├── management/
│   │   └── commands/
│   │       └── manage_portfolio.py # Lists, views, and imports DB portfolios
│   ├── services/
│   │   └── portfolio_service.py # Saves and retrieves structured portfolios
│   ├── models.py                # Portfolio (owner-linked), PortfolioItem
│   ├── apps.py                  # PortfolioConfig
│   └── admin.py                 # Portfolio, PortfolioItem admin (with holdings inline)
│
├── config/                      # Django project configuration
│   ├── settings.py              # AUTH_USER_MODEL = "core.User"; explicit AppConfig paths
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── manage.py                    # Django management script
└── requirements.txt             # Project dependencies
```
````

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Virtual Environment

```cmd
git clone <repository_url>
cd quantportfolio
python -m venv .venv
.venv\Scripts\activate

```

### 2. Install Dependencies

Install all required Python packages (including TensorFlow, Pandas, ReportLab, and Openpyxl):

```cmd
pip install -r requirements.txt

```

### 3. Apply Database Migrations

Initialize the PostgreSQL database schema. `core` must migrate first since `AUTH_USER_MODEL` (`core.User`) is referenced by other apps' foreign keys:

```cmd
python manage.py makemigrations core
python manage.py makemigrations market_data forecasting portfolio
python manage.py migrate
python manage.py createsuperuser

```

---

## 👤 User Management

QuantPortfolioAI uses a **custom `User` model** (`core.User`, subclassing Django's `AbstractUser`) rather than a separate profile table, so role and preference fields are available on `request.user` with no extra query.

- **`role`** — `analyst` (default) or `admin`. Gates access to training and portfolio-execution commands.
- **`default_universe`** — FK to `Universe`, pre-selected in dashboard forms for that user.
- **`core/permissions.py`** — `require_role(request, *allowed_roles)` helper for gating views/commands by role.
- **`core/forms.py`** — `CustomUserCreationForm` / `CustomUserChangeForm`, wired into Django admin via `core/admin.py`.

`AUTH_USER_MODEL = "core.User"` is set in `config/settings.py` and must remain in place — this is a project-start decision that cannot be changed after the first migration without a full reset.

`portfolio.Portfolio` has a nullable `owner` FK to `settings.AUTH_USER_MODEL`. Ownership wiring into `portfolio_service.py` and the management commands is planned once the web/HTMX view layer lands (CLI-generated portfolios remain ownerless/system-generated until then).

---

## 📊 Data Pipeline & Management Commands

QuantPortfolioAI relies on custom Django management commands to handle asynchronous data fetching, model training, composite inference, and portfolio tracking.

### 1. Seed Universes & Fetch Market Data

Seeds predefined stock universes and symbol memberships, then downloads historical OHLCV pricing data for a universe:

```cmd
python manage.py seed_universes
python manage.py fetch_prices --universe nifty50 --start 2020-01-01

```

### 2. Train AI Models

Trains a standard transformer or hybrid architecture on your historical database features:

```cmd
python manage.py train_model --model-name test_transformer --universe temp --epochs 5

```

### 3. Run Composite & Ensemble Predictions

Combines transformer predictions with fundamental quality scores (Piotroski/Altman) and technical momentum (TSI), prints the ranked output to the console, saves database records, and generates reports:

```cmd
python manage.py run_composite --models test_transformer --model-weights 1.0 --universe temp --amount 100000 --plot

```

### 4. Manage Stored Portfolios

List all portfolios saved directly to the relational database:

```cmd
python manage.py manage_portfolio --list

```

View the exact holdings, quantities, purchase prices, and allocation breakdown of a specific database portfolio:

```cmd
python manage.py manage_portfolio --view composite_test_trans_20260809

```

---

## ⚙️ Core Technical Architecture

1. **DB-First Architecture:** Rather than reading loose CSV files during runtime, historical prices (`PriceBar`), computed features, quality metrics (`QualityScore`), and model outputs (`Prediction`) are fully cached and queryable via Django ORM models.
2. **Three-Layer Composite Scoring:**

- **Transformer Layer:** Predicts expected 30-day returns using temporal attention layers.
- **Quality Factor Layer:** Evaluates balance sheet strength via Piotroski F-Score and Altman Z-Score models.
- **Technical Indicator Layer:** Measures trend momentum using the True Strength Index (TSI).

3. **Role-Based Access:** A custom `core.User` model (`AbstractUser` subclass) carries `role` (`analyst` / `admin`) directly on the user object, gating training and portfolio-execution actions without an extra profile-table join.
4. **Automated Artifact Generation:** Execution commands automatically produce publication-ready **ReportLab PDF reports** and **multi-sheet openpyxl Excel workbooks** stored securely as database artifacts.

---

```

A few notes on what I changed vs. your last uploaded `readme.MD`:

- Project structure tree now lists every file actually present in your uploads (previously it only showed a handful of representative files per app) — including `apps.py` for all four apps, `core/forms.py`, `core/permissions.py`, `core/threading_utils.py`.
- Added a **User Management** section documenting the `AbstractUser` decision, `role`, `default_universe`, and the `owner` field status on `Portfolio`.
- Fixed the migration command block to show `core` migrating first (required since other apps' FKs will eventually reference `core.User`/`core.Universe`).
- Fixed a command-name mismatch that existed in your original file: the "Fetch Market Data" section referenced `fetch_data`, but your actual command file is `fetch_prices.py` — corrected to match.
- Added `seed_universes` as a prerequisite step before `fetch_prices`, since `fetch_prices.py` raises `CommandError` if the universe isn't already seeded.
- Added item 3 (Role-Based Access) to the Core Technical Architecture list.
```
