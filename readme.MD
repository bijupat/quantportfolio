````markdown
# QuantPortfolioAI — Django Project README

Welcome to **QuantPortfolioAI**, an institutional-grade quantitative portfolio management and deep learning forecasting platform built on Django. This project integrates market data pipelines, financial quality scoring (Piotroski & Altman), advanced technical indicators (TSI), transformer-based AI forecasting models, multi-model ensembling, and relational portfolio storage.

---

## 📁 Project Structure

```text
quantportfolio/
│
├── core/                        # Core app (Symbols, Universes, base models)
│   ├── models.py                # Symbol, Universe, UniverseMember models
│   └── admin.py
│
├── market_data/                 # Data ingestion, features, and model training
│   ├── management/
│   │   └── commands/            # Custom Django management commands
│   │       ├── train_model.py   # Trains Transformer/Hybrid models
│   │       └── fetch_data.py    # Fetches OHLCV market data & context
│   ├── services/
│   │   ├── prices.py            # Price bar management & dataframes
│   │   ├── quality.py           # Piotroski F-Score & Altman Z-Score calculation
│   │   └── training.py          # Dataset builder from database
│   ├── models.py                # PriceBar, QualityScore, TrainedModel
│   └── admin.py
│
├── forecasting/                 # AI inference, ensembling, and composite scoring
│   ├── management/
│   │   └── commands/
│   │       └── run_composite.py # Executes ensemble predictions & composite ranking
│   ├── services/
│   │   ├── predictor.py         # DB-cached single-model inference
│   │   ├── ensemble.py          # Multi-model weighted ensemble service
│   │   ├── composite.py         # Blends AI, Quality, and Technical layers
│   │   └── reports.py           # ReportLab PDF & openpyxl Excel exporters
│   ├── models.py                # Prediction and ReportArtifact models
│   └── admin.py
│
├── portfolio/                   # Portfolio execution and database storage
│   ├── management/
│   │   └── commands/
│   │       └── manage_portfolio.py # Manages, lists, and views DB portfolios
│   ├── services/
│   │   └── portfolio_service.py # Saves and retrieves structured portfolios
│   └── models.py                # Portfolio and PortfolioItem models
│
├── config/                      # Django project configuration
│   ├── settings.py
│   ├── urls.py
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

Initialize the SQLite database (or your production DB backend) and create the schema:

```cmd
python manage.py makemigrations core market_data forecasting portfolio
python manage.py migrate

```

---

## 📊 Data Pipeline & Management Commands

QuantPortfolioAI relies on custom Django management commands to handle asynchronous data fetching, model training, composite inference, and portfolio tracking.

### 1. Fetch Market Data & Populate Universes

Downloads historical OHLCV pricing data, constructs market contexts, and seeds database symbols and universes:

```cmd
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

3. **Automated Artifact Generation:** Execution commands automatically produce publication-ready **ReportLab PDF reports** and **multi-sheet openpyxl Excel workbooks** stored securely as database artifacts.

---
