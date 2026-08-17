\`\`\`markdown

# QuantPortfolioAI — Django Project README

Welcome to **QuantPortfolioAI**, an institutional-grade quantitative portfolio management and deep
learning forecasting platform built on Django, focused on NSE/BSE equities. The project integrates
market data pipelines, financial quality scoring (Piotroski & Altman), technical indicators (RSI,
MACD, ADX, TSI, and more), transformer / hybrid deep-learning forecasting models, multi-model
ensembling, a rule-based technical screener, relational portfolio storage with ownership, and an
HTMX-driven web UI with background job tracking.

---

## 📁 Project Structure

\`\`\`text
quantportfolio/
│
├── core/ # Core app — auth, symbols, universes, background jobs
│ ├── models.py # User (AbstractUser), Symbol, Universe, UniverseMember, JobRun
│ ├── forms.py # CustomUserCreationForm, CustomUserChangeForm (admin)
│ ├── permissions.py # require_role() — gates role-restricted actions
│ ├── mixins.py # AdminRequiredMixin — class-based view guard (403, not redirect loop)
│ ├── threading_utils.py # launch_tracked_command() — background mgmt-command runner + JobRun tracking
│ ├── views.py # DashboardView, JobStatusView (shared HTMX polling endpoint)
│ ├── urls.py
│ ├── admin.py
│ ├── apps.py # CoreConfig
│ ├── management/commands/
│ │ └── seed_universes.py # Seeds ~20 predefined universes + symbol memberships
│ └── templates/core/
│ ├── index.html # Dashboard landing page
│ └── \_job_status.html # Shared HTMX-polled JobRun status partial (used by every app)
│
├── market_data/ # Data ingestion, indicators, market context, model training
│ ├── models.py # PriceBar, TechnicalIndicatorSnapshot, NewsSentiment, MarketContext,
│ │ # FundamentalSnapshot, QualityScore, TrainedModel, TrainingRun, NonTradingDay
│ ├── views.py # FetchDataView, SeedUniversesTriggerView, FetchPricesTriggerView
│ ├── urls.py
│ ├── admin.py
│ ├── apps.py # MarketDataConfig
│ ├── services/
│ │ ├── prices.py # get_price_bars (DB-first + yfinance fallback), sync_market_calendar,
│ │ │ # dataframe_from_bars, backfill_symbol_metadata
│ │ ├── indicators.py # calculate_technical_indicators, build_advanced_indicators, get_indicators
│ │ │ # FEATURE_COLUMNS / MARKET_CONTEXT_COLS / FINANCIAL_FEATURES / SENTIMENT_FEATURES
│ │ ├── market_context.py # MARKET_INDICES (^NSEI, ^BSESN), \_safe_index_name, get_market_context,
│ │ │ # get_market_context_features — combined, prefixed multi-index features
│ │ ├── quality.py # get_quality_scores_bulk — wraps piotroski.py + altman.py, DB-cached
│ │ ├── sentiment.py # get_sentiment — wraps news_sentiment.py, DB-cached
│ │ ├── fundamentals.py # get_fundamentals — placeholder for raw P/E, ROE, D/E screening
│ │ └── training.py # build_dataset_from_db, train_standard_model_service, train_hybrid_model_service
│ ├── management/commands/
│ │ ├── train_model.py # Trains a standard QuantTransformer
│ │ ├── train_hybrid_model.py # Trains a TripleBranchHybridModel
│ │ ├── fetch_prices.py # Fetches OHLCV for a pre-seeded universe; syncs holiday calendar first
│ │ ├── migrate_legacy_model.py # Imports legacy .h5 weights + JSON metadata into TrainedModel
│ │ ├── verify_price_coverage.py # Read-only PriceBar coverage/gap report for a universe
│ │ ├── repair_price_history.py # Remediates the NonTradingDay cache-poisoning bug (see below)
│ │ ├── fetch_sentiment.py # ⚠ empty stub — not yet implemented (sentiment is currently fetched
│ │ │ # lazily and on-demand via market_data.services.sentiment.get_sentiment)
│ │ └── backfill_indicators.py # ⚠ empty stub — not yet implemented
│ └── templates/market_data/
│ └── fetch.html # Data Pipeline trigger page (seed_universes + fetch_prices)
│
├── forecasting/ # AI inference, ensembling, composite scoring, screening, reports
│ ├── models.py # Prediction, CompositeScore, EnsembleRun, ScreenerResult, ReportArtifact
│ │ # (also still defines a legacy Portfolio/PortfolioHolding pair — unused;
│ │ # superseded by portfolio.models.Portfolio/PortfolioItem, see Roadmap)
│ ├── views.py # TrainModelView/Trigger, TrainHybridModelTrigger, ReportsView, RunCompositeTrigger
│ ├── urls.py
│ ├── admin.py
│ ├── apps.py # ForecastingConfig
│ ├── services/
│ │ ├── predictor.py # get_or_predict_bulk — DB-cached single-model Keras inference
│ │ ├── ensemble.py # run_ensemble_layer — multi-model weighted ensemble
│ │ ├── composite.py # get_technical_scores (TSI), compute_composite_scores, compute_config_hash,
│ │ │ # save_composite_scores_to_db, build_composite_portfolio
│ │ ├── screener.py # get_or_screen — 6-signal trend/momentum/volume screener, DB-cached
│ │ ├── reports.py # save_portfolio_report_pdf (ReportLab), save_score_breakdown_excel (openpyxl)
│ │ └── portfolio.py # ⚠ empty stub — not yet implemented
│ ├── management/commands/
│ │ ├── run_composite.py # Ensemble + composite scoring + portfolio build + PDF/Excel reports
│ │ ├── run_screener.py # Price/volume technical screener (trend, momentum, MACD, RSI, volume, drawdown)
│ │ ├── predict.py # Single-model inference & ranked portfolio (console + optional PDF)
│ │ ├── run_ensemble.py # ⚠ empty stub — not yet implemented (ensemble-only, no composite/quality layer)
│ │ └── rebalance_portfolio.py # ⚠ empty stub — not yet implemented
│ └── templates/forecasting/
│ ├── train.html # Model Training page (standard + hybrid triggers, registered models table)
│ └── reports.html # Composite Reports page (run trigger, generated report downloads)
│
├── portfolio/ # Portfolio persistence, ownership, and read-only web views
│ ├── models.py # Portfolio (owner FK to core.User), PortfolioItem
│ ├── views.py # PortfolioListView, PortfolioDetailView (ownership/visibility-filtered)
│ ├── urls.py
│ ├── admin.py # Portfolio/PortfolioItem admin, with holdings inline
│ ├── apps.py # PortfolioConfig
│ ├── services/
│ │ └── portfolio_service.py # save_portfolio_to_db(owner=...), load_portfolio_from_csv_to_db
│ ├── management/commands/
│ │ └── manage_portfolio.py # --list / --view <name> / --load-csv <path> --name <name>
│ └── templates/portfolio/
│ ├── list.html
│ └── detail.html
│
├── config/ # Django project configuration
│ ├── settings.py # AUTH_USER_MODEL = "core.User"; explicit AppConfig paths; NEWSAPI_KEY
│ ├── urls.py
│ ├── asgi.py
│ └── wsgi.py
│
├── templates/ # Project-level shared templates
│ ├── base.html # Bootstrap 5 + HTMX shell, role-aware nav
│ └── registration/login.html
│
├── metrics/ # Legacy training-run artifacts (JSON), read by migrate_legacy_model
│ ├── transformer_banking_metrics.json
│ ├── transformer_banking_eval.json
│ └── transformer_banking_scalers.json
│
├── util.py # Logging, JSON/dataframe helpers, sequence builder — used by services/training.py
├── news_sentiment.py # FinBERT / VADER / keyword sentiment scoring — used by services/sentiment.py
├── piotroski.py # Piotroski F-Score — used by market_data/services/quality.py
├── altman.py # Altman Z-Score — used by market_data/services/quality.py
├── model_transformer.py # QuantTransformer + TripleBranchHybridModel Keras architectures, save/load helpers
├── portfolio_optimizer.py # rank_stocks, construct_portfolio, backtest_strategy, walk_forward_splits
│
├── manage.py
├── requirements.txt
└── .gitignore
\`\`\`

> **Note on root-level modules:** `hybrid_main.py`, `generatorscripts.py`, `data_loader.py`,
> `feature_engineering.py`, `financial_indicators.py`, and `universes.py` from the original
> pre-Django prototype have been **deleted** — their functionality now lives entirely in the
> `market_data`/`forecasting` service layers. The remaining root-level files (`util.py`,
> `news_sentiment.py`, `piotroski.py`, `altman.py`, `model_transformer.py`,
> `portfolio_optimizer.py`) are **intentionally retained** as shared utility/model-definition
> modules — they're actively imported by the Django services (e.g.
> `market_data/services/quality.py` imports `piotroski.py` and `altman.py`), not legacy cruft.

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Virtual Environment

\`\`\`cmd
git clone <repository_url>
cd quantportfolio
python -m venv .venv
.venv\Scripts\activate
\`\`\`

### 2. Install Dependencies

Install all required Python packages (including TensorFlow, Pandas, ReportLab, and Openpyxl):

\`\`\`cmd
pip install -r requirements.txt
\`\`\`

### 3. Configure Environment

Create a `.env` file in the project root (read via `django-environ`):

\`\`\`
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=postgres://user:password@localhost:5432/quantportfolio
NEWSAPI_KEY= # optional — leave blank to fall back to synthetic news headlines
\`\`\`

### 4. Apply Database Migrations

Initialize the PostgreSQL database schema. `core` must migrate first since `AUTH_USER_MODEL`
(`core.User`) is referenced by other apps' foreign keys:

\`\`\`cmd
python manage.py makemigrations core
python manage.py makemigrations market_data forecasting portfolio
python manage.py migrate
python manage.py createsuperuser
\`\`\`

---

## 👤 User Management

QuantPortfolioAI uses a **custom `User` model** (`core.User`, subclassing Django's `AbstractUser`)
rather than a separate profile table, so role and preference fields are available on
`request.user` with no extra query.

- **`role`** — `analyst` (default) or `admin`. Gates access to data-pipeline, training, and
  composite-run trigger views (`core.mixins.AdminRequiredMixin`), and to management-command-backed
  actions via `core/permissions.py::require_role()`.
- **`default_universe`** — FK to `Universe`, pre-selected in dashboard forms for that user.

`AUTH_USER_MODEL = "core.User"` is set in `config/settings.py` and must remain in place — this is
a project-start decision that cannot be changed after the first migration without a full reset.

**Portfolio ownership is wired end-to-end.** `portfolio.Portfolio.owner` is a nullable FK to
`settings.AUTH_USER_MODEL`. `run_composite --user-id <id>` (and the "Run Composite Engine" web
form, which always passes `request.user.id`) sets the owner on every DB-saved portfolio via
`portfolio_service.save_portfolio_to_db(owner=...)`. Visibility is then enforced in
`portfolio/views.py::_visible_portfolios_qs()`:

- **Admins** see every portfolio.
- **Analysts** see their own portfolios, ownerless/system-generated ones (e.g. CLI runs without
  `--user-id`, or legacy CSV imports), and any `strategy="composite_ai"` portfolio regardless of
  which admin triggered the run — composite-engine output is treated as shared analytical
  content, not personal data, even though `owner` still records who ran the job for audit purposes.

---

## 🖥️ Web UI & Background Jobs

Every long-running pipeline step is triggered from a Bootstrap 5 + HTMX page and executed via
`core.threading_utils.launch_tracked_command()`, which:

1. Creates a `core.models.JobRun` row (`status=PENDING`) and returns it immediately.
2. Spawns a daemon thread that runs the underlying management command with `call_command(...)`,
   capturing stdout/stderr.
3. Updates the `JobRun` row to `RUNNING` → `SUCCESS`/`FAILED` as it progresses.

The triggering view renders `core/templates/core/_job_status.html`, which polls
`core:job_status` (`core.views.JobStatusView`) every 2 seconds via `hx-trigger="load delay:2s"`
until `job.is_finished` — at which point it stops polling and shows the captured output/error.
This single shared partial/endpoint is reused by every app, so a new triggerable command only
needs its own trigger view + form; the polling/status UI is free.

| Page              | URL                     | Guard                                                                              | Triggers                                       |
| ----------------- | ----------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| Data Pipeline     | `/market/fetch/`        | Admin only                                                                         | `seed_universes`, `fetch_prices`               |
| Model Training    | `/forecasting/train/`   | Admin only                                                                         | `train_model`, `train_hybrid_model`            |
| Composite Reports | `/forecasting/reports/` | Composite-run trigger is Admin only; page itself is visible to all logged-in users | `run_composite`                                |
| Portfolios        | `/portfolio/`           | Any logged-in user                                                                 | (read-only — lists/views DB-stored portfolios) |

---

## 📊 Management Commands

### Core

\`\`\`cmd
python manage.py seed_universes
\`\`\`
Seeds ~20 predefined universes (`nifty50`, `nifty100`, `nifty500`, sector universes, custom
watchlists, etc.) and their symbol memberships. Safe to re-run — existing symbols/universes are
kept in sync rather than duplicated.

### Market Data

\`\`\`cmd
python manage.py fetch_prices --universe nifty50 --start 2020-01-01
\`\`\`
Fetches OHLCV price bars for every symbol in a pre-seeded universe. Syncs the shared market
holiday calendar from `^NSEI` first, then backfills symbol metadata (name/sector) and price
history per symbol. **Requires `seed_universes` to have been run for that universe.**

\`\`\`cmd
python manage.py train_model --model-name test_transformer --universe temp --epochs 5
python manage.py train_hybrid_model --model-name test_hybrid --universe temp --epochs 10
\`\`\`
Trains a standard `QuantTransformer` or a `TripleBranchHybridModel` on database-cached OHLCV,
technical indicators, combined market context, and (optionally) sentiment, then registers the
result as a `TrainedModel` row with its weights, architecture, scalers, and eval metrics.

\`\`\`cmd
python manage.py migrate_legacy_model --name transformer_banking --universe banking
\`\`\`
Imports a pre-Django `.weights.h5` file plus its `_arch.json`/`_scalers.json`/`_eval.json`
companions from `models/`/`metrics/` into a `TrainedModel` row.

\`\`\`cmd
python manage.py verify_price_coverage --universe nifty100 --start 2015-01-01 --min-coverage 0.9
\`\`\`
**Read-only** diagnostic — reports per-symbol `PriceBar` row counts, date ranges, and coverage
gaps for a universe. Flags symbols below the coverage threshold without fetching or modifying
anything; low coverage is expected for stocks listed/renamed after `--start`.

\`\`\`cmd
python manage.py repair_price_history --universe nifty100 --start 2015-01-01
\`\`\`
One-off remediation for the `NonTradingDay` cache-poisoning bug (see **Architecture** below):
wipes and rebuilds the shared holiday cache from `^NSEI`, then re-fetches every symbol in the
given universe(s) so previously wrongly-skipped trading days get backfilled. Idempotent.

> `fetch_sentiment` and `backfill_indicators` exist as empty command stubs and are **not yet
> implemented** — sentiment and indicators are currently computed lazily and DB-cached on demand
> (via `get_sentiment()` / `get_indicators()`) whenever `train_model`, `train_hybrid_model`, or
> `run_composite` need them, so there's no standalone CLI for pre-warming them yet.

### Forecasting

\`\`\`cmd
python manage.py run_composite --models test_transformer --model-weights 1.0 --universe temp --amount 100000 --plot
\`\`\`
Blends transformer predictions (ensemble layer), fundamental quality scores (Piotroski/Altman),
and technical momentum (TSI) into a ranked, weighted portfolio. Prints the ranking to console,
persists `CompositeScore` rows (idempotent per `config_hash` + `as_of_date`), and — with
`--plot` — saves the portfolio to the DB (`portfolio.Portfolio`) plus a PDF ranking report and an
Excel score-breakdown workbook as `ReportArtifact` rows. Pass `--user-id <id>` to attribute the
saved portfolio to a specific `core.User`.

\`\`\`cmd
python manage.py run_screener --universe nifty50 --min-score 3 --top-n 30 --save-csv
\`\`\`
Runs the 6-signal technical screener (trend, 20/60-day momentum, MACD, RSI band, volume ratio,
drawdown-from-52wk-high) against a universe, DB-caching results per `(symbol, as_of_date)` in
`ScreenerResult` and labeling each `PASS` / `PASS (not selected)` / `NEAR MISS` / `FAIL`. `--save-csv`
exports an Excel report via `save_screener_excel`.

\`\`\`cmd
python manage.py predict --model-name test_transformer --universe temp --top-n 10 --plot
\`\`\`
Runs single-model DB-cached inference and prints a ranked BUY/HOLD/AVOID table with portfolio
weights; `--plot` saves a PDF report.

> `run_ensemble` and `rebalance_portfolio` exist as empty command stubs and are **not yet
> implemented**.

### Portfolio

\`\`\`cmd
python manage.py manage_portfolio --list
python manage.py manage_portfolio --view composite_test_trans_20260809
python manage.py manage_portfolio --load-csv legacy_portfolio.csv --name imported_v1
\`\`\`
Lists all DB-stored portfolios, views a specific portfolio's holdings (quantity, purchase price,
allocation %, value), or imports a legacy CSV portfolio into the database.

---

## ⚙️ Core Technical Architecture

1. **DB-First Architecture:** Rather than reading loose CSV files during runtime, historical
   prices (`PriceBar`), computed technical indicators, market context, quality scores, screener
   signals, sentiment, and model outputs are fully cached and queryable via Django ORM models —
   every `get_or_*` / `get_*` service function checks the DB first and only computes/fetches
   what's missing.
2. **Combined Multi-Index Market Context:** `market_data/services/market_context.py` computes
   regime features (1/5/20-day return, above-SMA flag, 20-day volatility, RSI) for **both**
   `^NSEI` and `^BSESN`, storing each index's rows separately in `MarketContext` but exposing them
   through `get_market_context_features()` as one wide, prefixed frame (`NSEI_ret1`,
   `BSESN_above_sma`, etc. — see `MARKET_CONTEXT_COLS`). `training.py::build_dataset_from_db`
   joins this combined frame onto every symbol's feature matrix, so `BSESN_*` columns are no
   longer silently dropped as they were when only a single index was fetched.
3. **Three-Layer Composite Scoring:**
   - **Transformer Layer** (`forecasting/services/ensemble.py`) — predicts expected 30-day
     returns using one or more `TrainedModel`s, weighted-averaged if more than one is supplied.
   - **Quality Factor Layer** (`market_data/services/quality.py`) — Piotroski F-Score and Altman
     Z-Score, blended 60/40 (100% Piotroski where Altman is excluded/unavailable, e.g. for banks).
   - **Technical Indicator Layer** (`forecasting/services/composite.py::get_technical_scores`) —
     True Strength Index (TSI), computed live from `PriceBar` history and mapped to 0–1 via `tanh`.
4. **Technical Screener:** A separate, lighter-weight 6-signal PASS/FAIL screener
   (`forecasting/services/screener.py`) for quick trend/momentum/volume filtering, independent of
   the AI/quality/technical composite pipeline — DB-cached per `(symbol, as_of_date)`.
5. **Role-Based Access:** A custom `core.User` model (`AbstractUser` subclass) carries `role`
   (`analyst` / `admin`) directly on the user object, gating data-pipeline, training, and
   composite-run trigger views/commands without an extra profile-table join.
6. **Background Job Tracking:** `core.models.JobRun` + `core.threading_utils.launch_tracked_command()`
   run every triggerable management command in a daemon thread and expose live status/output to
   the browser via a shared HTMX-polled partial — no external task queue (Celery, etc.) is used;
   this is a deliberate, scope-bounded choice for the current phase. See **Roadmap** for
   production-concurrency considerations this defers.
7. **Portfolio Ownership:** `portfolio.Portfolio.owner` (nullable FK to `core.User`) plus
   visibility filtering in `portfolio/views.py` — see **User Management** above.
8. **Non-Trading-Day Cache Safety:** Only broad, long-lived index tickers
   (`market_data.services.prices.MARKET_CALENDAR_AUTHORITIES = {"^NSEI", "^BSESN"}`) are trusted
   to write to the shared `NonTradingDay` cache. This guards against a confirmed historical bug
   (Aug 2026) where a newly-listed/demerged equity's pre-listing gap got recorded as a
   market-wide holiday, silently truncating other, older symbols' history — several `nifty100`
   symbols (VEDL, TATAPOWER, TORNTPHARM, UNIONBANK, UNITDSPR, ZYDUSLIFE) lost 11+ years of data
   this way before the fix. `repair_price_history` remediates any data still affected.
9. **Automated Artifact Generation:** `run_composite --plot`, `predict --plot`, and
   `run_screener --save-csv` produce publication-ready **ReportLab PDF reports** and multi-sheet
   **openpyxl Excel workbooks**, stored as `ReportArtifact` rows and downloadable from the
   Composite Reports page.

---

## 🚧 Known Limitations & Roadmap

- **Unported universe/config:** the pre-Django `"screened"` universe (25 tickers) and its
  `DATE_RANGES` presets were never migrated into `seed_universes.py`.
- **Unported indicators:** `advance_decline_ratio`, `new_highs_new_lows`, `market_breadth_index`,
  `relative_strength_vs_index`, and `relative_strength_vs_sector` from the pre-Django scripts have
  no Django service equivalent yet.
- **`KBANK.NS`** (in the not-yet-ported `"screened"` universe) is suspected to be a typo for
  `KTKBANK.NS` (Karnataka Bank) — pending verification before that universe is ported. A
  `verify_symbols` management command (auditing all seeded tickers for validity against
  yfinance) has been proposed but not yet built.
- **Empty command/service stubs:** `market_data/management/commands/{fetch_sentiment,backfill_indicators}.py`,
  `forecasting/management/commands/{run_ensemble,rebalance_portfolio}.py`, and
  `forecasting/services/portfolio.py` exist as placeholder files with no implementation.
- **Dead code:** `forecasting/models.py` still defines a legacy `Portfolio`/`PortfolioHolding`
  pair that nothing in the codebase uses — all live portfolio persistence goes through
  `portfolio.models.Portfolio`/`PortfolioItem`. Candidate for removal in a future cleanup.
- **Production concurrency is out of scope for now:** background jobs run as daemon threads
  (`core.threading_utils`) rather than through a task queue (Celery, RQ, etc.), so there's no job
  queueing, retry policy, or CPU/GPU contention handling for simultaneous training runs — fine for
  the current single-process deployment, but a blocker before scaling to concurrent multi-user
  training workloads.

---
