"""
data_loader.py - Fetch stock, index and market data from Yahoo Finance
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Optional, Dict

from util import log, cache_path, DATA_CACHE

warnings.filterwarnings("ignore")
from universes import UNIVERSES

# ─────────────────────────────────────────────
# Default universe: NIFTY 50 stocks
# ─────────────────────────────────────────────



nifty_smallcap_100= [
    "AADHAR.NS", "AARTIIND.NS", "ABREL.NS", "ACE.NS", "AEGISLOG.NS", "AFCONS.NS", "AFFLE.NS", 
    "AMARARAJA.NS", "AMBER.NS", "ANANDRATHI.NS", "ANANTRAJ.NS", "ANGELONE.NS", "APTUS.NS", 
    "ASTERDM.NS", "ATUL.NS", "BANDHANBNK.NS", "BEML.NS", "BLS.NS", "BRIGADE.NS", "BSOFT.NS", 
    "CAMS.NS", "CAPACITE.NS", "CASTROLIND.NS", "CDSL.NS", "CESC.NS", "CHAMBLFERT.NS", 
    "CHOLAHLDNG.NS", "CLEAN.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CROMPTON.NS", "CYIENT.NS", 
    "DATAPATTNS.NS", "DEEPAKFERT.NS", "DELHIVERY.NS", "DEVYANI.NS", "DLINK.NS", "DODLA.NS", 
    "EASEMYTRIP.NS", "ELGIEQUIP.NS", "ERIS.NS", "FSL.NS", "GLS.NS", "GODIGIT.NS", "GPIL.NS", 
    "GRSE.NS", "HAPPYFORGE.NS", "HBLENGINE.NS", "HINDCOPPER.NS", "HSCL.NS", "HUDCO.NS", 
    "IEX.NS", "IFCI.NS", "IIFL.NS", "IRB.NS", "IRCON.NS", "ITI.NS", "J&KBANK.NS", "JBCHEPHARM.NS", 
    "JBMA.NS", "JINDALSAW.NS", "JWL.NS", "KAJARIACER.NS", "KAYNES.NS", "KEC.NS", "KFINTECH.NS", 
    "KPITTECH.NS", "LATENTVIEW.NS", "LAURUSLABS.NS", "LTF.NS", "MANAPPURAM.NS", "MCX.NS", 
    "METROPOLIS.NS", "MOTILALOFS.NS", "NATCOPHARM.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS", 
    "NEULANDLAB.NS", "NUVAMA.NS", "OLAELEC.NS", "ORCHPHARMA.NS", "PCBL.NS", "PFIZER.NS", 
    "PGEL.NS", "PIRPHARMA.NS", "PNBHOUSING.NS", "POONAWALLA.NS", "QUESS.NS", "RADICO.NS", 
    "RAILTEL.NS", "RAMCOCEM.NS", "RBLBANK.NS", "REDINGTON.NS", "RELIANCEPWR.NS", "RITES.NS", 
    "RVNL.NS", "SAPPHIRE.NS", "SHYAMMETL.NS", "SIGNATURE.NS", "SONATSOFTW.NS", "STARHEALTH.NS", 
    "SWANENERGY.NS", "TANLA.NS", "TATACHEM.NS", "TEJASNET.NS", "TRIDENT.NS", "TRITURBINE.NS", 
    "VIJAYA.NS", "WELCORP.NS", "WHIRLPOOL.NS", "WOCKHARDT.NS", "ZENSARTECH.NS", "ZENTEC.NS"
]



nifty_500 = [
    "360ONE.NS", "3MINDIA.NS", "ABB.NS", "ACC.NS", "AIAENG.NS", "APLAPOLLO.NS", "AUBANK.NS", "AARTIDRUGS.NS", "AARTIIND.NS", 
    "AAVAS.NS", "ABBOTINDIA.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", 
    "ADANITOTAL.NS", "ADVENZYMES.NS", "AEGISLOG.NS", "AETHER.NS", "AFFLE.NS", "AJANTPHARM.NS", "AKZOINDIA.NS", 
    "ALKYLAMINE.NS", "ALLCARGO.NS", "ALOKINDS.NS", "AMBER.NS", "AMBUJACEM.NS", "ANANDRATHI.NS", "ANGELONE.NS", 
    "APARINDS.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "APTUS.NS", "ARCELORMTN.NS", "ARCHIDPLY.NS", "ARVIND.NS", 
    "ARVINDFASN.NS", "ASAHIINDIA.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTERDM.NS", "ASTRAZEN.NS", "ASTRAL.NS", 
    "ATUL.NS", "AUROPHARM.NS", "AVANTIFEED.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJCON.NS", "BAJAJELEC.NS", 
    "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "BAJFINANCE.NS", "BALAMINES.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", 
    "BANKBARODA.NS", "BANKINDIA.NS", "BASF.NS", "BATAINDIA.NS", "BAYERCROP.NS", "BEEER.NS", "BEL.NS", "BEML.NS", 
    "BERGEPAINT.NS", "BHANSALIO.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BIRLACORPN.NS", 
    "BSOFT.NS", "BLS.NS", "BLUESTARCO.NS", "BOMDYEING.NS", "BOSCHLTD.NS", "BPCL.NS", "BRIGADE.NS", "BRITANNIA.NS", 
    "BSE.NS", "CAMPUS.NS", "CANBK.NS", "CANFINHOME.NS", "CAPLIPOINT.NS", "CGPOWER.NS", "CHALET.NS", "CHAMBLFERT.NS", 
    "CHOLAHLDNG.NS", "CHOLAFIN.NS", "CIPLA.NS", "CLEAN.NS", "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS", 
    "CONCOR.NS", "COROMANDEL.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "CYIENT.NS", 
    "DCMSHRIRAM.NS", "DLF.NS", "DABUR.NS", "DALBHARAT.NS", "DATAPATTNS.NS", "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", 
    "DELTACORP.NS", "DEVYANI.NS", "DIVISLAB.NS", "DIXON.NS", "DRREDDY.NS", "EASEMYTRIP.NS", "EICHERMOT.NS", "EIDPARRY.NS", 
    "EIHOTEL.NS", "ELGIEQUIP.NS", "EMAMILTD.NS", "ENDURANCE.NS", "ENGINERSIN.NS", "EQUITASBNK.NS", "ERIS.NS", "ESCORTS.NS", 
    "EXIDEIND.NS", "FDC.NS", "FEDERALBNK.NS", "FACT.NS", "FINEORG.NS", "FINPIPE.NS", "FINCABLES.NS", "FORTIS.NS", 
    "FSNKYN.NS", "GAIL.NS", "GMMPFAUDLR.NS", "GMRINFRA.NS", "GRSE.NS", "GICRE.NS", "GILLETTE.NS", "GLAND.NS", 
    "GLAXO.NS", "GLENMARK.NS", "GOCOLORS.NS", "GODREJAGRO.NS", "GODREJCP.NS", "GODREJIND.NS", "GODREJPROP.NS", 
    "GRANULES.NS", "GRAPHITE.NS", "GRASIM.NS", "GRINDWELL.NS", "GUJGASLTD.NS", "GNFC.NS", "GPPL.NS", "GSFC.NS", 
    "GSPL.NS", "GULFOILLUB.NS", "HAL.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HDFCAMC.NS", "HFCL.NS", 
    "HAPPSTMNDS.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HIKAL.NS", "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", 
    "HINDUNILVR.NS", "HINDZINC.NS", "HITACHI.NS", "HOMEFIRST.NS", "HONAUT.NS", "HUDCO.NS", "ICICIBANK.NS", "ICICIGI.NS", 
    "ICICIPRULI.NS", "ISEC.NS", "IDBI.NS", "IDFCFIRSTB.NS", "IDFC.NS", "IEX.NS", "IIFL.NS", "IRB.NS", "IRCON.NS", 
    "ITC.NS", "ITI.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIANB.NS", "IOLCP.NS", "INDIGO.NS", "INDIGOPNTS.NS", 
    "INDUSINDBK.NS", "INDUSTOWER.NS", "INFIBEAM.NS", "INFY.NS", "INOXWIND.NS", "INTELLECT.NS", "INDHOTEL.NS", "IOC.NS", 
    "IRCTC.NS", "IRFC.NS", "ISGEC.NS", "JBCHEPHARM.NS", "JKCEMENT.NS", "JKPAPER.NS", "JKTYRE.NS", "JMFINANCIL.NS", 
    "JSWENERGY.NS", "JSWSTEEL.NS", "JAMNAAUTO.NS", "JBMA.NS", "JSL.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "JUBLINGREA.NS", 
    "JUBLPHARMA.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", "KPRMILL.NS", "KEI.NS", "KNRCON.NS", "KPITTECH.NS", "KSAILS.NS", 
    "KOTAKBANK.NS", "KRSNAA.NS", "KALYANKJIL.NS", "KANSAINER.NS", "KARURVYSYA.NS", "KEC.NS", "KRBL.NS", "L&TFH.NS", 
    "LTTS.NS", "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LAURUSLABS.NS", "LAXMIMACH.NS", "LICI.NS", "LINDEINDIA.NS", 
    "LUPIN.NS", "LUXIND.NS", "MMTC.NS", "MOIL.NS", "MRF.NS", "MTARTECH.NS", "MGL.NS", "M&MFIN.NS", "M&M.NS", 
    "MAHINDCIE.NS", "MAHLOG.NS", "MAHSEAMLES.NS", "MANAPPURAM.NS", "MAPMYINDIA.NS", "MARICO.NS", "MARUTI.NS", 
    "MASTEK.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MEDPLUS.NS", "METROPOLIS.NS", "MFSL.NS", "MINDACORP.NS", "MSUMI.NS", 
    "MOTILALOFS.NS", "MPHASIS.NS", "MCX.NS", "MUTHOOTFIN.NS", "NATCOPHARM.NS", "NBCC.NS", "NCC.NS", "NESCO.NS", 
    "NHPC.NS", "NLCINDIA.NS", "NMDC.NS", "NOCIL.NS", "NTPC.NS", "NH.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", 
    "NAZARA.NS", "NESTLEIND.NS", "NETWORK18.NS", "NIFTYBEES.NS", "NILKAMAL.NS", "NIACL.NS", "OBEROIRLTY.NS", 
    "ONGC.NS", "OIL.NS", "OLECTRA.NS", "PAYTM.NS", "PCBL.NS", "PIIND.NS", "PNB.NS", "PVRINOX.NS", "PAGEIND.NS", 
    "PATANJALI.NS", "PEL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PHOENIXLTD.NS", "PIDILITIND.NS", "POLYMED.NS", 
    "POLYCAB.NS", "POONAWALLA.NS", "PPLPHARMA.NS", "PRAJIND.NS", "PRESTIGE.NS", "PRINCEPIPE.NS", "PRSMJOHNSN.NS", 
    "PGHH.NS", "PNBHOUSING.NS", "QUESS.NS", "RVNL.NS", "RECLTD.NS", "RELIANCE.NS", "RBLBANK.NS", "RITES.NS", 
    "RADICO.NS", "RAIN.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RATNAMANI.NS", "RAYMOND.NS", "REDINGTON.NS", 
    "RELAXO.NS", "RHS.NS", "RHIM.NS", "RUSTOMJEE.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SJVN.NS", 
    "SKFINDIA.NS", "SRF.NS", "SAFARI.NS", "SANOFI.NS", "SAPPHIRE.NS", "SAREGAMA.NS", "SCHAEFFLER.NS", "SHREECEM.NS", 
    "SHRIRAMFIN.NS", "SIEMENS.NS", "SOBHA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SONATSOFTW.NS", "STARHEALTH.NS", 
    "STERTOOLS.NS", "SUMICHEM.NS", "SUNPHARMA.NS", "SUNTV.NS", "SUPRAJIT.NS", "SUPREMEIND.NS", "SUZLON.NS", 
    "SYNGENE.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", 
    "TATATECH.NS", "TCS.NS", "TEAMLEASE.NS", "TECHM.NS", "TEJASNET.NS", "THERMAX.NS", "THYROCARE.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TRIDENT.NS", "TRIVENI.NS", "TRITURBINE.NS", "TIINDIA.NS", 
    "UCOBANK.NS", "UNOMINDA.NS", "UPL.NS", "UTIAMC.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UJJIVANSFB.NS", 
    "USHAMART.NS", "VGUARD.NS", "V-MART.NS", "VIPIND.NS", "VAIBHAVGBL.NS", "VAKRANGEE.NS", "VALIANTORG.NS", 
    "VARROC.NS", "VBL.NS", "VEDL.NS", "VENKEYS.NS", "VIJAYA.NS", "VINATIORGA.NS", "VOLTAS.NS", "WELCORP.NS", 
    "WELSPUNLIV.NS", "WHIRLPOOL.NS", "WIPRO.NS", "YESBANK.NS", "ZEEL.NS", "ZENSARTECH.NS", "ZOMATO.NS", 
    "ZYDUSLIFE.NS", "ZYDUSWELL.NS"
]


MARKET_INDICES = ["^NSEI", "^BSESN"]

# ─────────────────────────────────────────────
# Core fetch function
# ─────────────────────────────────────────────
def fetch_stock_data(
    symbol: str,
    start: str = "2013-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    cache_ttl_days: int = 1,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol.  Returns a clean DataFrame with
    columns [Open, High, Low, Close, Volume].
    """
    path = cache_path(symbol)

    # ── Cache check ──────────────────────────
    if use_cache and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days < cache_ttl_days:
            log.debug(f"Cache hit  ->{symbol}")
            df = pd.read_parquet(path)
            return df

    # ── Download ─────────────────────────────
    log.info(f"Downloading {symbol} [{start} ->{end or 'today'}]")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            log.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df = df.sort_index()

        # Basic cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(how="all", inplace=True)
        df["Close"] = df["Close"].ffill()
        for col in ["Open", "High", "Low"]:
            df[col] = df[col].fillna(df["Close"])
        df["Volume"] = df["Volume"].fillna(0)

        # Persist cache
        df.to_parquet(path)
        log.debug(f"Cached {symbol} ->{path.name}")
        return df

    except Exception as e:
        log.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_index_data(
    indices: List[str] = MARKET_INDICES,
    start: str = "2013-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch index OHLCV data for each index symbol."""
    result = {}
    for idx in indices:
        df = fetch_stock_data(idx, start=start, end=end)
        if not df.empty:
            result[idx] = df
    return result


def fetch_multi_stock(
    symbols: List[str] = UNIVERSES["nifty50"],
    start: str = "2013-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Batch fetch for all stocks with retry and throttle."""
    results = {}
    for i, sym in enumerate(symbols):
        df = fetch_stock_data(sym, start=start, end=end)
        if not df.empty:
            results[sym] = df
        if i > 0 and i % 10 == 0:
            time.sleep(1)  # polite throttle
    log.info(f"Loaded {len(results)}/{len(symbols)} symbols successfully.")
    return results


# ─────────────────────────────────────────────
# Market context builder
# ─────────────────────────────────────────────
def build_market_context(
    index_data: Dict[str, pd.DataFrame],
    window: int = 20,
) -> pd.DataFrame:
    """
    Build daily market-regime features from index data.
    Returns a DataFrame indexed by Date.
    """
    dfs = []
    for sym, df in index_data.items():
        safe = sym.replace("^", "").replace(" ", "_")
        sub = df[["Close"]].copy()

        # Returns
        sub[f"{safe}_ret1"]  = sub["Close"].pct_change(1)
        sub[f"{safe}_ret5"]  = sub["Close"].pct_change(5)
        sub[f"{safe}_ret20"] = sub["Close"].pct_change(20)

        # Trend
        sub[f"{safe}_sma20"] = sub["Close"].rolling(20).mean()
        sub[f"{safe}_above_sma"] = (sub["Close"] > sub[f"{safe}_sma20"]).astype(float)

        # Volatility
        sub[f"{safe}_vol20"] = sub[f"{safe}_ret1"].rolling(20).std()

        # RSI
        delta = sub["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        sub[f"{safe}_rsi"] = 100 - 100 / (1 + rs)

        sub.drop(columns=["Close", f"{safe}_sma20"], inplace=True)
        dfs.append(sub)

    if not dfs:
        return pd.DataFrame()

    ctx = pd.concat(dfs, axis=1)
    ctx.fillna(method="ffill", inplace=True)
    ctx.dropna(inplace=True)
    return ctx


# ─────────────────────────────────────────────
# Label creation  (future 30-day return)
# ─────────────────────────────────────────────
def create_labels(close: pd.Series, horizon: int = 30) -> pd.Series:
    """
    future_return_t = (close_{t+horizon} - close_t) / close_t
    Last `horizon` rows become NaN (no future yet).
    """
    future_close = close.shift(-horizon)
    labels = (future_close - close) / (close + 1e-9)
    labels.name = "future_return"
    return labels
