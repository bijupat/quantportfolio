from django.core.management.base import BaseCommand
from core.models import Symbol, Universe, UniverseMember

UNIVERSES = {
    # ── Custom watchlist ──────────────────────────
    "temp": ['NUVOCO.NS', 'SAPPHIRE.NS', 'SAIL.NS', 'SUZLON.NS', 'TMCV.NS', 'USHAMART.NS', 'VMM.NS', 'IDEA.NS'],
    "screened": ['MRPL.NS', 'SAMMAANCAP.NS', 'ANGELONE.NS', 'BSOFT.NS', 'KPITTECH.NS', 'TMPV.NS', 'NSLNISP.NS', 'NATIONALUM.NS', 'HFCL.NS', 'SAIL.NS', 'POONAWALLA.NS', 'NBCC.NS', 'PREMIERENE.NS', 'CENTRALBK.NS', 'BHEL.NS', 'IIFL.NS', 'LATENTVIEW.NS', 'CESC.NS', 'KEC.NS', 'CASTROLIND.NS', 'NUVOCO.NS', 'KBANK.NS', 'TRITURBINE.NS', 'GRAPHITE.NS', 'USHAMART.NS'],
    "bought": ['AARTIIND.NS', 'AMBUJACEM.NS', 'BHARTIARTL.NS', 'BHARTIHEXA.NS', 'GROWW.NS', 'BRITANNIA.NS', 'CONCOR.NS', 'DEVYANI.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'THELEELA.NS', 'LENSKART.NS', 'MRPL.NS', 'MANKIND.NS', 'NTPC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SAPPHIRE.NS', 'SAIL.NS', 'SUZLON.NS', 'TCS.NS', 'TMCV.NS', 'TATASTEEL.NS', 'TARIL.NS', 'VMM.NS', 'IDEA.NS'],
    "bse_bluechip": ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "WIPRO.BO"],
    # ── NSE Index-based ─────────────────────────────
    "nifty50": ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BEL.NS', 'BHARTIARTL.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ETERNAL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'INFY.NS', 'INDIGO.NS', 'JSWSTEEL.NS', 'JIOFIN.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'MAXHEALTH.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHRIRAMFIN.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TMPV.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'],
    "niftynext50": ['ABB.NS', 'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'AMBUJACEM.NS', 'DMART.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BPCL.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CGPOWER.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CUMMINSIND.NS', 'DLF.NS', 'DIVISLAB.NS', 'GAIL.NS', 'GODREJCP.NS', 'HDFCAMC.NS', 'HAL.NS', 'HINDZINC.NS', 'HYUNDAI.NS', 'INDHOTEL.NS', 'IOC.NS', 'IRFC.NS', 'JINDALSTEL.NS', 'LTM.NS', 'LODHA.NS', 'MAZDOCK.NS', 'MUTHOOTFIN.NS', 'PIDILITIND.NS', 'PFC.NS', 'PNB.NS', 'RECLTD.NS', 'MOTHERSON.NS', 'SHREECEM.NS', 'ENRIN.NS', 'SIEMENS.NS', 'SOLARINDS.NS', 'TVSMOTOR.NS', 'TATACAP.NS', 'TMCV.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS', 'UNIONBANK.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'ZYDUSLIFE.NS'],
    "nifty100": ['ABB.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'DMART.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANKBARODA.NS', 'BEL.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BOSCHLTD.NS', 'BRITANNIA.NS', 'CGPOWER.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS', 'CUMMINSIND.NS', 'DLF.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ETERNAL.NS', 'GAIL.NS', 'GODREJCP.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HINDALCO.NS', 'HAL.NS', 'HINDUNILVR.NS', 'HINDZINC.NS', 'HYUNDAI.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDHOTEL.NS', 'IOC.NS', 'IRFC.NS', 'INFY.NS', 'INDIGO.NS', 'JSWSTEEL.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'KOTAKBANK.NS', 'LTM.NS', 'LT.NS', 'LODHA.NS', 'M&M.NS', 'MARUTI.NS', 'MAXHEALTH.NS', 'MAZDOCK.NS', 'MUTHOOTFIN.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'PIDILITIND.NS', 'PFC.NS', 'POWERGRID.NS', 'PNB.NS', 'RECLTD.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'MOTHERSON.NS', 'SHREECEM.NS', 'SHRIRAMFIN.NS', 'ENRIN.NS', 'SIEMENS.NS', 'SOLARINDS.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TVSMOTOR.NS', 'TATACAP.NS', 'TCS.NS', 'TATACONSUM.NS', 'TMCV.NS', 'TMPV.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'TRENT.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'UNITDSPR.NS', 'VBL.NS', 'VEDL.NS', 'WIPRO.NS', 'ZYDUSLIFE.NS'],
    "sensex": ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INFY.BO", "HINDUNILVR.BO", "ITC.BO", "SBIN.BO", "BHARTIARTL.BO", "KOTAKBANK.BO", "LT.BO", "AXISBANK.BO", "BAJFINANCE.BO", "ASIANPAINT.BO", "MARUTI.BO", "SUNPHARMA.BO", "HCLTECH.BO", "ULTRACEMCO.BO", "NTPC.BO", "POWERGRID.BO", "TITAN.BO", "TECHM.BO", "NESTLEIND.BO", "M&M.BO", "BAJAJFINSV.BO", "WIPRO.BO", "JSWSTEEL.BO", "TATASTEEL.BO", "INDUSINDBK.BO", "ONGC.BO"],
    # ── Sector-based ─────────────────────────────
    "banking": ['AUBANK.NS', 'AXISBANK.NS', 'BANKBARODA.NS', 'CANBK.NS', 'FEDERALBNK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'IDFCFIRSTB.NS', 'INDUSINDBK.NS', 'KOTAKBANK.NS', 'PNB.NS', 'SBIN.NS', 'UNIONBANK.NS', 'YESBANK.NS'],
    "it_sector": ['COFORGE.NS', 'HCLTECH.NS', 'INFY.NS', 'LTM.NS', 'MPHASIS.NS', 'OFSS.NS', 'PERSISTENT.NS', 'TCS.NS', 'TECHM.NS', 'WIPRO.NS'],
    "pharma": ['ABBOTINDIA.NS', 'AJANTPHARM.NS', 'ALKEM.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'GLAND.NS', 'GLENMARK.NS', 'IPCALAB.NS', 'JBCHEPHARM.NS', 'LAURUSLABS.NS', 'LUPIN.NS', 'MANKIND.NS', 'PPLPHARMA.NS', 'SUNPHARMA.NS', 'TORNTPHARM.NS', 'WOCKPHARMA.NS', 'ZYDUSLIFE.NS'],
    "auto": ['ASHOKLEY.NS', 'BAJAJ-AUTO.NS', 'BHARATFORG.NS', 'BOSCHLTD.NS', 'EICHERMOT.NS', 'EXIDEIND.NS', 'HEROMOTOCO.NS', 'M&M.NS', 'MARUTI.NS', 'MOTHERSON.NS', 'SONACOMS.NS', 'TVSMOTOR.NS', 'TMPV.NS', 'TIINDIA.NS', 'UNOMINDA.NS'],
    "fmcg": ['BRITANNIA.NS', 'COLPAL.NS', 'DABUR.NS', 'EMAMILTD.NS', 'GODREJCP.NS', 'HINDUNILVR.NS', 'ITC.NS', 'MARICO.NS', 'NESTLEIND.NS', 'PATANJALI.NS', 'RADICO.NS', 'TATACONSUM.NS', 'UBL.NS', 'UNITDSPR.NS', 'VBL.NS'],
}


class Command(BaseCommand):
    """Seeds the database with predefined stock universes and symbol memberships."""

    help = "Seeds database with predefined stock universes and symbol memberships."

    def handle(self, *args, **kwargs) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("Starting Universe and Symbol Seeding..."))

        total_universes = 0
        total_symbols_created = 0
        total_memberships = 0

        banking_tickers = set(UNIVERSES.get("banking", []))

        for universe_name, tickers in UNIVERSES.items():
            universe, _ = Universe.objects.get_or_create(
                name=universe_name,
                defaults={"description": f"Predefined universe: {universe_name}"}
            )
            total_universes += 1

            for ticker in tickers:
                is_financial = ticker in banking_tickers

                symbol, sym_created = Symbol.objects.get_or_create(
                    ticker=ticker,
                    defaults={
                                "is_financial": is_financial,
                                "is_active": True,
                                }
    )

                if sym_created:
                    total_symbols_created += 1
                else:
                    # Keep existing symbols in sync if re-run with updated data
                    updated_fields = []
                    if is_financial and not symbol.is_financial:
                        symbol.is_financial = True
                        updated_fields.append("is_financial")
                    if not symbol.is_active:
                        symbol.is_active = True
                        updated_fields.append("is_active")
                    if updated_fields:
                        symbol.save(update_fields=updated_fields)

                _, mem_created = UniverseMember.objects.get_or_create(
                    universe=universe,
                    symbol=symbol,
                )
                if mem_created:
                    total_memberships += 1

        self.stdout.write(self.style.SUCCESS(
            f"Successfully seeded database:\n"
            f" - Universes processed: {total_universes}\n"
            f" - Unique Symbols created: {total_symbols_created}\n"
            f" - Total Memberships linked: {total_memberships}"
        ))