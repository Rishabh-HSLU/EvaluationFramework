"""
Download the canonical 948-ticker 1-minute bar dataset from Alpaca Markets.

This reproduces the raw CSV corpus used to build the benchmark eval set.
Each ticker is saved as <output_dir>/<TICKER>.csv with columns:
    timestamp (UTC, e.g. 2019-09-03 13:31:00+00:00)
    open, high, low, close, volume, trade_count, vwap

A checkpoint file (download_log.csv) lets interrupted runs resume safely.

Usage
-----
    python -m data_pipeline.fetch_alpaca \\
        --api-key  PKxxx \\
        --secret   yyy  \\
        --out-dir  data/raw_intraday

Or from Python:

    from data_pipeline.fetch_alpaca import download_corpus
    download_corpus(api_key="PKxxx", secret_key="yyy", out_dir="data/raw_intraday")

Requirements
------------
    pip install "alpaca-py>=0.13"   (alpaca-trade-api-python v3+)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical parameters — must not change for reproducibility
# ---------------------------------------------------------------------------

START = datetime(2019, 9, 3, 13, 31, 0)
END = datetime(2020, 3, 20, 20, 3, 0)

CHECKPOINT_NAME = "download_log.csv"

# 948 tickers — the exact universe from the original benchmark dataset
CANONICAL_TICKERS: list[str] = [
    "AA", "AAL", "AAN", "AAP", "AAT", "AAXN", "ABCB", "ABM", "ABMD", "ACA",
    "ACAD", "ACC", "ACHC", "ACIW", "ACM", "ADC", "ADPT", "ADS", "ADT", "ADUS",
    "AEIS", "AEL", "AEO", "AER", "AES", "AFG", "AGCO", "AGIO", "AGNC", "AGO",
    "AIT", "AIV", "AIZ", "AJRD", "AKR", "AL", "ALB", "ALE", "ALEC", "ALGT",
    "ALK", "ALKS", "ALLE", "ALLK", "ALLO", "ALLY", "ALRM", "ALSN", "ALV",
    "AMBA", "AMCX", "AMED", "AMG", "AMH", "AMKR", "AMN", "AN", "ANGI", "AOS",
    "APA", "APAM", "APLE", "APLS", "APTV", "AQUA", "ARCC", "ARI", "ARMK",
    "ARNA", "ARNC", "ARVN", "ARW", "ARWR", "ASB", "ASGN", "ASH", "ATCO",
    "ATGE", "ATH", "ATHM", "ATR", "ATSG", "AUB", "AVA", "AVAV", "AVLR",
    "AVNS", "AVT", "AVTR", "AVY", "AWI", "AWR", "AXS", "AXTA", "AY", "AYI",
    "AYX", "AZPN", "B", "BAH", "BAND", "BANR", "BBIO", "BC", "BCO", "BDC",
    "BDN", "BECN", "BEN", "BERY", "BEST", "BFAM", "BG", "BGCP", "BGS", "BHF",
    "BHVN", "BILI", "BILL", "BJ", "BKH", "BKI", "BKU", "BL", "BLD", "BLDR",
    "BLKB", "BLUE", "BMCH", "BOH", "BOKF", "BOX", "BPMC", "BPOP", "BPY",
    "BRC", "BRKR", "BRKS", "BRX", "BURL", "BWA", "BWXT", "BXMT", "BXS",
    "BYND", "BZUN", "CACC", "CALM", "CARG", "CASY", "CATM", "CATY", "CBRL",
    "CBSH", "CBT", "CBU", "CC", "CCK", "CCL", "CCMP", "CCOI", "CCXI", "CDAY",
    "CDK", "CE", "CF", "CFG", "CFR", "CFX", "CG", "CGNX", "CHDN", "CHGG",
    "CHH", "CHNG", "CHRW", "CIEN", "CIM", "CIT", "CLDR", "CLF", "CLGX",
    "CLH", "CLI", "CLR", "CMA", "CMC", "CMD", "CMP", "CMPR", "CNA", "CNMD",
    "CNO", "CNP", "CNX", "COG", "COHR", "COLB", "COLD", "COLM", "COMM",
    "CONE", "COR", "CORE", "COTY", "COUP", "CPRI", "CPT", "CR", "CREE",
    "CRI", "CRL", "CRUS", "CRWD", "CSFL", "CSGS", "CSII", "CSL", "CSOD",
    "CTLT", "CUB", "CUBE", "CUZ", "CVA", "CVBF", "CVI", "CVLT", "CVNA",
    "CW", "CWK", "CWST", "CWT", "CXO", "CXP", "CXW", "CY", "CYBR", "CZR",
    "CZZ", "DAR", "DBX", "DCI", "DCPH", "DDOG", "DEA", "DECK", "DEI", "DIOD",
    "DISCA", "DISH", "DKS", "DLB", "DNKN", "DNLI", "DOC", "DOX", "DOYU",
    "DRI", "DRQ", "DT", "DVA", "DVN", "DXC", "EAF", "EBS", "EDIT", "EEFT",
    "EGHT", "EGOV", "EGP", "EHC", "EHTH", "ELAN", "ELS", "EME", "EMN", "ENR",
    "ENS", "ENSG", "ENTG", "ENV", "EPAC", "EPAM", "EPAY", "EPC", "EPZM",
    "EQC", "EQH", "EQM", "EQT", "ESI", "ESNT", "ESRT", "ESTC", "ETFC",
    "ETRN", "ETSY", "EV", "EVR", "EVTC", "EWBC", "EXEL", "EXP", "EXPE",
    "EXPO", "EYE", "FAF", "FANG", "FBC", "FBHS", "FCFS", "FCN", "FCPT",
    "FCX", "FEYE", "FFBC", "FFIN", "FFIV", "FG", "FGEN", "FHB", "FHI",
    "FHN", "FIBK", "FICO", "FIT", "FIVE", "FIX", "FIZZ", "FL", "FLEX",
    "FLIR", "FLO", "FLS", "FMBI", "FMC", "FN", "FNB", "FND", "FNF", "FORM",
    "FOXF", "FR", "FRT", "FSLR", "FSLY", "FSS", "FTCH", "FTDR", "FTI",
    "FTSV", "FUL", "FULT", "FWRD", "G", "GATX", "GBCI", "GBT", "GCP",
    "GDDY", "GEO", "GGG", "GH", "GKOS", "GL", "GLPI", "GMED", "GNRC",
    "GNTX", "GO", "GPC", "GPK", "GPS", "GRA", "GSX", "GT", "GTN", "GWB",
    "GWRE", "H", "HAE", "HAIN", "HAL", "HALO", "HAS", "HASI", "HBI", "HCSG",
    "HDS", "HE", "HEI", "HELE", "HES", "HFC", "HHC", "HI", "HIG", "HII",
    "HIW", "HLF", "HLI", "HMSY", "HOG", "HOLX", "HOMB", "HOPE", "HP", "HPP",
    "HQY", "HR", "HRB", "HRC", "HSIC", "HST", "HTA", "HTH", "HTHT", "HTLD",
    "HUBB", "HUBG", "HUBS", "HUN", "HWC", "HXL", "HZNP", "IAA", "IART",
    "IBKC", "IBOC", "IBTX", "ICLR", "ICPT", "ICUI", "IDA", "IDCC", "IEX",
    "IIVI", "INDB", "INGR", "INO", "INOV", "INSM", "INST", "INT", "IONS",
    "IPG", "IPGP", "IPHI", "IR", "IRDM", "IRM", "IRTC", "IRWD", "ISBC",
    "IT", "ITGR", "ITRI", "ITT", "IVZ", "JAZZ", "JBGS", "JBHT", "JBL",
    "JBLU", "JCOM", "JEF", "JHG", "JJSF", "JLL", "JNPR", "JWN", "KALU",
    "KAR", "KBR", "KEM", "KEX", "KFY", "KIM", "KLIC", "KMPR", "KMT", "KMX",
    "KN", "KNSL", "KNX", "KPTI", "KRC", "KSS", "KTB", "KTOS", "L", "LAD",
    "LAMR", "LANC", "LAUR", "LAZ", "LBTYA", "LEA", "LECO", "LEG", "LEN",
    "LEVI", "LFUS", "LHCG", "LII", "LITE", "LIVN", "LK", "LKQ", "LM",
    "LMNX", "LNC", "LOGM", "LOPE", "LPLA", "LPX", "LSCC", "LSI", "LSTR",
    "LTC", "LVGO", "LW", "LXP", "LYFT", "LYV", "M", "MAN", "MANH", "MANT",
    "MAS", "MASI", "MAT", "MC", "MCY", "MDC", "MDLA", "MDU", "MEDP", "MGM",
    "MGP", "MHK", "MIC", "MIDD", "MIME", "MKSI", "MLCO", "MLM", "MLNX",
    "MMP", "MMS", "MMSI", "MNRO", "MNTA", "MOH", "MOMO", "MOS", "MPLX",
    "MPW", "MPWR", "MRCY", "MRO", "MRTX", "MSM", "MTG", "MTH", "MTN",
    "MTSI", "MTZ", "MUSA", "MWA", "MYL", "MYOK", "NATI", "NAV", "NBIX",
    "NBL", "NCLH", "NCR", "NDSN", "NEOG", "NET", "NEWR", "NFG", "NGVT",
    "NHI", "NI", "NIO", "NJR", "NKTR", "NLOK", "NLSN", "NNN", "NOMD",
    "NOV", "NRG", "NRZ", "NSA", "NSIT", "NSP", "NTAP", "NTCT", "NTNX",
    "NTRA", "NUAN", "NUE", "NUVA", "NVCR", "NVRO", "NVST", "NVT", "NWBI",
    "NWE", "NWL", "NWN", "NWSA", "NXST", "NYCB", "NYT", "OC", "OFC", "OGE",
    "OGS", "OHI", "OKE", "OLED", "OLN", "OMCL", "OMF", "ON", "ONB", "ONEM",
    "ONTO", "OPK", "OR", "ORA", "ORI", "OSIS", "OSK", "OUT", "OXY", "OZK",
    "PAA", "PACW", "PAG", "PAGS", "PB", "PBCT", "PBH", "PCG", "PCH", "PCRX",
    "PCTY", "PD", "PDCO", "PDM", "PE", "PEGA", "PEN", "PFGC", "PFSI", "PGNY",
    "PGRE", "PHM", "PII", "PINC", "PING", "PINS", "PK", "PKG", "PKI", "PLAN",
    "PLNT", "PLXS", "PNFP", "PNM", "PNR", "PNW", "PODD", "POOL", "POR",
    "POST", "POWI", "PPC", "PPD", "PRAA", "PRAH", "PRGO", "PRGS", "PRI",
    "PRNB", "PRSP", "PS", "PSMT", "PSTG", "PSXP", "PTC", "PTCT", "PTON",
    "PVH", "PWR", "PXD", "PZZA", "QDEL", "QGEN", "QLYS", "QRTEA", "QRVO",
    "QTS", "QTWO", "QURE", "R", "RAD", "RAMP", "RARE", "RBC", "RCL", "RDN",
    "REG", "REXR", "REYN", "RF", "RGA", "RGEN", "RGLD", "RH", "RHI", "RL",
    "RMBS", "RNR", "RNST", "ROCK", "ROKU", "RP", "RPM", "RS", "RUN", "RXN",
    "RYN", "SABR", "SAFM", "SAIA", "SAIC", "SAIL", "SANM", "SATS", "SBGI",
    "SBH", "SBNY", "SBRA", "SC", "SCI", "SDC", "SDGR", "SEDG", "SEE", "SEM",
    "SERV", "SF", "SFIX", "SFM", "SFNC", "SHAK", "SHEN", "SHLX", "SHO",
    "SHOO", "SIGI", "SIMO", "SINA", "SITE", "SIVB", "SJI", "SKX", "SLAB",
    "SLG", "SLM", "SMAR", "SMG", "SMTC", "SNA", "SNDR", "SNV", "SNX", "SON",
    "SPB", "SPCE", "SPR", "SPSC", "SPXC", "SR", "SRC", "SRCL", "SRPT", "SSB",
    "SSD", "SSNC", "SSRM", "STAG", "STAY", "STL", "STLD", "STMP", "STNE",
    "STOR", "STRA", "STWD", "SVMK", "SWCH", "SWI", "SWN", "SWX", "SXT",
    "SYF", "SYNA", "SYNH", "TAP", "TCBI", "TCF", "TCO", "TDC", "TDOC",
    "TDS", "TDY", "TECD", "TECH", "TECK", "TENB", "TER", "TERP", "TGNA",
    "THC", "THG", "THO", "THS", "TKR", "TNET", "TOL", "TPR", "TPTX", "TPX",
    "TREE", "TREX", "TRGP", "TRIP", "TRMB", "TRMK", "TRN", "TRNO", "TRTN",
    "TSCO", "TTC", "TTEC", "TTEK", "TW", "TWO", "TXRH", "TXT", "UAA", "UAL",
    "UBSI", "UCBI", "UE", "UFPI", "UFS", "UGI", "UHS", "UI", "ULTA", "UMBF",
    "UMPQ", "UNIT", "UNM", "UNVR", "URBN", "URI", "USFD", "UTHR", "VAC",
    "VC", "VER", "VG", "VGR", "VIAC", "VIAV", "VICI", "VIPS", "VIRT", "VLY",
    "VNO", "VOYA", "VRNS", "VRNT", "VSAT", "VSH", "VST", "VTR", "VVV", "W",
    "WAB", "WAFD", "WAL", "WB", "WBS", "WDC", "WEN", "WERN", "WES", "WEX",
    "WH", "WHR", "WING", "WIX", "WK", "WLK", "WMGI", "WMS", "WOR", "WPC",
    "WRE", "WRI", "WRK", "WSBC", "WSFS", "WSM", "WSO", "WTFC", "WTRG", "WU",
    "WUBA", "WWD", "WWE", "WWW", "WYND", "WYNN", "XEC", "XP", "XPO", "XRAY",
    "XRX", "YELP", "YETI", "YEXT", "YY", "ZEN", "ZG", "ZION", "ZNGA", "ZS",
]


def download_corpus(
    api_key: str,
    secret_key: str,
    out_dir: str | Path = "data/raw_intraday",
    tickers: list[str] | None = None,
    rate_limit_secs: float = 0.4,
    verbose: bool = True,
) -> None:
    """
    Download 1-minute bars for every ticker in *tickers* (default:
    CANONICAL_TICKERS) and write one CSV per ticker to *out_dir*.

    Already-downloaded tickers are skipped via the checkpoint log.

    Parameters
    ----------
    api_key, secret_key
        Alpaca Markets credentials (paper or live both work; data access
        requires a funded account or a data subscription).
    out_dir
        Directory for output CSVs and the checkpoint log.
    tickers
        Override the canonical list; defaults to all 948 CANONICAL_TICKERS.
    rate_limit_secs
        Pause between API calls; Alpaca free tier allows ~200 req/min.
    verbose
        Print progress to stdout.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is required: pip install 'alpaca-py>=0.13'"
        ) from exc

    if tickers is None:
        tickers = CANONICAL_TICKERS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / CHECKPOINT_NAME

    if checkpoint.exists():
        log_df = pd.read_csv(checkpoint)
        done = set(log_df.loc[log_df["status"] == "ok", "ticker"].tolist())
        if verbose:
            print(f"Resuming — {len(done)} already downloaded")
    else:
        log_df = pd.DataFrame(columns=["ticker", "bars", "status", "note"])
        done = set()

    client = StockHistoricalDataClient(api_key, secret_key)
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        if ticker in done:
            continue

        if verbose:
            print(f"[{i + 1}/{n}] {ticker} ...", end=" ", flush=True)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=START,
            end=END,
            adjustment="split",
            feed="sip",
        )

        try:
            bars = client.get_stock_bars(request)
            df = bars.df

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")

            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "timestamp"

            df.to_csv(out_dir / f"{ticker}.csv")
            n_bars = len(df)

            if verbose:
                print(f"{n_bars} bars")

            log_df = pd.concat(
                [log_df, pd.DataFrame([{"ticker": ticker, "bars": n_bars, "status": "ok", "note": ""}])],
                ignore_index=True,
            )

        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"ERROR — {exc}")
            log_df = pd.concat(
                [log_df, pd.DataFrame([{"ticker": ticker, "bars": 0, "status": "error", "note": str(exc)}])],
                ignore_index=True,
            )

        log_df.to_csv(checkpoint, index=False)
        time.sleep(rate_limit_secs)

    if verbose:
        counts = log_df["status"].value_counts().to_dict()
        print(f"\nDone. {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download canonical Alpaca 1-minute bars for the benchmark corpus."
    )
    parser.add_argument("--api-key", required=True, help="Alpaca API key")
    parser.add_argument("--secret", required=True, help="Alpaca secret key")
    parser.add_argument(
        "--out-dir",
        default="data/raw_intraday",
        help="Output directory (default: data/raw_intraday)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.4,
        help="Seconds between API calls (default: 0.4)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Override: download only these tickers instead of the full 948",
    )
    args = parser.parse_args()

    download_corpus(
        api_key=args.api_key,
        secret_key=args.secret,
        out_dir=args.out_dir,
        tickers=args.tickers,
        rate_limit_secs=args.rate_limit,
    )
