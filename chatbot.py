# chatbot_optimized.py
# Financial Chatbot - Optimized with Riskfolio-Lib, Risk Score, Multi-Source News Sentiment + ETS Forecasting
# (Removed pandas-ta dependency and technical strategy scanning)

# WARNING: Hardcoding API keys directly in the script is a SIGNIFICANT SECURITY RISK.
#          It is STRONGLY RECOMMENDED to use Environment Variables or Streamlit Secrets (`secrets.toml`)
#          to manage your API keys securely.
#          This code includes the keys as requested, but this is NOT best practice.

import os
import re
import time
import io
import traceback
import logging
from datetime import datetime, timedelta, date, timezone
from itertools import product
import requests
import json
from collections import Counter
import sys
import warnings


# --- Data Science & Math ---
import pandas as pd
import numpy as np
from scipy.special import binom
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing # For ETS
import scipy.stats as stats
import sklearn.covariance as skcov
from sklearn.metrics import mean_squared_error # For ETS evaluation
from numpy.linalg import inv

# --- Financial Data APIs ---
import yfinance as yf

# --- Technical Analysis (Basic SMA only) ---
# Removed: import pandas_ta as ta

# --- AI & Streamlit ---
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError # Explicitly import AuthenticationError

# --- NEWS SENTIMENT LIBS ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser # For RSS feeds
from dateutil import parser as date_parser # For flexible date parsing

# --- Configure Logging and Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # For ETS function logging

warnings.filterwarnings("ignore", module="statsmodels") # General statsmodels warnings
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="Non-stationary starting parameters found")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Could not find frequency for") # yfinance history index sometimes lacks freq

# --- Constants, Weights, Ranges ---
DEFAULT_WEIGHTS = {
    'volatility': 0.10, 'semi_deviation': 0.10, 'market_cap': 0.08, 'liquidity': 0.05,
    'pe_or_ps': 0.08, 'price_vs_sma': 0.05, 'beta': 0.08, 'piotroski': 0.05,
    'vix': 0.12, 'cvar': 0.09, 'cdar': 0.10, 'mdd': 0.05, 'gmd': 0.05,
}
_weight_sum = sum(DEFAULT_WEIGHTS.values())
if abs(_weight_sum - 1.0) > 1e-6:
    DEFAULT_WEIGHTS = {k: v / _weight_sum for k, v in DEFAULT_WEIGHTS.items()}

VOLATILITY_RANGE = (0.10, 1.00); SEMIDEV_RANGE = (0.05, 0.70)
PE_RATIO_RANGE = (5.0, 50.0); PS_RATIO_RANGE = (0.5, 15.0)
PRICE_VS_SMA_RANGE = (-0.40, 0.40); BETA_RANGE = (0.5, 2.5); VIX_RANGE = (10.0, 50.0)
CVAR_ALPHA = 0.01; CVAR_RANGE = (0.02, 0.20)
GMD_RANGE = (0.0005, 0.015); MDD_RANGE = (0.05, 0.75); CDAR_RANGE = (0.07, 0.60)
MARKET_CAP_RANGE_LOG = (np.log10(50e6), np.log10(2e12)); VOLUME_RANGE_LOG = (np.log10(50000), np.log10(10e6))
VIX_CACHE_DURATION_SECONDS = 3600
HISTORY_CACHE_DURATION_SECONDS = 300 # Cache unified history for 5 mins
NEWS_SENTIMENT_CACHE_DURATION_SECONDS = 1800 # Cache combined news sentiment for 30 mins
ETS_FORECAST_CACHE_DURATION_SECONDS = 1800 # Cache ETS forecast results for 30 mins


# --- API Key Loading (HARDCODED as requested - NOT RECOMMENDED FOR SECURITY) ---
# WARNING: Exposing API keys like this is highly insecure. Use environment variables or secrets management.
OPENAI_API_KEY = "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" # Replace with your actual key
NEWS_API_KEY = 'd76a502fa00946bfae52c439094dd578' # Replace with your actual key
FMP_API_KEY = 'yZ8fSVddKFjMZH722j8ABVX9qjCUKbgF' # Replace with your actual key
# --- End of Hardcoded API Keys ---

# --- NEWS SENTIMENT Configuration ---
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'
FMP_ENDPOINT = 'https://financialmodelingprep.com/api'
RSS_FEEDS = [
    'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US', # Yahoo Finance Ticker Specific
    'https://www.reuters.com/pf/reuters/us/rss/technologySector', # Reuters Tech
    'http://feeds.marketwatch.com/marketwatch/topstories/', # MarketWatch Top Stories
    'https://www.cnbc.com/id/19854910/device/rss/rss.html', # CNBC Top News (needs filtering)
    'https://seekingalpha.com/feed.xml' # Seeking Alpha Top News (needs filtering)
]
NEWS_DAYS_BACK = 7 # How many days back to fetch news
NEWS_PAGE_SIZE = 30 # Articles per API source (max 100 for NewsAPI)
NEWS_FMP_LIMIT = 30 # Limit for FMP news
NEWS_MAX_ARTICLES_DISPLAY = 50 # Max articles to process after combining/deduplicating

# VADER Sentiment Analyzer (Global Instance)
vader_analyzer = SentimentIntensityAnalyzer()

# --- Riskfolio-Lib Import/Handling ---
# [Riskfolio import/fallback code remains the same]
try:
    import riskfolio.src.AuxFunctions as af
    import riskfolio.src.DBHT as db
    import riskfolio.src.GerberStatistic as gs
    import riskfolio.src.RiskFunctions as rk
    import riskfolio.src.OwaWeights as owa
    RISKFOLIO_AVAILABLE = True
    logging.info("Riskfolio-Lib found and imported successfully.")
    # Ensure fallback functions are defined if Riskfolio is missing
    if not all(hasattr(rk, func) for func in ['SemiDeviation', 'CDaR_Abs']):
        logging.warning("Missing required Riskfolio functions. Disabling factors.")
        if not hasattr(rk, 'SemiDeviation'): DEFAULT_WEIGHTS['semi_deviation'] = 0.0
        if not hasattr(rk, 'CDaR_Abs'): DEFAULT_WEIGHTS['cdar'] = 0.0
    if not hasattr(owa, 'owa_gmd'): DEFAULT_WEIGHTS['gmd'] = 0.0
    # Renormalize weights if any were disabled
    total_w = sum(DEFAULT_WEIGHTS.values())
    if total_w > 0 and abs(total_w - 1.0) > 1e-6:
        DEFAULT_WEIGHTS = {k: v / total_w for k, v in DEFAULT_WEIGHTS.items()}
    elif total_w <= 0:
        logging.error("All risk weights became zero after disabling Riskfolio factors!")

except ImportError:
    logging.warning("Riskfolio-Lib not installed or import failed. Advanced cov methods & risk factors disabled.")
    RISKFOLIO_AVAILABLE = False
    # Define dummy/fallback functions if Riskfolio is not available
    class af:
        @staticmethod
        def is_pos_def(x):
            try: x = np.array(x, dtype=float); return np.all(np.linalg.eigvalsh(x) >= -1e-8)
            except: return False
        @staticmethod
        def cov_fix(cov, method="clipped", threshold=1e-8):
            logging.warning("Using dummy cov_fix (eigenvalue clipping).");
            try: cov_arr = np.array(cov, dtype=float); eigvals, eigvecs = np.linalg.eigh(cov_arr); eigvals_clipped = np.maximum(eigvals, threshold); return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            except: return np.array(cov, dtype=float)
        @staticmethod
        def denoiseCov(*args, **kwargs): raise NotImplementedError("Riskfolio not available")
    class db:
         @staticmethod
         def PMFG_T2s(*args, **kwargs): raise NotImplementedError("Riskfolio not available")
         @staticmethod
         def j_LoGo(*args, **kwargs): raise NotImplementedError("Riskfolio not available")
    class gs:
         @staticmethod
         def gerber_cov_stat1(*args, **kwargs): raise NotImplementedError("Riskfolio not available")
         @staticmethod
         def gerber_cov_stat2(*args, **kwargs): raise NotImplementedError("Riskfolio not available")
    def _fallback_SemiDeviation(X):
        a = np.array(X, ndmin=1).flatten();
        if len(a) < 2: return np.nan
        mu = np.mean(a); diff = a - mu; downside_diff = diff[diff < 0];
        if len(downside_diff) < 1: return 0.0
        variance = np.sum(downside_diff**2) / max(1, len(a) - 1); return np.sqrt(variance)
    def _fallback_CDaR_Abs(X, alpha=0.01):
        a = np.array(X, ndmin=1).flatten();
        if len(a) < 2: return np.nan
        prices = np.insert(a, 0, 0); NAV = np.cumsum(prices) + 1; DD = []; peak = -np.inf;
        for i in NAV: peak = max(peak, i); DD.append(peak - i)
        if not DD or all(d <= 0 for d in DD): return 0.0
        sorted_DD = np.sort(np.array(DD)); index = int(np.ceil((1 - alpha) * len(sorted_DD)) - 1);
        index = max(0, min(index, len(sorted_DD) - 1)); return sorted_DD[index];
    def _fallback_owa_gmd(T): T_ = int(T); w_ = [2*i - 1 - T_ for i in range(1, T_ + 1)]; return (2 * np.array(w_) / max(1, T_ * (T_ - 1))).reshape(-1, 1)
    class rk: SemiDeviation = staticmethod(_fallback_SemiDeviation); CDaR_Abs = staticmethod(_fallback_CDaR_Abs)
    class owa: owa_gmd = staticmethod(_fallback_owa_gmd)
    logging.warning("Disabling Riskfolio factors: semi_deviation, cdar, gmd.")
    DEFAULT_WEIGHTS['semi_deviation'] = 0.00; DEFAULT_WEIGHTS['cdar'] = 0.00; DEFAULT_WEIGHTS['gmd'] = 0.00
    total_w = sum(DEFAULT_WEIGHTS.values());
    if total_w > 0: DEFAULT_WEIGHTS = {k: v / total_w for k, v in DEFAULT_WEIGHTS.items()}
    else: logging.error("All risk weights zero after disabling Riskfolio!")


# --- Streamlit Page Config ---
# Updated Title
st.set_page_config(page_title="📈 Financial Chat + Risk + News + Forecasting", layout="wide", initial_sidebar_state="expanded")

# --- API Key Validation ---
# Basic checks even with hardcoded keys
missing_keys = []
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    missing_keys.append("OpenAI API Key")
if not NEWS_API_KEY or len(NEWS_API_KEY) < 20:
    missing_keys.append("NewsAPI Key")
if not FMP_API_KEY or len(FMP_API_KEY) < 20:
    missing_keys.append("Financial Modeling Prep (FMP) API Key")

if missing_keys:
    keys_str = ', '.join(missing_keys)
    st.error(f"❌ Invalid or missing API keys: {keys_str}. Please check the hardcoded values.");
    logging.error(f"API keys invalid or missing: {keys_str}");
    if "OpenAI API Key" in missing_keys: st.stop()
    else:
         if "NewsAPI Key" in missing_keys: logging.warning("NewsAPI key missing/invalid, News Sentiment may be limited.")
         if "FMP API Key" in missing_keys: logging.warning("FMP API key missing/invalid, News Sentiment may be limited.")
else:
    logging.info("OpenAI, NewsAPI & FMP API Keys seem present.")
    logging.warning("Reminder: API Keys are hardcoded in this script, which is insecure.")


# --- Initialize Clients ---
try: client = OpenAI(api_key=OPENAI_API_KEY); logging.info("Initialized OpenAI client.")
except Exception as e: st.error(f"Error initializing OpenAI client: {e}"); logging.error(f"Initialization Error: {e}", exc_info=True); st.stop()

# --- Global Settings ---
MODEL_NAME = "gpt-4o"; MAX_TOKENS = 1200; TEMPERATURE = 0.5

# --- Helper Functions (Including Date Parsing) ---
# [parse_date function remains the same]
def parse_date(date_string):
    if not date_string: return None
    try:
        dt = date_parser.parse(date_string)
        return dt.isoformat().split('+')[0].split('.')[0] + "Z"
    except (ValueError, TypeError, OverflowError):
        logging.debug(f"Could not parse date: {date_string}")
        return None

# [get_stock_data function remains the same]
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    logging.info(f"Fetching Yahoo Finance summary data for {ticker}")
    try:
        yf_ticker = ticker.replace('$', '').upper(); logging.info(f"Requesting yfinance info for symbol: {yf_ticker}")
        ticker_obj = yf.Ticker(yf_ticker); info = ticker_obj.info
        if not info or not info.get('symbol'):
            logging.warning(f"Yahoo info for {ticker} empty or missing symbol. Checking history as fallback.")
            hist = ticker_obj.history(period="5d")
            if hist.empty: logging.warning(f"Ticker {ticker} not found or invalid on Yahoo Finance."); st.toast(f"⚠️ Couldn't find Yahoo equity data for '{ticker}'.", icon="⚠️"); return None
            else:
                 logging.warning(f"Ticker {ticker} info sparse/invalid, but history exists.")
                 if not info: info = {}
                 if 'symbol' not in info: info['symbol'] = yf_ticker
                 if 'quoteType' not in info: info['quoteType'] = 'EQUITY'
                 if info.get('currentPrice') is None and info.get('regularMarketPrice') is None and info.get('previousClose') is None and not hist.empty: info['currentPrice'] = hist['Close'].iloc[-1]; logging.info(f"Patched price for {ticker}.")

        quote_type = info.get('quoteType', 'N/A')
        if quote_type not in ['EQUITY', 'ETF']:
             logging.warning(f"Ticker {ticker} is not EQUITY or ETF (Type: {quote_type}). Forecasting may be unreliable.");
             if quote_type not in ['N/A', 'Undefined']:
                 st.toast(f"⚠️ '{ticker}' not a supported stock type (Type: {quote_type}). Features may be limited.", icon="⚠️")

        num_opinions = info.get("numberOfAnalystOpinions")
        data = {
            "ticker": info.get("symbol", yf_ticker).upper(), "companyName": info.get("shortName", info.get("longName", "N/A")), "sector": info.get("sector", "N/A"), "industry": info.get("industry", "N/A"), "quoteType": quote_type,
            "priceForDisplay": info.get("currentPrice", info.get("regularMarketPrice", info.get("regularMarketOpen", info.get("previousClose")))), "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"), "forwardPE": info.get("forwardPE"), "dividendYield": info.get("dividendYield"), "sma50": info.get("fiftyDayAverage"), "sma200": info.get("twoHundredDayAverage"),
            "beta": info.get("beta"), "dayLow": info.get("dayLow"), "dayHigh": info.get("dayHigh"), "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"), "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("volume", info.get("regularMarketVolume")), "recommendationKey": info.get("recommendationKey"), "targetMeanPrice": info.get("targetMeanPrice"),
            "targetLowPrice": info.get("targetLowPrice"), "targetHighPrice": info.get("targetHighPrice"), "numberOfAnalystOpinions": int(num_opinions) if num_opinions is not None and isinstance(num_opinions, (int, float)) and num_opinions > 0 else None,
            "website": info.get("website"), "longBusinessSummary": info.get("longBusinessSummary")
        }
        if data["priceForDisplay"] is None: logging.warning(f"Could not determine price for {ticker}.")
        logging.info(f"Successfully fetched Yahoo summary data for {data['ticker']} (Type: {data['quoteType']})"); return data
    except Exception as e:
        err_str = str(e).lower()
        if "no data found" in err_str or "404 client error" in err_str or "failed to decrypt" in err_str or "internal server error" in err_str: logging.warning(f"Yahoo Finance: Ticker {ticker} likely invalid/temp issue: {e}"); st.toast(f"⚠️ Couldn't find Yahoo data for '{ticker}'.", icon="⚠️")
        else: logging.error(f"Error fetching yfinance data for {ticker}: {e}", exc_info=True); st.toast(f"⚠️ Error fetching data from Yahoo for {ticker}.", icon="❌")
        return None

# [get_unified_yfinance_history function remains the same]
@st.cache_data(ttl=HISTORY_CACHE_DURATION_SECONDS)
def get_unified_yfinance_history(ticker: str, period="3y"):
    logging.info(f"Fetching UNIFIED yfinance history for {ticker}, period: {period}")
    try:
        yf_ticker_obj = yf.Ticker(ticker)
        df = yf_ticker_obj.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty: logging.warning(f"No data from unified yfinance history for {ticker}."); st.toast(f"⚠️ yfinance no unified history for {ticker}.", icon="⚠️"); return None
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             logging.warning(f"Missing columns in unified yfinance data for {ticker}. Found: {df.columns.tolist()}. Required: {required_cols}");
             return None
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            try: df.index = df.index.tz_convert('America/New_York').tz_localize(None)
            except Exception as tz_err: logging.warning(f"Unified yfinance index tz conversion failed: {tz_err}."); pass
        elif not isinstance(df.index, pd.DatetimeIndex): logging.warning(f"Unified yfinance index for {ticker} not DatetimeIndex.")
        min_rows_needed = 252
        if len(df) < min_rows_needed:
            logging.warning(f"Unified yfinance history for {ticker} has only {len(df)} rows (period={period}). Risk calculations might fail.")
        else:
            logging.info(f"Processed unified yfinance history for {ticker} ({len(df)} rows)")
        return df['Close'], df[required_cols] # Return Close separately
    except Exception as e: logging.error(f"General unified yfinance history fetch error for {ticker}: {e}", exc_info=True); st.toast(f"⚠️ Error fetching history data for {ticker}.", icon="❌"); return None, None

# [get_vix_cached function remains the same]
@st.cache_data(ttl=VIX_CACHE_DURATION_SECONDS)
def get_vix_cached():
    logging.info("Fetching new VIX value (or using cache)...");
    try:
        vix_ticker = yf.Ticker("^VIX"); vix_info = vix_ticker.info
        current_vix = vix_info.get('regularMarketPrice', vix_info.get('previousClose'))
        if current_vix is not None and isinstance(current_vix, (int, float)):
            logging.info(f"VIX from info: {current_vix:.2f}"); return float(current_vix)
        else:
            logging.warning("VIX info missing price, trying history.")
            hist = vix_ticker.history(period="1d", interval="1d");
            if not hist.empty:
                 last_close = hist['Close'].iloc[-1]
                 if last_close is not None and isinstance(last_close, (int, float)):
                     logging.info(f"VIX from history: {last_close:.2f}"); return float(last_close)
                 else: logging.warning(f"VIX history invalid: {last_close}")
            else: logging.warning("VIX history empty.")
        logging.warning("Could not get VIX value."); return None
    except Exception as e: logging.error(f"Error getting VIX: {e}"); logging.debug(traceback.format_exc()); return None

# [forecast_stock_ets_advanced function remains the same]
@st.cache_data(ttl=ETS_FORECAST_CACHE_DURATION_SECONDS)
def forecast_stock_ets_advanced( ticker: str, close_prices: pd.Series, forecast_days: int = 7, volatility_window_recent: int = 21, volatility_window_long: int = 252, volatility_threshold: float = 1.5, seasonal_period: int = 21, eval_test_size: int = 21 ):
    logger.info(f"[Forecast] Starting ETS forecast for {ticker} ({forecast_days} days)...")
    if close_prices is None or close_prices.empty:
        logger.error(f"[Forecast] No close price data provided for {ticker}.")
        return None, None, "Data Error - Missing Prices"
    min_data_required = max(volatility_window_long + 2, eval_test_size + seasonal_period + 1)
    if len(close_prices) < min_data_required:
        logger.error(f"[Forecast] Not enough historical data ({len(close_prices)} days) for robust ETS model. Need at least {min_data_required}.")
        return None, None, "Data Error - Insufficient Data"
    logger.info("[Forecast] Analyzing volatility...")
    use_seasonal = False
    try:
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_returns) >= volatility_window_long:
            recent_std = log_returns[-volatility_window_recent:].std()
            long_std = log_returns[-volatility_window_long:].std()
            if long_std > 1e-9 and (recent_std / long_std) > volatility_threshold:
                use_seasonal = True
                logger.info("[Forecast] High recent volatility detected relative to long term, enabling seasonality.")
            else: logger.info("[Forecast] Volatility within normal range, using non-seasonal model.")
        else: logger.warning("[Forecast] Not enough data for long-term volatility comparison, using non-seasonal model.")
    except Exception as e: logger.error(f"[Forecast] Error calculating volatility: {e}"); logger.warning("[Forecast] Proceeding with non-seasonal model due to volatility calculation error."); use_seasonal = False
    model_params = { 'trend': 'add', 'initialization_method': 'estimated', 'use_boxcox': False, 'damped_trend': True }
    if use_seasonal: model_params.update({ 'seasonal': 'add', 'seasonal_periods': seasonal_period })
    rmse = None; evaluation_summary = "Evaluation: Not performed (insufficient data or error)."
    if len(close_prices) > eval_test_size + seasonal_period + 1:
        train_eval = close_prices[:-eval_test_size]; test_eval = close_prices[-eval_test_size:]
        logger.info(f"[Forecast] Evaluating model on last {eval_test_size} days...")
        try:
            train_freq = train_eval.index.freq if hasattr(train_eval.index, 'freq') else None
            eval_model = ExponentialSmoothing(train_eval, freq=train_freq, **model_params)
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); fitted_eval = eval_model.fit(optimized=True)
            eval_forecast = fitted_eval.forecast(steps=eval_test_size)
            rmse = np.sqrt(mean_squared_error(test_eval, eval_forecast))
            avg_price_eval = test_eval.mean()
            mape = np.mean(np.abs((test_eval - eval_forecast) / test_eval)) * 100 if avg_price_eval > 0 else np.inf
            evaluation_summary = f"Evaluation (last {eval_test_size} days): RMSE={rmse:.2f} (MAPE={mape:.2f}%)"; logger.info(f"[Forecast] {evaluation_summary}")
        except Exception as e: logger.error(f"[Forecast] Error during evaluation fitting/forecasting: {e}"); evaluation_summary = f"Evaluation Error: {e}"; rmse = None
    else: logger.warning("[Forecast] Not enough data to perform separate evaluation.")
    logger.info("[Forecast] Fitting final model on full dataset...")
    try:
        final_freq = close_prices.index.freq if hasattr(close_prices.index, 'freq') else None
        final_model = ExponentialSmoothing(close_prices, freq=final_freq, **model_params)
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); fitted_final_model = final_model.fit(optimized=True)
        forecast_result = fitted_final_model.forecast(steps=forecast_days)
        if not close_prices.empty and isinstance(close_prices.index, pd.DatetimeIndex):
            last_date = close_prices.index[-1]
            try:
                 future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
                 if len(future_dates) == forecast_days: forecast_result.index = future_dates
                 else: logger.warning("[Forecast] Could not generate expected number of business dates for forecast index.")
            except Exception as date_err: logger.warning(f"[Forecast] Error generating forecast date index: {date_err}. Using default numerical index.")
        else: logger.warning("[Forecast] Cannot generate date index for forecast (input index issue). Using default numerical index.")
        model_used_desc = f"ETS({'A' if model_params.get('trend')=='add' else 'M'}, {'A' if model_params.get('seasonal')=='add' else 'N'}, {'N' if not use_seasonal else 'A'}{f' damped' if model_params.get('damped_trend') else ''})"
        logger.info(f"[Forecast] Forecast generation successful using {model_used_desc}.")
        return forecast_result, evaluation_summary, model_used_desc
    except Exception as e: logger.error(f"[Forecast] Error during final model fitting/forecasting: {e}"); return None, evaluation_summary, "Model Error"

# [Multi-Source News Sentiment functions remain the same]
# --- START: MULTI-SOURCE NEWS SENTIMENT FUNCTIONS ---
def analyze_sentiment_vader(text):
    if not text: return "Neutral", 0.0
    vs = vader_analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05: sentiment = "Positive"
    elif compound_score <= -0.05: sentiment = "Negative"
    else: sentiment = "Neutral"
    return sentiment, compound_score
def get_news_newsapi(ticker, api_key, days_back=NEWS_DAYS_BACK, page_size=NEWS_PAGE_SIZE):
    logging.info(f"[News] Fetching news from NewsAPI for '{ticker}'...")
    if not api_key or len(api_key) < 20: logging.warning("[News] NewsAPI key not set or seems invalid. Skipping NewsAPI."); return []
    to_date = datetime.now(); from_date = to_date - timedelta(days=days_back); from_date_str = from_date.strftime('%Y-%m-%d')
    query = f'"{ticker}" AND (stock OR shares OR earnings OR market OR business)'; params = { 'q': query, 'apiKey': api_key, 'from': from_date_str, 'sortBy': 'relevancy', 'language': 'en', 'pageSize': min(page_size, 100), }
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=15); response.raise_for_status(); data = response.json()
        if data.get('status') == 'ok':
            articles = data.get('articles', []); formatted_articles = []
            for article in articles:
                title = article.get('title');
                if not title or title == '[Removed]': continue
                formatted_articles.append({ 'title': title, 'description': article.get('description'), 'url': article.get('url'), 'publishedAt': parse_date(article.get('publishedAt')), 'source_api': 'NewsAPI', 'source_name': article.get('source', {}).get('name', 'NewsAPI') })
            logging.info(f"[News] NewsAPI: Found {len(formatted_articles)} articles (total results: {data.get('totalResults')})."); return formatted_articles
        else:
            if data.get('code') == 'rateLimited': logging.error("[News] Error from NewsAPI: Rate limit exceeded.")
            elif data.get('code') == 'maximumResultsReached': logging.warning("[News] Error from NewsAPI: Maximum results reached for plan.")
            else: logging.error(f"[News] Error from NewsAPI: {data.get('code')} - {data.get('message')}")
            return []
    except requests.exceptions.Timeout: logging.error("[News] NewsAPI Network/Request Error: Request timed out."); return []
    except requests.exceptions.RequestException as e: logging.error(f"[News] NewsAPI Network/Request Error: {e}"); return []
    except json.JSONDecodeError: logging.error("[News] Error: Could not decode JSON response from NewsAPI."); return []
    except Exception as e: logging.error(f"[News] An unexpected error occurred with NewsAPI: {e}", exc_info=True); return []
def get_news_fmp(ticker, api_key, limit=NEWS_FMP_LIMIT):
    logging.info(f"[News] Fetching news from Financial Modeling Prep for '{ticker}'...")
    if not api_key or len(api_key) < 20: logging.warning("[News] FMP API key not set or seems invalid. Skipping FMP."); return []
    fmp_news_url = f"{FMP_ENDPOINT}/v3/stock_news"; params = {'tickers': ticker, 'limit': limit, 'apikey': api_key}
    try:
        response = requests.get(fmp_news_url, params=params, timeout=15)
        if response.status_code != 200:
             if response.status_code == 401: logging.error(f"[News] FMP Request Error: Unauthorized (401). Check API key.")
             elif response.status_code == 403: logging.error(f"[News] FMP Request Error: Forbidden (403). Check API key or plan limits.")
             else: response.raise_for_status(); return []
        data = response.json()
        if isinstance(data, dict) and 'Error Message' in data:
            error_message = data['Error Message'];
            if "Limit Reach" in error_message: logging.error(f"[News] Error from FMP: API limit reached. {error_message}")
            else: logging.error(f"[News] Error from FMP: {error_message}")
            return []
        if not isinstance(data, list): logging.error(f"[News] Error from FMP: Unexpected response format. Expected list, got {type(data)}"); return []
        formatted_articles = []
        for article in data:
             title = article.get('title');
             if not title: continue
             formatted_articles.append({ 'title': title, 'description': article.get('text'), 'url': article.get('url'), 'publishedAt': parse_date(article.get('publishedDate')), 'source_api': 'FMP', 'source_name': article.get('site', 'Financial Modeling Prep') })
        logging.info(f"[News] FMP: Found {len(formatted_articles)} articles."); return formatted_articles
    except requests.exceptions.Timeout: logging.error("[News] FMP Network/Request Error: Request timed out."); return []
    except requests.exceptions.RequestException as e: logging.error(f"[News] FMP Network/Request Error: {e}"); return []
    except json.JSONDecodeError: logging.error(f"[News] Error: Could not decode JSON response from FMP. Raw: {response.text[:200]}..."); return []
    except Exception as e: logging.error(f"[News] An unexpected error occurred with FMP: {e}", exc_info=True); return []
def get_news_rss(ticker, feed_urls, days_back=NEWS_DAYS_BACK):
    logging.info("[News] Fetching news from RSS Feeds...")
    all_rss_articles = []; cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back);
    for url_template in feed_urls:
        url = url_template.replace('{ticker}', ticker) if '{ticker}' in url_template else url_template
        logging.debug(f"[News]   Parsing RSS feed: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; FinancialBot/1.0; +http://example.com/bot)'}
            feed_data = feedparser.parse(url, request_headers=headers, timeout=10)
            if feed_data.bozo: logging.warning(f"[News]     Warning: Malformed feed or error parsing {url}. Exception: {feed_data.bozo_exception}")
            for entry in feed_data.entries:
                title = entry.get('title');
                if not title: continue
                summary = entry.get('summary') or entry.get('description') or (entry.get('content')[0]['value'] if entry.get('content') else '')
                link = entry.get('link'); published_time_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                published_dt_aware = None; parsed_date_iso = None
                if published_time_struct:
                    try: ts = time.mktime(published_time_struct); published_dt_aware = datetime.fromtimestamp(ts, timezone.utc); parsed_date_iso = published_dt_aware.isoformat().replace('+00:00', 'Z')
                    except (TypeError, ValueError, OverflowError): pass
                elif entry.get('published'):
                    parsed_date_iso = parse_date(entry.get('published'))
                    if parsed_date_iso:
                         try: published_dt_aware = date_parser.isoparse(parsed_date_iso.replace('Z', '+00:00'))
                         except ValueError: pass
                if published_dt_aware and published_dt_aware < cutoff_date: continue
                is_ticker_specific_feed = '{ticker}' in url_template; ticker_lower = ticker.lower(); title_lower = title.lower(); summary_lower = summary.lower()
                needs_ticker_mention = not is_ticker_specific_feed; pattern = r'\b' + re.escape(ticker_lower) + r'\b|\(' + re.escape(ticker_lower) + r'\)|' + re.escape(f'${ticker_lower}') + r'\b'
                mentions_ticker = bool(re.search(pattern, title_lower) or re.search(pattern, summary_lower))
                if is_ticker_specific_feed or (needs_ticker_mention and mentions_ticker):
                     all_rss_articles.append({ 'title': title, 'description': summary, 'url': link, 'publishedAt': parsed_date_iso or 'N/A', 'source_api': 'RSS', 'source_name': feed_data.feed.get('title', url) })
                elif needs_ticker_mention and not mentions_ticker: logging.debug(f"[News]     RSS Skipping (General Feed, No Ticker Match): {title[:50]}...")
        except Exception as e: logging.error(f"[News]     Error processing RSS feed {url}: {e}")
        time.sleep(0.2)
    logging.info(f"[News] RSS Feeds: Found {len(all_rss_articles)} potentially relevant articles within date range."); return all_rss_articles
@st.cache_data(ttl=NEWS_SENTIMENT_CACHE_DURATION_SECONDS)
def get_multi_source_news_sentiment(ticker: str, news_api_key: str, fmp_api_key: str):
    logging.info(f"--- Starting Multi-Source News Sentiment Analysis for {ticker} ---"); start_time = time.time()
    newsapi_articles = get_news_newsapi(ticker, news_api_key); time.sleep(0.2); fmp_articles = get_news_fmp(ticker, fmp_api_key); time.sleep(0.2); rss_articles = get_news_rss(ticker, RSS_FEEDS)
    all_articles = newsapi_articles + fmp_articles + rss_articles; logging.info(f"[News] Total articles fetched across sources: {len(all_articles)}")
    seen_urls = set(); seen_titles_lower = set(); unique_articles = []
    for article in all_articles:
         url = article.get('url'); title = article.get('title'); title_lower = title.lower().strip() if title else None
         if url and url not in seen_urls: unique_articles.append(article); seen_urls.add(url);
         elif not url and title_lower and title_lower not in seen_titles_lower: unique_articles.append(article); seen_titles_lower.add(title_lower)
    logging.info(f"[News] Articles after deduplication: {len(unique_articles)}")
    def get_sort_key(article):
        date_str = article.get('publishedAt');
        if date_str and date_str != 'N/A':
            try: return date_parser.isoparse(date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError): return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)
    unique_articles.sort(key=get_sort_key, reverse=True); articles_to_analyze = unique_articles[:NEWS_MAX_ARTICLES_DISPLAY]
    logging.info(f"[News] Analyzing sentiment for latest {len(articles_to_analyze)} unique articles...")
    positive_count, negative_count, neutral_count = 0, 0, 0; compound_scores = []
    if not articles_to_analyze:
        logging.warning(f"[News] No unique articles found to analyze for {ticker}.")
        summary_str = f"Multi-Source News Sentiment ({NEWS_DAYS_BACK}d, VADER):\n  - No relevant news articles found for {ticker}."; counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0, 'avg_score': 0.0}; return summary_str, counts
    for article in articles_to_analyze:
        title = article.get('title', ''); description = article.get('description', ''); text_to_analyze = f"{title}. {description}" if description and len(description) > 20 else title
        sentiment_label, compound_score = analyze_sentiment_vader(text_to_analyze); compound_scores.append(compound_score)
        if sentiment_label == "Positive": positive_count += 1
        elif sentiment_label == "Negative": negative_count += 1
        else: neutral_count += 1
    total_analyzed = len(articles_to_analyze); avg_score = np.mean(compound_scores) if compound_scores else 0.0
    if positive_count > negative_count + neutral_count * 0.5: overall_bias = "Positive Bias ✅"
    elif negative_count > positive_count + neutral_count * 0.5: overall_bias = "Negative Bias ❌"
    else: overall_bias = "Neutral/Mixed Bias ⚖️"
    summary_str = f"Multi-Source News Sentiment ({NEWS_DAYS_BACK}d, VADER Analysis on ~{total_analyzed} unique articles):\n"; summary_str += f"  - ✅ Positive Articles: {positive_count}\n"; summary_str += f"  - ❌ Negative Articles: {negative_count}\n"; summary_str += f"  - ➖ Neutral Articles: {neutral_count}\n"; summary_str += f"  - Overall Sentiment (heuristic): {overall_bias}"
    counts = { 'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count, 'total': total_analyzed, 'avg_score': avg_score }; end_time = time.time()
    logging.info(f"[News] Multi-source sentiment analysis for {ticker} completed in {end_time - start_time:.2f} sec."); logging.info(f"[News] Result: P={positive_count}, N={negative_count}, Neut={neutral_count}, AvgScore={avg_score:.3f}"); return summary_str, counts
# --- END: MULTI-SOURCE NEWS SENTIMENT FUNCTIONS ---

# --- Other Helper Functions (Piotroski, Normalization, etc.) ---
# [get_financial_data, safe_get, get_stock_beta, normalize_score, calculate_sma, calculate_mdd functions remain the same]
def get_financial_data(ticker_obj, statement_type: str, periods: int = 2):
    try:
        if statement_type == 'balance_sheet': data = ticker_obj.balance_sheet
        elif statement_type == 'income_stmt': data = ticker_obj.income_stmt
        elif statement_type == 'cashflow': data = ticker_obj.cashflow
        else: logging.warning(f"Invalid statement type: {statement_type}"); return None
        if data is not None and not data.empty:
            actual_periods = min(periods, data.shape[1])
            if actual_periods >= 1: return data.iloc[:, :actual_periods]
            else: logging.warning(f"[{ticker_obj.ticker}] Report '{statement_type}' has no columns."); return None
        else: logging.warning(f"[{ticker_obj.ticker}] Report '{statement_type}' is empty/unavailable."); return None
    except AttributeError: logging.warning(f"[{ticker_obj.ticker}] Statement '{statement_type}' not found."); return None
    except Exception as e: logging.warning(f"[{ticker_obj.ticker}] Error getting report '{statement_type}': {e}"); return None
def safe_get(series: pd.Series, key: str, default=None):
    if series is None: return default
    value = series.get(key, default)
    if pd.isna(value): return default
    if isinstance(value, str):
        try: return pd.to_numeric(value)
        except ValueError: return default
    return value
def get_stock_beta(ticker_info: dict):
    if not ticker_info: return None
    beta = ticker_info.get('beta', ticker_info.get('beta3Year'))
    if beta is not None and not isinstance(beta, (int, float)):
        try: beta = float(beta)
        except (ValueError, TypeError): beta = None
    return beta
def normalize_score(value, range_min, range_max, higher_is_riskier=True, is_log_range=False):
    if value is None or pd.isna(value): return 50.0
    current_min, current_max = range_min, range_max; value_to_norm = value
    if is_log_range:
        if value > 1e-9:
            try: value_to_norm = np.log10(value); logging.debug(f"Normalize Log: {value:.2f} -> {value_to_norm:.2f}")
            except Exception as e: logging.warning(f"Log10 failed: {e}. Using original."); value_to_norm = value;
        else: return 0.0 if higher_is_riskier else 100.0
    if abs(current_max - current_min) < 1e-9: return 50.0
    clipped_value = np.clip(value_to_norm, current_min, current_max)
    normalized = (clipped_value - current_min) / (current_max - current_min)
    risk_score = np.clip(normalized * 100 if higher_is_riskier else (1 - normalized) * 100, 0, 100)
    logging.debug(f"Normalize: Val={value:.2f}, Range=({range_min:.2f},{range_max:.2f}), Log={is_log_range}, HigherRisk={higher_is_riskier} => Score={risk_score:.2f}")
    return risk_score
def calculate_sma(data: pd.Series, window: int):
    if data is None or data.empty or len(data) < window: return None
    try: return data.rolling(window=window).mean().iloc[-1]
    except Exception as e: logging.warning(f"SMA{window} error: {e}"); return None
def calculate_mdd(prices: pd.Series):
    if prices is None or prices.empty or len(prices) < 2: return None
    try:
        cumulative_max = prices.cummax(); drawdown = (prices - cumulative_max) / cumulative_max; max_drawdown = drawdown.min(); return min(0.0, max_drawdown) if pd.notna(max_drawdown) else None
    except Exception as e: logging.warning(f"MDD error: {e}"); return None

# [calculate_piotroski_f_score function remains the same]
@st.cache_data(ttl=3600*6)
def calculate_piotroski_f_score(ticker_symbol: str):
    logging.info(f"[{ticker_symbol}] Calculating Piotroski F-Score (or cache)..."); score = 0; details = {}
    try:
        ticker = yf.Ticker(ticker_symbol); income = get_financial_data(ticker, 'income_stmt', 2); balance = get_financial_data(ticker, 'balance_sheet', 2); cashflow = get_financial_data(ticker, 'cashflow', 2)
        if income is None or balance is None or cashflow is None or \
           income.shape[1] < 2 or balance.shape[1] < 2 or cashflow.shape[1] < 2: logging.warning(f"[{ticker_symbol}] Insufficient financials for F-Score."); return None, details
        inc_t, inc_tm1 = income.iloc[:, 0], income.iloc[:, 1]; bal_t, bal_tm1 = balance.iloc[:, 0], balance.iloc[:, 1]; cf_t, cf_tm1 = cashflow.iloc[:, 0], cashflow.iloc[:, 1]
        net_income_t = safe_get(inc_t, 'Net Income', 0); details['NI > 0']=(net_income_t is not None and net_income_t > 0); score += details['NI > 0']
        op_cashflow_t=safe_get(cf_t, 'Operating Cash Flow', safe_get(cf_t, 'Cash Flow From Continuing Operating Activities', 0)); details['OCF > 0']=(op_cashflow_t is not None and op_cashflow_t > 0); score += details['OCF > 0']
        assets_t=safe_get(bal_t, 'Total Assets', 0); assets_tm1=safe_get(bal_tm1, 'Total Assets', 0); avg_assets_t=(assets_t + assets_tm1) / 2 if assets_t!=0 and assets_tm1!=0 else (assets_t or assets_tm1 or 0); avg_assets_tm1 = assets_tm1; roa_check=False
        if avg_assets_t != 0 and avg_assets_tm1 != 0: net_income_tm1=safe_get(inc_tm1, 'Net Income', 0); roa_t=net_income_t/avg_assets_t if avg_assets_t != 0 else 0; roa_tm1=net_income_tm1/avg_assets_tm1 if avg_assets_tm1 != 0 else 0; roa_check=(roa_t > roa_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate ROA change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta ROA > 0']=roa_check; score += details['Delta ROA > 0']
        details['OCF > NI']=(op_cashflow_t > net_income_t) if op_cashflow_t is not None and net_income_t is not None else False; score += details['OCF > NI']
        debt_lt_t=safe_get(bal_t, 'Long Term Debt And Capital Lease Obligation', safe_get(bal_t, 'Long Term Debt', 0)); debt_lt_tm1=safe_get(bal_tm1, 'Long Term Debt And Capital Lease Obligation', safe_get(bal_tm1, 'Long Term Debt', 0)); leverage_check=False
        if avg_assets_t != 0 and avg_assets_tm1 != 0: leverage_t=debt_lt_t/avg_assets_t if debt_lt_t is not None and avg_assets_t != 0 else np.inf; leverage_tm1=debt_lt_tm1/avg_assets_tm1 if debt_lt_tm1 is not None and avg_assets_tm1 != 0 else np.inf; leverage_check=(leverage_t < leverage_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate Leverage change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta Leverage < 0']=leverage_check; score += details['Delta Leverage < 0']
        current_assets_t=safe_get(bal_t, 'Current Assets', 0); current_liab_t=safe_get(bal_t, 'Current Liabilities', 0); current_ratio_t=current_assets_t/current_liab_t if current_liab_t!=0 and current_assets_t is not None else 0; current_assets_tm1=safe_get(bal_tm1, 'Current Assets', 0); current_liab_tm1=safe_get(bal_tm1, 'Current Liabilities', 0); current_ratio_tm1=current_assets_tm1/current_liab_tm1 if current_liab_tm1!=0 and current_assets_tm1 is not None else 0; details['Delta Current Ratio > 0']=(current_ratio_t > current_ratio_tm1); score += details['Delta Current Ratio > 0']
        shares_t=safe_get(bal_t, 'Share Issued', safe_get(inc_t,'Diluted Average Shares',None)); shares_tm1=safe_get(bal_tm1, 'Share Issued', safe_get(inc_tm1,'Diluted Average Shares',None)); shares_check=None
        if shares_t is not None and shares_tm1 is not None and shares_tm1 > 0: shares_check=(shares_t <= shares_tm1 * 1.01); details['Shares Issued Not Increased']=shares_check
        else:
            logging.warning(f"[{ticker_symbol}] Shares Issued not found or zero, using equity proxy for F-Score."); equity_t=safe_get(bal_t, 'Stockholders Equity', 0); equity_tm1=safe_get(bal_tm1, 'Stockholders Equity', 0); ni_for_calc=net_income_t if net_income_t is not None and pd.notna(net_income_t) else 0; equity_growth_non_re=(equity_t - equity_tm1) - ni_for_calc if equity_t is not None and equity_tm1 is not None else None
            if equity_growth_non_re is not None:
                 if equity_tm1 is not None and equity_tm1 != 0: shares_check=(equity_growth_non_re < equity_tm1 * 0.02)
                 else: shares_check=(equity_growth_non_re < 1e6)
                 details['Shares Issued Not Increased (Equity Proxy)'] = shares_check
            else: logging.warning(f"[{ticker_symbol}] Cannot calculate equity growth for F-Score shares check. Equity: t={equity_t}, tm1={equity_tm1}, NI: {ni_for_calc}"); details['Shares Issued Not Increased (Equity Proxy)'] = False
        score += shares_check if shares_check is not None else 0
        gross_profit_t=safe_get(inc_t, 'Gross Profit', 0); revenue_t=safe_get(inc_t, 'Total Revenue', safe_get(inc_t,'Operating Revenue', 0)); gross_margin_t=gross_profit_t/revenue_t if revenue_t!=0 and gross_profit_t is not None else 0; gross_profit_tm1=safe_get(inc_tm1, 'Gross Profit', 0); revenue_tm1=safe_get(inc_tm1, 'Total Revenue', safe_get(inc_tm1,'Operating Revenue', 0)); gross_margin_tm1=gross_profit_tm1/revenue_tm1 if revenue_tm1!=0 and gross_profit_tm1 is not None else 0; details['Delta Gross Margin > 0']=(gross_margin_t > gross_margin_tm1); score += details['Delta Gross Margin > 0']
        turnover_check=False
        if avg_assets_t != 0 and avg_assets_tm1 != 0: asset_turnover_t=revenue_t/avg_assets_t if revenue_t is not None and avg_assets_t != 0 else 0; asset_turnover_tm1=revenue_tm1/avg_assets_tm1 if revenue_tm1 is not None and avg_assets_tm1 != 0 else 0; turnover_check=(asset_turnover_t > asset_turnover_tm1)
        else: logging.warning(f"[{ticker_symbol}] Cannot calculate Asset Turnover change (F-Score). Assets: t={assets_t}, tm1={assets_tm1}")
        details['Delta Asset Turnover > 0']=turnover_check; score += details['Delta Asset Turnover > 0']
        logging.info(f"[{ticker_symbol}] Piotroski F-Score calculated: {score}/9"); logging.debug(f"[{ticker_symbol}] F-Score Details: {details}"); return score, details
    except Exception as e: logging.error(f"[{ticker_symbol}] General F-Score error: {e}", exc_info=True); logging.debug(traceback.format_exc()); return None, details

# [Risk Score functions: owa_cvar, covar_matrix, calculate_owa_risk, calculate_dynamic_risk_score, get_risk_category functions remain the same]
# ... Risk Score Calculation functions ...
def owa_cvar(T, alpha=0.01):
    T_ = int(T);
    if T_ == 0: return np.array([]).reshape(-1, 1)
    if alpha <= 0 or alpha >= 1: raise ValueError("alpha must be between 0 and 1")
    k = int(np.ceil(T_ * alpha));
    if k == 0: return np.zeros((T_, 1))
    elif k > T_: k = T_
    w_ = np.zeros((T_, 1)); tail_strict_count = int(np.floor(T_ * alpha));
    if tail_strict_count > 0: w_[:tail_strict_count, :] = -1 / (T_ * alpha)
    k_idx = k - 1; remaining_weight_to_assign = -1 - np.sum(w_[:k_idx, :])
    if k_idx < T_: w_[k_idx, :] = remaining_weight_to_assign
    return w_
def covar_matrix(X, method="hist", d=0.94, alpha=0.1, bWidth=0.01, detone=False, mkt_comp=1, threshold=0.5):
    if not isinstance(X, pd.DataFrame): raise ValueError("X must be a DataFrame")
    assets = X.columns.tolist(); n_assets = len(assets); cov = None
    try:
        if method == "hist": cov = np.cov(X.to_numpy(), rowvar=False)
        elif method == "semi": T, N = X.shape; mu = X.mean().to_numpy().reshape(1, -1); a = X.to_numpy() - np.repeat(mu, T, axis=0); a = np.minimum(a, np.zeros_like(a)); cov = 1/(T - 1) * a.T @ a
        elif method == "ewma1": cov = X.ewm(alpha=1-d, min_periods=max(1,n_assets)).cov(); item = cov.index.get_level_values(0)[-1]; cov = cov.loc[(item, slice(None)), :]
        elif method == "ewma2": cov = X.ewm(alpha=1-d, adjust=False, min_periods=max(1,n_assets)).cov(); item = cov.index.get_level_values(0)[-1]; cov = cov.loc[(item, slice(None)), :]
        elif method == "ledoit": lw = skcov.LedoitWolf(); lw.fit(X); cov = lw.covariance_
        elif method == "oas": oas = skcov.OAS(); oas.fit(X); cov = oas.covariance_
        elif method == "shrunk": sc = skcov.ShrunkCovariance(shrinkage=alpha); sc.fit(X); cov = sc.covariance_
        elif method == "gl": gl = skcov.GraphicalLassoCV(); gl.fit(X); cov = gl.covariance_
        elif method == "jlogo":
            if not RISKFOLIO_AVAILABLE: raise ModuleNotFoundError("jlogo requires Riskfolio-Lib")
            S=np.cov(X.to_numpy(), rowvar=False); R=np.corrcoef(X.to_numpy(), rowvar=False); D=np.sqrt(np.clip((1-R)/2, a_min=0.0, a_max=1.0)); np.fill_diagonal(D, 0); D=(D + D.T)/2; Sim=1 - D**2; (_, _, separators, cliques, _) = db.PMFG_T2s(Sim, nargout=4); cov = db.j_LoGo(S, separators, cliques); cov = np.linalg.inv(cov)
        elif method in ["fixed", "spectral", "shrink"]:
            if not RISKFOLIO_AVAILABLE: raise ModuleNotFoundError("Denoising requires Riskfolio-Lib")
            cov_hist=np.cov(X.to_numpy(), rowvar=False); T, N = X.shape; q = T / N; cov=af.denoiseCov(cov_hist, q, kind=method, bWidth=bWidth, detone=detone, mkt_comp=int(mkt_comp), alpha=alpha)
        elif method == "gerber1":
            if not RISKFOLIO_AVAILABLE: raise ModuleNotFoundError("gerber1 requires Riskfolio-Lib");
            cov = gs.gerber_cov_stat1(X, threshold=threshold)
        elif method == "gerber2":
            if not RISKFOLIO_AVAILABLE: raise ModuleNotFoundError("gerber2 requires Riskfolio-Lib");
            cov = gs.gerber_cov_stat2(X, threshold=threshold)
        else: raise ValueError(f"Unknown covariance method: {method}")
        if cov is None: raise ValueError(f"Covariance calculation failed for method: {method}")
        if not isinstance(cov, np.ndarray): cov = cov.to_numpy()
        cov = pd.DataFrame(np.array(cov, ndmin=2), columns=assets, index=assets)
        if not af.is_pos_def(cov.values): logging.warning(f"Cov matrix (method: {method}) not positive semidefinite. Fixing."); cov_fixed = af.cov_fix(cov.values, method="clipped"); cov = pd.DataFrame(cov_fixed, index=assets, columns=assets)
        return cov
    except ModuleNotFoundError as e: logging.error(f"Cannot use '{method}': {e}. Falling back to 'hist'."); return covar_matrix(X, method='hist')
    except Exception as e: logging.error(f"Error calculating cov '{method}': {e}"); logging.debug(traceback.format_exc()); return covar_matrix(X, method='hist')
def calculate_owa_risk(returns: pd.Series, owa_weights_func, **kwargs):
    if returns is None or returns.empty: return None
    try: T = len(returns);
    except Exception as e: logging.error(f"Error getting length for OWA risk: {e}"); return None
    if T < 2: return None
    try:
        sorted_returns = np.sort(returns.values); owa_weights = owa_weights_func(T, **kwargs)
        if owa_weights is None or owa_weights.size != T: logging.warning(f"Could not calc OWA weights for {owa_weights_func.__name__} T={T}"); return None
        risk_value = -np.dot(owa_weights.flatten(), sorted_returns.flatten()); return risk_value
    except Exception as e: logging.error(f"Error in OWA risk ({owa_weights_func.__name__}): {e}"); logging.debug(traceback.format_exc()); return None
@st.cache_data(ttl=600)
def calculate_dynamic_risk_score(ticker_symbol: str, hist_df: pd.DataFrame, weights: dict = None, cov_method: str = 'hist', vol_window: int = 90):
    try:
        logging.info(f"[{ticker_symbol}] Risk score func: Using pre-fetched history. Fetching yfinance info..."); ticker = yf.Ticker(ticker_symbol);
        try: info = ticker.info;
        except Exception as info_err: logging.warning(f"[{ticker_symbol}] Risk score: Info fetch failed: {info_err}."); info = None
        if hist_df is None or hist_df.empty or 'Close' not in hist_df.columns: logging.error(f"[{ticker_symbol}] FAILED Risk Score: Invalid/Empty FULL history DataFrame provided."); return None, {}, 0.0
        if info is None and not hist_df.empty: info = {'symbol': ticker_symbol, 'quoteType': 'EQUITY', 'currentPrice': hist_df['Close'].iloc[-1]}; logging.info(f"[{ticker_symbol}] Reconstructed minimal info for risk calc.")
        elif info is None: logging.error(f"[{ticker_symbol}] FAILED Risk Score: No valid info or history."); return None, {}, 0.0
        returns_1y = hist_df['Close'].pct_change().dropna(); logging.info(f"[{ticker_symbol}] Risk score: Valid return rows for risk: {len(returns_1y)}")
        min_returns_needed_advanced_risk = max(vol_window, int(np.ceil(1 / CVAR_ALPHA)) + 5); can_calc_advanced_risk = len(returns_1y) >= min_returns_needed_advanced_risk
        if not can_calc_advanced_risk: logging.warning(f"[{ticker_symbol}] Insufficient returns ({len(returns_1y)} < {min_returns_needed_advanced_risk}) for advanced risk calcs (CVaR, CDaR, GMD).")
        last_price = hist_df['Close'].iloc[-1] if not hist_df.empty else info.get('currentPrice', info.get('regularMarketPrice')); vix_value = get_vix_cached()
    except Exception as e: logging.error(f"[{ticker_symbol}] FAILED Risk Score during setup/info fetch: {e}", exc_info=True); return None, {}, 0.0
    current_weights = weights if weights is not None else DEFAULT_WEIGHTS.copy(); intermediate_scores = {}; available_factors_weight = 0.0
    # Volatility
    vol = None; vol_ann = None; vol_score = None
    if len(returns_1y) >= vol_window:
        try: returns_for_vol = returns_1y.tail(vol_window); returns_df = returns_for_vol.to_frame(name=ticker_symbol); cov_mat = covar_matrix(returns_df, method=cov_method)
        except Exception as e: logging.error(f"[{ticker_symbol}] Volatility setup/cov error: {e}"); cov_mat = None
        if cov_mat is not None and not cov_mat.empty:
            try: variance = cov_mat.iloc[0, 0];
            except IndexError: logging.error(f"[{ticker_symbol}] Covariance matrix index error."); variance = None
            if variance is not None and pd.notna(variance) and variance >= 0: vol = np.sqrt(variance); vol_ann = vol * np.sqrt(252)
            else: logging.warning(f"[{ticker_symbol}] Variance calc invalid: {variance}")
        else: logging.warning(f"[{ticker_symbol}] Cov matrix None/empty for '{cov_method}'.")
    else: logging.warning(f"[{ticker_symbol}] Volatility calc skipped: insufficient returns ({len(returns_1y)} < {vol_window}).")
    if vol_ann is not None: vol_score = normalize_score(vol_ann, VOLATILITY_RANGE[0], VOLATILITY_RANGE[1], True); intermediate_scores['volatility'] = vol_score; available_factors_weight += current_weights.get('volatility', 0); logging.info(f"[{ticker_symbol}] Volatility ({cov_method}, ann.): {vol_ann:.4f} -> Score: {vol_score:.2f}")
    else: intermediate_scores['volatility'] = None
    # Semi Deviation
    semi_dev = None; semi_dev_score = None
    if len(returns_1y) >= 2 and current_weights.get('semi_deviation', 0) > 1e-6:
         try: semi_dev_daily = rk.SemiDeviation(returns_1y.values);
         except Exception as e: logging.error(f"[{ticker_symbol}] SemiDeviation error: {e}", exc_info=True); semi_dev_daily = None
         if semi_dev_daily is not None and pd.notna(semi_dev_daily): semi_dev = semi_dev_daily * np.sqrt(252)
         else: logging.warning(f"[{ticker_symbol}] SemiDeviation daily calc None/NaN.")
    else: logging.warning(f"[{ticker_symbol}] SemiDeviation calc skipped.")
    if semi_dev is not None: semi_dev_score = normalize_score(semi_dev, SEMIDEV_RANGE[0], SEMIDEV_RANGE[1], True); intermediate_scores['semi_deviation'] = semi_dev_score; available_factors_weight += current_weights.get('semi_deviation', 0); logging.info(f"[{ticker_symbol}] Semi Deviation (ann.): {semi_dev:.4f} -> Score: {semi_dev_score:.2f}")
    else: intermediate_scores['semi_deviation'] = None
    # Market Cap
    market_cap_raw = info.get('marketCap'); mcap_score = None
    if market_cap_raw is not None and isinstance(market_cap_raw, (int, float)) and market_cap_raw > 0: mcap_score = normalize_score(market_cap_raw, MARKET_CAP_RANGE_LOG[0], MARKET_CAP_RANGE_LOG[1], False, True); intermediate_scores['market_cap'] = mcap_score; available_factors_weight += current_weights.get('market_cap', 0); logging.info(f"[{ticker_symbol}] Market Cap: {market_cap_raw:,.0f} -> Score: {mcap_score:.2f}")
    else: intermediate_scores['market_cap'] = None; logging.warning(f"[{ticker_symbol}] Market Cap unavailable: {market_cap_raw}")
    # Liquidity
    volume_raw = info.get('averageDailyVolume10Day', info.get('averageVolume', info.get('regularMarketVolume'))); liquidity_score = None
    if volume_raw is not None and isinstance(volume_raw, (int, float)) and volume_raw > 0: liquidity_score = normalize_score(volume_raw, VOLUME_RANGE_LOG[0], VOLUME_RANGE_LOG[1], False, True); intermediate_scores['liquidity'] = liquidity_score; available_factors_weight += current_weights.get('liquidity', 0); logging.info(f"[{ticker_symbol}] Avg Volume: {volume_raw:,.0f} -> Score: {liquidity_score:.2f}")
    else: intermediate_scores['liquidity'] = None; logging.warning(f"[{ticker_symbol}] Avg Volume unavailable: {volume_raw}")
    # PE/PS
    pe_ratio = info.get('trailingPE'); ps_ratio = info.get('priceToSalesTrailing12Months'); valuation_score = None
    if pe_ratio is not None and isinstance(pe_ratio, (int, float)) and pd.notna(pe_ratio):
        if pe_ratio > 0: valuation_score = normalize_score(pe_ratio, PE_RATIO_RANGE[0], PE_RATIO_RANGE[1], True); logging.info(f"[{ticker_symbol}] Using PE: {pe_ratio:.2f} -> Score: {valuation_score:.2f}")
        elif pe_ratio < 0: valuation_score = 100.0; logging.info(f"[{ticker_symbol}] Negative PE -> Score: 100.00")
        else: logging.warning(f"[{ticker_symbol}] PE is zero.")
    elif ps_ratio is not None and isinstance(ps_ratio, (int, float)) and pd.notna(ps_ratio) and ps_ratio > 0: valuation_score = normalize_score(ps_ratio, PS_RATIO_RANGE[0], PS_RATIO_RANGE[1], True); logging.info(f"[{ticker_symbol}] Using P/S: {ps_ratio:.2f} -> Score: {valuation_score:.2f}")
    else: logging.warning(f"[{ticker_symbol}] No valid PE/PS (PE: {pe_ratio}, PS: {ps_ratio})."); valuation_score = None
    if valuation_score is not None: intermediate_scores['pe_or_ps'] = valuation_score; available_factors_weight += current_weights.get('pe_or_ps', 0)
    else: intermediate_scores['pe_or_ps'] = None
    # Price vs SMA
    sma200 = calculate_sma(hist_df['Close'], 200); price_vs_sma_score = None
    if sma200 is not None and last_price is not None and sma200 > 1e-6: price_diff_pct = (last_price - sma200) / sma200; price_vs_sma_score = normalize_score(price_diff_pct, PRICE_VS_SMA_RANGE[0], PRICE_VS_SMA_RANGE[1], False); intermediate_scores['price_vs_sma'] = price_vs_sma_score; available_factors_weight += current_weights.get('price_vs_sma', 0); logging.info(f"[{ticker_symbol}] Price/SMA200 (% diff: {price_diff_pct*100:.1f}%) -> Score: {price_vs_sma_score:.2f}")
    else: intermediate_scores['price_vs_sma'] = None; logging.warning(f"[{ticker_symbol}] SMA200/Price unavailable.")
    # Beta
    beta = get_stock_beta(info); beta_score = None
    if beta is not None: beta_score = normalize_score(beta, BETA_RANGE[0], BETA_RANGE[1], True); intermediate_scores['beta'] = beta_score; available_factors_weight += current_weights.get('beta', 0); logging.info(f"[{ticker_symbol}] Beta: {beta:.2f} -> Score: {beta_score:.2f}")
    else: intermediate_scores['beta'] = None; logging.warning(f"[{ticker_symbol}] Beta unavailable.")
    # Piotroski
    f_score, _ = calculate_piotroski_f_score(ticker_symbol); piotroski_risk_score = None
    if f_score is not None: piotroski_risk_score = normalize_score(f_score, 0, 9, False); intermediate_scores['piotroski'] = piotroski_risk_score; available_factors_weight += current_weights.get('piotroski', 0); logging.info(f"[{ticker_symbol}] Piotroski F-Score: {f_score}/9 -> Score: {piotroski_risk_score:.2f}")
    else: intermediate_scores['piotroski'] = None; logging.warning(f"[{ticker_symbol}] Piotroski F-Score calc failed.")
    # VIX
    vix_score = None
    if vix_value is not None: vix_score = normalize_score(vix_value, VIX_RANGE[0], VIX_RANGE[1], True); intermediate_scores['vix'] = vix_score; available_factors_weight += current_weights.get('vix', 0); logging.info(f"[{ticker_symbol}] VIX: {vix_value:.2f} -> Score: {vix_score:.2f}")
    else: intermediate_scores['vix'] = None; logging.warning(f"[{ticker_symbol}] VIX unavailable.")
    # CVaR
    cvar_score = None
    if can_calc_advanced_risk and current_weights.get('cvar', 0) > 1e-6:
        try: cvar_val = calculate_owa_risk(returns_1y, owa_cvar, alpha=CVAR_ALPHA)
        except Exception as e: logging.error(f"[{ticker_symbol}] Error calling CVaR: {e}", exc_info=True); cvar_val = None
        if cvar_val is not None and pd.notna(cvar_val): cvar_abs = abs(cvar_val); cvar_score = normalize_score(cvar_abs, CVAR_RANGE[0], CVAR_RANGE[1], True); intermediate_scores['cvar'] = cvar_score; available_factors_weight += current_weights.get('cvar', 0); logging.info(f"[{ticker_symbol}] CVaR ({CVAR_ALPHA*100:.0f}%, daily): {cvar_val:.4f} -> Score: {cvar_score:.2f}")
        else: intermediate_scores['cvar'] = None; logging.warning(f"[{ticker_symbol}] CVaR calc None/NaN.")
    else: intermediate_scores['cvar'] = None; logging.warning(f"[{ticker_symbol}] CVaR calc skipped (low returns or zero weight).")
    # CDaR
    cdar_score = None
    if can_calc_advanced_risk and current_weights.get('cdar', 0) > 1e-6:
         try: cdar_val = rk.CDaR_Abs(returns_1y.values, alpha=CVAR_ALPHA)
         except Exception as e: logging.error(f"[{ticker_symbol}] Error calculating CDaR_Abs: {e}", exc_info=True); cdar_val = None
         if cdar_val is not None and pd.notna(cdar_val): cdar_score = normalize_score(cdar_val, CDAR_RANGE[0], CDAR_RANGE[1], True); intermediate_scores['cdar'] = cdar_score; available_factors_weight += current_weights.get('cdar', 0); logging.info(f"[{ticker_symbol}] CDaR ({CVAR_ALPHA*100:.0f}%): {cdar_val:.4f} -> Score: {cdar_score:.2f}")
         else: intermediate_scores['cdar'] = None; logging.warning(f"[{ticker_symbol}] CDaR calc None/NaN.")
    else: intermediate_scores['cdar'] = None; logging.warning(f"[{ticker_symbol}] CDaR calc skipped (low returns or zero weight).")
    # MDD
    mdd_score = None
    if not hist_df.empty:
         mdd_val = calculate_mdd(hist_df['Close'])
         if mdd_val is not None and pd.notna(mdd_val): mdd_abs = abs(mdd_val); mdd_score = normalize_score(mdd_abs, MDD_RANGE[0], MDD_RANGE[1], True); intermediate_scores['mdd'] = mdd_score; available_factors_weight += current_weights.get('mdd', 0); logging.info(f"[{ticker_symbol}] MDD (hist period): {mdd_val*100:.2f}% -> Score: {mdd_score:.2f}")
         else: intermediate_scores['mdd'] = None; logging.warning(f"[{ticker_symbol}] MDD calc failed or None/NaN.")
    else: intermediate_scores['mdd'] = None; logging.warning(f"[{ticker_symbol}] MDD calc skipped (no history).")
    # GMD
    gmd_score = None
    if can_calc_advanced_risk and current_weights.get('gmd', 0) > 1e-6:
        try: owa_func = owa.owa_gmd if RISKFOLIO_AVAILABLE else _fallback_owa_gmd; gmd_val = calculate_owa_risk(returns_1y, owa_func)
        except Exception as e: logging.error(f"[{ticker_symbol}] Error calculating GMD: {e}", exc_info=True); gmd_val = None
        if gmd_val is not None and pd.notna(gmd_val): gmd_score = normalize_score(gmd_val, GMD_RANGE[0], GMD_RANGE[1], True); intermediate_scores['gmd'] = gmd_score; available_factors_weight += current_weights.get('gmd', 0); logging.info(f"[{ticker_symbol}] GMD (daily): {gmd_val:.4f} -> Score: {gmd_score:.2f}")
        else: intermediate_scores['gmd'] = None; logging.warning(f"[{ticker_symbol}] GMD calc None/NaN.")
    else: intermediate_scores['gmd'] = None; logging.warning(f"[{ticker_symbol}] GMD calc skipped (low returns or zero weight).")

    # Final Score Calculation
    final_score = 0.0; valid_scores_count = sum(1 for score in intermediate_scores.values() if score is not None); logging.info(f"[{ticker_symbol}] Available Factors Original Weight Sum: {available_factors_weight:.3f}, Valid Scores: {valid_scores_count}")
    valid_factors = {k: v for k, v in current_weights.items() if intermediate_scores.get(k) is not None}; total_valid_original_weight = sum(valid_factors.values())
    if total_valid_original_weight < 0.01 or valid_scores_count == 0: logging.error(f"[{ticker_symbol}] FAILED Risk Score: Low valid factor weight ({total_valid_original_weight:.3f})."); return None, intermediate_scores, total_valid_original_weight
    renormalized_weights = {k: v / total_valid_original_weight for k, v in valid_factors.items()}
    for factor, score in intermediate_scores.items():
        if score is not None and factor in renormalized_weights: renormalized_weight = renormalized_weights[factor]; final_score += score * renormalized_weight; logging.debug(f"[{ticker_symbol}] Score contribution: '{factor}', Score={score:.2f}, RenormW={renormalized_weight:.3f}, Add={score * renormalized_weight:.2f}")
        elif score is None: renormalized_weights[factor] = 0
    vol_score_val = intermediate_scores.get('volatility'); cvar_score_val = intermediate_scores.get('cvar')
    if vol_score_val is not None and cvar_score_val is not None and vol_score_val > 95 and cvar_score_val > 95: boost = 5.0; final_score += boost; final_score = min(100.0, final_score); logging.info(f"[{ticker_symbol}] Extreme Volatility + Tail Risk Boost (+{boost:.1f})")
    final_score = max(0.0, min(100.0, final_score)); logging.info(f"[{ticker_symbol}] --- Final Weighted Risk Score: {final_score:.2f} (Based on {total_valid_original_weight:.2f} original weight) ---"); return final_score, intermediate_scores, total_valid_original_weight
def get_risk_category(score):
    if score is None or pd.isna(score): return "N/A"
    try:
        score_float = float(score);
        if score_float >= 65: return "⚠️ High Risk"
        elif score_float >= 50: return "⚖️ Medium Risk"
        else: return "✅ Low Risk"
    except (ValueError, TypeError): logging.error(f"Could not convert risk score '{score}' to float."); return "N/A"

# --- Technical Strategy Functions REMOVED ---
# def strategy_golden_cross_rsi(...) REMOVED
# def strategy_macd_rsi_oversold(...) REMOVED
# def strategy_macd_bullish_adx(...) REMOVED
# def strategy_adx_heikin_ashi(...) REMOVED
# def strategy_psar_rsi(...) REMOVED

# --- Strategy Scanning Function REMOVED ---
# def scan_strategies(...) REMOVED

# [LLM Prompt Formatting functions remain the same]
def format_stock_data_for_prompt(data):
    if not data or not data.get('ticker'): return "No current Yahoo Finance summary data found for this ticker."
    ticker = data['ticker']; lines = [ f"Context - Summary data for {ticker} ({data.get('companyName', 'N/A')}) from Yahoo Finance:", f"- Current/Last Price: {format_val(data.get('priceForDisplay'), '$', prec=2)}", f"- Market Cap: {format_val(data.get('marketCap'), '$', prec=2)}", f"- P/E (Trailing): {format_val(data.get('trailingPE'), prec=2)}", f"- P/E (Forward): {format_val(data.get('forwardPE'), prec=2)}", f"- Dividend Yield: {format_val(data.get('dividendYield', 0) * 100 if data.get('dividendYield') is not None else 0, suffix='%', prec=2)}", f"- 50d SMA: {format_val(data.get('sma50'), '$', prec=2)}", f"- 200d SMA: {format_val(data.get('sma200'), '$', prec=2)}", f"- Beta: {format_val(data.get('beta'), prec=2)}", f"- Day Range: {format_val(data.get('dayLow'), '$', prec=2)} - {format_val(data.get('dayHigh'), '$', prec=2)}", f"- 52 Week Range: {format_val(data.get('fiftyTwoWeekLow'), '$', prec=2)} - {format_val(data.get('fiftyTwoWeekHigh'), '$', prec=2)}", f"- Volume: {format_val(data.get('volume'), prec=0)}", ]
    rec_key = data.get('recommendationKey'); num_opinions = data.get('numberOfAnalystOpinions'); mean_target = data.get('targetMeanPrice'); low_target = data.get('targetLowPrice'); high_target = data.get('targetHighPrice'); mapped_rec = map_recommendation_key_to_english(rec_key); analyst_line = "- Analyst Consensus (Aggregated Yahoo): Not Available"
    if mapped_rec not in ["Not Available", "N/A"] and mean_target is not None: analyst_count_str = f"{num_opinions} analysts" if num_opinions and isinstance(num_opinions, int) and num_opinions > 0 else "count unavailable"; target_mean_str = format_val(mean_target, '$', prec=2); target_low_str = format_val(low_target, '$', prec=2); target_high_str = format_val(high_target, '$', prec=2); range_str = f", ranging from {target_low_str} to {target_high_str}" if target_low_str != "Not Available" and target_high_str != "Not Available" else ""; analyst_line = f"- Analyst Consensus (Aggregated Yahoo): Based on {analyst_count_str}, recommendation: {mapped_rec}, avg target: {target_mean_str}{range_str}."
    elif mapped_rec not in ["Not Available", "N/A"]: analyst_count_str = f"{num_opinions} analysts" if num_opinions and isinstance(num_opinions, int) and num_opinions > 0 else "count unavailable"; analyst_line = f"- Analyst Consensus (Aggregated Yahoo): Based on {analyst_count_str}, recommendation: {mapped_rec} (target price N/A)."
    lines.append(analyst_line)
    if data.get('sector') and data['sector'] != 'N/A': lines.append(f"- Sector: {data['sector']}")
    if data.get('industry') and data['industry'] != 'N/A': lines.append(f"- Industry: {data['industry']}")
    filtered_lines = [ln for ln in lines if not (ln.strip().endswith(": Not Available") or ln.strip().endswith(": N/A") ) or ln.startswith("- Analyst Consensus")]; valid_data_points = sum(1 for ln in filtered_lines[1:] if "Not Available" not in ln and "N/A" not in ln and ln != analyst_line)
    if valid_data_points < 3 and not (data and data.get('priceForDisplay')): logging.warning(f"Very limited Yahoo data for {ticker}.");
    if not any("Price" in ln for ln in filtered_lines): return f"Could not retrieve key data (like price) from Yahoo for {ticker}."
    return "\n".join(filtered_lines)
def format_val(v, prefix="", suffix="", prec=2):
    if v is None or pd.isna(v) or str(v).lower() == 'n/a' or str(v).strip() == '': return "Not Available"
    try:
        v_float = float(v);
        if prec > 0 or v_float != int(v_float):
            if abs(v_float) >= 1e12: formatted_num = f"{v_float / 1e12:,.{prec}f}T"
            elif abs(v_float) >= 1e9: formatted_num = f"{v_float / 1e9:,.{prec}f}B"
            elif abs(v_float) >= 1e6: formatted_num = f"{v_float / 1e6:,.{prec}f}M"
            else: formatted_num = f"{v_float:,.{prec}f}"
        else: formatted_num = f"{int(v_float):,}"
        return f"{prefix}{formatted_num}{suffix}"
    except (ValueError, TypeError): return str(v).strip() if str(v).strip() else "Not Available"
def map_recommendation_key_to_english(key):
    mapping = { 'strong_buy': 'Strong Buy', 'buy': 'Buy', 'hold': 'Hold', 'sell': 'Sell', 'strong_sell': 'Strong Sell', 'underperform': 'Underperform', 'outperform': 'Outperform', 'none': 'N/A' }
    if key is None: return "Not Available"
    return mapping.get(str(key).lower(), str(key).capitalize() if key else "Not Available")

# [Wikipedia Index Lookup functions remain the same]
def build_sp500_ticker_map(cache_duration_hours=24, force_refresh=False):
    cache_file = "sp500_data.pkl"; ticker_map = None; loaded_from_cache = False
    logging.info(f"Checking S&P 500 cache (file: {cache_file}, force_refresh={force_refresh}).")
    if not force_refresh and os.path.exists(cache_file):
        try: cache_data = pd.read_pickle(cache_file); last_fetch_time = cache_data.get('timestamp', 0)
        except Exception as e: logging.warning(f"S&P 500 cache read error: {e}."); last_fetch_time = 0
        if (time.time() - last_fetch_time) / 3600 < cache_duration_hours: logging.info("Using cached S&P 500 data."); ticker_map = cache_data.get('ticker_map'); loaded_from_cache = bool(ticker_map)
        else: logging.info("S&P 500 cache expired.")
    if ticker_map is None:
        logging.info("Fetching fresh S&P 500 data."); url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); sp500_table = None
            try: html_content = io.StringIO(response.text); tables = pd.read_html(html_content, flavor='lxml'); sp500_table = tables[0]; logging.info("Using table 0 for S&P 500.")
            except Exception as e: logging.error(f"Error reading S&P 500 HTML: {e}."); return None
            ticker_col, name_col = None, None; possible_ticker_cols = ['Symbol', 'Ticker']; possible_name_cols = ['Security', 'Company', 'Name']
            for col in sp500_table.columns: col_str = str(col);
            if col_str in possible_ticker_cols and ticker_col is None: ticker_col = col_str
            if col_str in possible_name_cols and name_col is None: name_col = col_str
            if not ticker_col or not name_col: logging.error(f"Could not find S&P 500 cols."); return None
            scraped_ticker_map = {}
            for _, row in sp500_table.iterrows():
                 ticker_val, name_val = row.get(ticker_col), row.get(name_col)
                 if isinstance(ticker_val, str) and isinstance(name_val, str) and ticker_val.strip() and name_val.strip():
                    ticker_clean = ticker_val.strip().replace('.', '-'); name_lower = name_val.strip().lower(); name_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc)\.?\b|,', '', name_lower, flags=re.IGNORECASE).strip()
                    scraped_ticker_map[name_lower] = ticker_clean
                    if name_cleaned != name_lower and name_cleaned not in scraped_ticker_map: scraped_ticker_map[name_cleaned] = ticker_clean
            ticker_map = scraped_ticker_map; logging.info(f"Scraped {len(ticker_map)} S&P 500 entries.")
        except requests.exceptions.RequestException as e: logging.error(f"FATAL: Error fetching S&P 500 URL '{url}': {e}"); return None
        except Exception as e: logging.error(f"FATAL: Unexpected error during S&P 500 fetch: {e}", exc_info=True); return None
    if ticker_map is not None:
        overrides = { "google": "GOOGL", "alphabet": "GOOGL", "alphabet class c": "GOOG", "meta": "META", "facebook": "META", "amazon": "AMZN", "amazon.com": "AMZN", "berkshire hathaway": "BRK-B", "berkshire hathaway class b": "BRK-B", "3m": "MMM", "3m company": "MMM", "at&t": "T", "coca-cola": "KO", "the coca-cola company": "KO", "exxon mobil": "XOM", "exxonmobil": "XOM", "johnson & johnson": "JNJ", }
        ticker_map.update(overrides); logging.info(f"S&P 500 map updated, size: {len(ticker_map)}.")
        if not loaded_from_cache or force_refresh:
            try: pd.to_pickle({'timestamp': time.time(), 'ticker_map': ticker_map}, cache_file); logging.info(f"Saved S&P 500 map to cache.")
            except Exception as e: logging.warning(f"Warning: Could not write S&P 500 cache: {e}")
    else: logging.error("ERROR: S&P 500 Ticker map is None."); return None
    return ticker_map
def build_nasdaq100_ticker_map(cache_duration_hours=24, force_refresh=False):
    cache_file = "nasdaq100_data.pkl"; ticker_map = None; loaded_from_cache = False
    logging.info(f"Checking Nasdaq 100 cache (file: {cache_file}, force_refresh={force_refresh}).")
    if not force_refresh and os.path.exists(cache_file):
        try: cache_data = pd.read_pickle(cache_file); last_fetch_time = cache_data.get('timestamp', 0)
        except Exception as e: logging.warning(f"Nasdaq 100 cache read error: {e}."); last_fetch_time = 0
        if (time.time() - last_fetch_time) / 3600 < cache_duration_hours: logging.info("Using cached Nasdaq 100 data."); ticker_map = cache_data.get('ticker_map'); loaded_from_cache = bool(ticker_map)
        else: logging.info("Nasdaq 100 cache expired.")
    if ticker_map is None:
        logging.info("Fetching fresh Nasdaq 100 data."); url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); table_index = 4; nasdaq_table = None
            try:
                html_content = io.StringIO(response.text); tables = pd.read_html(html_content, flavor='lxml')
                if len(tables) <= table_index:
                    logging.warning(f"Expected Nasdaq 100 table index {table_index} not found. Auto-finding...")
                    found_table = False
                    for i, df in enumerate(tables): cols_lower = {str(col).lower() for col in df.columns}; has_ticker = any(t in cols_lower for t in ['ticker', 'symbol']); has_name = any(n in cols_lower for n in ['company', 'name', 'security'])
                    if has_ticker and has_name and len(df) > 50: nasdaq_table = df; logging.info(f"Found Nasdaq 100 table at index {i}."); found_table = True; break
                    if not found_table: raise IndexError("Could not find Nasdaq 100 table.")
                else: nasdaq_table = tables[table_index]; logging.info(f"Using table {table_index} for Nasdaq 100.")
            except Exception as e: logging.error(f"Error reading Nasdaq 100 HTML: {e}."); return None
            ticker_col, name_col = None, None; possible_ticker_cols = ['Ticker', 'Symbol']; possible_name_cols = ['Company', 'Name', 'Security']
            for col in nasdaq_table.columns: col_str = str(col)
            if col_str in possible_ticker_cols and ticker_col is None: ticker_col = col_str
            if col_str in possible_name_cols and name_col is None: name_col = col_str
            if not ticker_col or not name_col: logging.error(f"Could not find Nasdaq 100 columns."); return None
            scraped_ticker_map = {}
            for _, row in nasdaq_table.iterrows():
                 ticker_val, name_val = row.get(ticker_col), row.get(name_col)
                 if isinstance(ticker_val, str) and isinstance(name_val, str) and ticker_val.strip() and name_val.strip():
                    ticker_clean = ticker_val.strip().replace('.', '-'); name_lower = name_val.strip().lower(); name_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc)\.?\b|,', '', name_lower, flags=re.IGNORECASE).strip()
                    scraped_ticker_map[name_lower] = ticker_clean
                    if name_cleaned != name_lower and name_cleaned not in scraped_ticker_map: scraped_ticker_map[name_cleaned] = ticker_clean
            ticker_map = scraped_ticker_map; logging.info(f"Scraped {len(ticker_map)} Nasdaq 100 entries.")
        except requests.exceptions.RequestException as e: logging.error(f"FATAL: Error fetching Nasdaq 100 URL '{url}': {e}"); return None
        except Exception as e: logging.error(f"FATAL: Unexpected error during Nasdaq 100 fetch: {e}", exc_info=True); return None
    if ticker_map is not None:
        overrides = { "google": "GOOGL", "alphabet": "GOOGL", "alphabet class c": "GOOG", "alphabet inc.": "GOOGL", "meta": "META", "facebook": "META", "meta platforms": "META", "amazon": "AMZN", "amazon.com": "AMZN", "paypal": "PYPL", "paypal holdings": "PYPL", "netflix": "NFLX", "nvidia": "NVDA", "nvidia corporation": "NVDA", "moderna": "MRNA", "intel": "INTC", "intel corporation": "INTC", "cisco": "CSCO", "cisco systems": "CSCO", "adobe": "ADBE", "adobe inc.": "ADBE", "tesla": "TSLA", "tesla, inc.": "TSLA", "microsoft": "MSFT", "microsoft corporation": "MSFT", "apple": "AAPL", "apple inc.": "AAPL", }
        ticker_map.update(overrides); logging.info(f"Nasdaq 100 map updated, size: {len(ticker_map)}.")
        if not loaded_from_cache or force_refresh:
            try: pd.to_pickle({'timestamp': time.time(), 'ticker_map': ticker_map}, cache_file); logging.info(f"Saved Nasdaq 100 map to cache.")
            except Exception as e: logging.warning(f"Warning: Could not write Nasdaq 100 cache: {e}")
    else: logging.error("ERROR: Nasdaq 100 Ticker map is None."); return None
    return ticker_map
def get_ticker_from_combined_map(query, combined_map):
    if not combined_map: logging.warning("Combined ticker map unavailable."); return None
    query_lower = query.lower().strip();
    if not query_lower: return None
    ticker = combined_map.get(query_lower)
    if ticker: logging.debug(f"Combined map direct hit for '{query}': {ticker}"); return ticker
    query_cleaned = re.sub(r'\s+(inc|incorporated|corp|corporation|ltd|plc)\.?\b|,', '', query_lower, flags=re.IGNORECASE).strip()
    if query_lower == "coca cola": query_cleaned = "coca-cola"
    elif query_lower == "johnson and johnson": query_cleaned = "johnson & johnson"
    elif query_lower == "google": query_cleaned = "alphabet"
    if query_cleaned != query_lower:
         ticker = combined_map.get(query_cleaned)
         if ticker: logging.debug(f"Combined map cleaned hit for '{query}' -> '{query_cleaned}': {ticker}"); return ticker
    if query_lower.endswith(" inc"):
        ticker = combined_map.get(query_lower[:-4].strip())
        if ticker: logging.debug(f"Combined map 'inc' variation hit for '{query}': {ticker}"); return ticker
    logging.debug(f"Query '{query}' not found in combined map."); return None
@st.cache_resource(ttl=3600 * 12)
def load_combined_ticker_map():
    logging.info("\n" + "="*30 + " Building/Loading Combined Ticker Map " + "="*30); logging.info("--- Processing S&P 500 ---"); sp500_map = build_sp500_ticker_map();
    if sp500_map is None: logging.warning("Failed S&P 500 map."); sp500_map = {}
    logging.info("\n--- Processing Nasdaq 100 ---"); nasdaq100_map = build_nasdaq100_ticker_map()
    if nasdaq100_map is None: logging.warning("Failed Nasdaq 100 map."); nasdaq100_map = {}
    logging.info("\n--- Merging Maps ---"); combined_tickers = sp500_map.copy(); combined_tickers.update(nasdaq100_map); logging.info(f"Total entries in combined map: {len(combined_tickers)}"); logging.info(" Combined Ticker Map Ready " + "="*30 + "\n"); return combined_tickers
@st.cache_data(ttl=3600)
def lookup_ticker_by_company_name(query):
    if not query or len(query.strip()) < 1: return None
    search_term = query.strip(); logging.info(f"=== Starting Ticker Lookup for: '{search_term}' ==="); direct_ticker_attempt = search_term.upper().replace('$', '');
    is_potential_ticker = bool(re.fullmatch(r'[A-Z0-9\-\.]{1,10}', direct_ticker_attempt)) and not (direct_ticker_attempt.isdigit() and len(direct_ticker_attempt) > 4)
    if is_potential_ticker:
        logging.info(f"Step 1: Query '{search_term}' looks like ticker '{direct_ticker_attempt}'. Direct yf check...")
        try:
            ticker_obj = yf.Ticker(direct_ticker_attempt); info = ticker_obj.info
            if info and info.get('symbol', '').upper() == direct_ticker_attempt and info.get('quoteType') in ['EQUITY', 'ETF'] and info.get('regularMarketPrice') is not None: logging.info(f"Step 1 SUCCESS: Direct yf found {info.get('quoteType')}: {direct_ticker_attempt}"); return direct_ticker_attempt
            else:
                 hist = ticker_obj.history(period="1d")
                 if not hist.empty:
                     if info and info.get('quoteType') and info.get('quoteType') not in ['EQUITY', 'ETF']: logging.info(f"Step 1 FAILED: Ticker '{direct_ticker_attempt}' exists but not supported type (Type: {info.get('quoteType')}).")
                     else: logging.info(f"Step 1 SUCCESS (via history): Direct yf '{direct_ticker_attempt}' confirmed (likely EQUITY/ETF)."); return direct_ticker_attempt
                 else: logging.info(f"Step 1 FAILED: Direct yf check '{direct_ticker_attempt}' - no info/history.")
        except Exception as e: logging.warning(f"Step 1 EXCEPTION: Direct yf check failed: {e}.")
    else: logging.info(f"Step 1: Query '{search_term}' not formatted like ticker.")
    logging.info(f"Step 2: Checking Combined S&P/Nasdaq Map for '{search_term}'...")
    map_ticker = get_ticker_from_combined_map(search_term, COMBINED_TICKERS)
    if map_ticker: logging.info(f"Step 2 SUCCESS: Found '{search_term}' in Combined Map: {map_ticker}"); return map_ticker.upper()
    else: logging.info(f"Step 2 FAILED: Query '{search_term}' not in combined map.")
    logging.info(f"Step 3: Falling back to Yahoo Finance Search API for '{search_term}'...")
    try:
        matches = yf.utils.get_json("https://query1.finance.yahoo.com/v1/finance/search", params={"q": search_term}); quotes = matches.get("quotes", [])
        if not quotes: logging.info(f"Step 3 FAILED: No matches in Yahoo search."); return None
        best_match, highest_score = None, -1; search_term_lower = search_term.lower()
        for item in quotes:
            symbol = item.get("symbol"); quote_type = item.get("quoteType"); score = item.get("score", 0); short_name = item.get("shortname", "").lower(); long_name = item.get("longname", "").lower(); exch_disp = item.get("exchDisp", "")
            if (quote_type not in ["EQUITY", "ETF"] or not symbol or '^' in symbol or any(ft in quote_type for ft in ['FUTURE', 'INDEX', 'CURRENCY', 'OPTION', 'MUTUALFUND']) or item.get("typeDisp") in ['Index', 'Currency', 'Futures', 'Option', 'Mutual Fund']): logging.debug(f"Step 3 Filtered (Type/Symbol): {symbol} ({quote_type})"); continue
            is_allowed_exchange = ('.' not in symbol or symbol.endswith(('.TA', '.TL', '.AS', '.BR', '.DE', '.PA', '.L', '.TO', '.V', '.HE', '.SW')) or exch_disp in ["TLV", "TASE", "NMS", "NYQ", "ASE", "AMS", "BRU", "GER", "PAR", "LSE", "TOR", "VAN", "HEL", "EBS"])
            if '.' in symbol and not is_allowed_exchange: logging.debug(f"Step 3 Filtered (Exchange): {symbol}"); continue
            current_score = score;
            if search_term_lower == short_name: current_score += 1000
            elif search_term_lower == long_name: current_score += 500
            elif short_name and search_term_lower in short_name: current_score += (len(search_term_lower) / len(short_name)) * 100
            elif long_name and search_term_lower in long_name: current_score += (len(search_term_lower) / len(long_name)) * 50
            if search_term_lower.upper() == symbol.upper(): current_score += 200
            if item.get("isYahooFinance", False): current_score += 10
            logging.debug(f"Step 3 Kept: {symbol} ({quote_type}), Score: {current_score:.2f}")
            if current_score > highest_score: highest_score = current_score; best_match = symbol
        if best_match: logging.info(f"Step 3 SUCCESS: Best match from Fallback Search (Score: {highest_score:.2f}): {best_match}"); return best_match.upper()
        else: logging.info(f"Step 3 FAILED: No suitable EQUITY/ETF found via Fallback Search."); return None
    except Exception as e: logging.warning(f"Step 3 EXCEPTION: Fallback search failed: {e}", exc_info=False); return None
    finally: logging.info(f"=== Finished Ticker Lookup for: '{search_term}' ===")


# --- Load Combined Ticker Map ---
COMBINED_TICKERS = load_combined_ticker_map()

# --- Chat Management ---
if "messages" not in st.session_state: st.session_state.messages = []
if "predefined_question" not in st.session_state: st.session_state.predefined_question = None

# --- UI Elements ---
# Updated Title
st.markdown('<h1 style="text-align: left;">📈 Financial Chat, Risk Score, News & Forecasting</h1>', unsafe_allow_html=True)
# Updated description
st.markdown(f'<p style="text-align: left; font-size: small;">Ask about stocks ($AAPL, Microsoft), compare, or discuss finance. Includes Dynamic Risk Score, Recent News Sentiment (Multi-Source/{NEWS_DAYS_BACK}d/VADER), and ETS Price Forecasting (Calculated). No charts or technical scans.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.markdown("## Examples")
    # Updated MENU_OPTIONS - removed Scan Signals
    MENU_OPTIONS = {
        "🔍 Stock Info": ["What's up with $TSLA?", "Tell me about Apple", "Info on Coca-Cola?", "$ESLT.TA details", "Microsoft data?", "3M Company info?"],
        # "🤖 Scan Signals": ["Scan $SPY", "Entry signal on $MSFT?", "Scan $NVDA for signals"], # REMOVED
        "🆚 Compare": ["Compare $MSFT to $GOOGL", "Intel vs Nvidia?", "Compare 3M and General Electric"],
        "📊 TA Concepts": ["What is SMA?", "Explain Moving Averages?", "What is Support/Resistance?", "Candlesticks?", "What are technical indicators?"], # Kept TA concepts
        "⚖️ Risk": ["What's the risk score for $AMD?", "Explain the risk score model", "How risky is $TQQQ?", "Risk for GOOG?"],
        "📰 News Sentiment": ["News sentiment for $MSFT?", "Recent news sentiment for $NVDA?", "Sentiment analysis for META?"],
        "📈 ETS Forecast": ["Forecast $AAPL price", "What's the ETS forecast for $MSFT?", "Price projection for $GOOG?"],
        "💼 Portfolio": ["How to diversify?", "Risks of single stocks?"],
        "📰 Market/General": ["Impact of interest rates?", "Inflation effect?", "What are ETFs?"],
    }
    for category, questions in MENU_OPTIONS.items():
        is_expanded = (category in ["🔍 Stock Info", "⚖️ Risk", "📰 News Sentiment", "📈 ETS Forecast"]) # Adjusted expanded categories
        with st.expander(f"**{category}**", expanded=is_expanded):
            for i, q in enumerate(questions):
                safe_category = re.sub(r'\W+', '', category); button_key = f"menu_{safe_category}_{i}"
                if st.button(q, key=button_key, use_container_width=True): st.session_state.predefined_question = q; st.rerun()
    st.caption("Click question to ask."); st.divider(); st.info("Enter ticker ($GOOGL) or company name (Microsoft, 3M) for specific data."); st.divider()

    # Display API Key warnings in sidebar
    if not RISKFOLIO_AVAILABLE: st.warning("Riskfolio-Lib not found. Some risk factors/methods disabled.", icon="⚠️")
    if not NEWS_API_KEY or len(NEWS_API_KEY) < 20: st.warning("NewsAPI Key missing/invalid. News sentiment may be limited.", icon="⚠️")
    if not FMP_API_KEY or len(FMP_API_KEY) < 20: st.warning("FMP API Key missing/invalid. News sentiment may be limited.", icon="⚠️")


# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(str(msg["content"]), unsafe_allow_html=True)

# --- Ticker Extraction ---
# [extract_tickers function remains the same]
FORBIDDEN_TICKERS = {"TELL", "LOVE", "LIFE", "SOLO", "PLAY", "YOU", "REAL", "CASH", "WORK", "HOPE", "GOOD", "SAFE", "FAST", "COOK", "HUGE", "YOLO", "BOOM", "DUDE", "WISH", "ME", "ARE", "IS", "THE", "FOR", "AND", "NOW", "SEE", "CAN", "HAS", "WAS", "BUY", "SELL", "ALL", "ONE", "TWO", "BIG", "NEW", "OLD", "TOP", "LOW", "HIGH", "DATA"}
def extract_tickers(text):
    dollar_tickers = re.findall(r"(?<![a-zA-Z0-9])\$([A-Z0-9\-\.]{1,10})\b", text, re.IGNORECASE)
    plain_tickers = re.findall(r"\b([A-Z]{1,4}[A-Z0-9]{0,6})\b", text) # Keep slightly more restrictive regex
    raw = dollar_tickers + [t for t in plain_tickers if '$'+t.upper() not in ['$' + dt.upper() for dt in dollar_tickers]]
    clean = [t.upper() for t in raw if t and len(t) <= 10 and not t.isdigit() and t.upper() not in FORBIDDEN_TICKERS]
    seen = set(); unique_clean = [t for t in clean if not (t in seen or seen.add(t))]
    unique_clean = [t for t in unique_clean if len(t) > 1 or t in ["T"]] # Keep single-letter tickers like T
    logging.info(f"Extracted potential tickers: {unique_clean} from text: '{text}'"); return unique_clean


# --- User Input Handling ---
user_input_triggered = None
if st.session_state.predefined_question: user_input_triggered = st.session_state.predefined_question; st.session_state.predefined_question = None; logging.info(f"Processing predefined: '{user_input_triggered}'")
# Updated input hint
chat_input_value = st.chat_input(f"Ask about stocks ($AAPL, Microsoft), risk, news ({NEWS_DAYS_BACK}d), forecast, or finance...")
if chat_input_value: user_input_triggered = chat_input_value.strip(); logging.info(f"Processing user input: '{user_input_triggered}'")


# --- Main Processing Logic ---
if user_input_triggered:
    user_input = user_input_triggered
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty(); placeholder.markdown("⏳ Thinking...")

        # --- Ticker Identification & Lookup ---
        extracted_tickers = extract_tickers(user_input); lookup_query = None
        if extracted_tickers:
             lookup_query = extracted_tickers[0]; logging.info(f"Prioritizing ticker '{lookup_query}'.")
        else:
            # Command prefixes updated - removed 'scan'
            command_prefixes = ("compare ", "risk ", "risky ", "risk score ", "what is the risk score for ", "news sentiment for ", "recent news sentiment for ", "sentiment analysis for ", "forecast ", "price forecast ", "ets forecast ", "price projection for ")
            question_prefixes = ("what", "how", "explain", "who", "why", "list", "define", "is ", "are ", "tell me about ")
            input_lower = user_input.lower(); is_command = any(input_lower.startswith(p) for p in command_prefixes); is_question = any(input_lower.startswith(p) for p in question_prefixes)
            potential_name = user_input;
            if is_command:
                 for prefix in command_prefixes:
                     if input_lower.startswith(prefix): potential_name = user_input[len(prefix):].strip(); break
            elif is_question and len(input_lower.split()) > 4: potential_name = None; logging.info("General question, skipping name lookup.")
            finance_terms = ["stock", "share", "company", "ticker", "market", "index", "etf", "bond", "risk", "volatility", "prediction", "news", "sentiment", "scan", "compare", "forecast", "projection"]
            if potential_name and len(potential_name) > 2 and potential_name.lower() != 'me' and potential_name.lower() not in finance_terms:
                lookup_query = potential_name; logging.info(f"Trying input '{lookup_query}' for name lookup.")
            else: logging.info("No ticker/plausible name found. Skipping lookup.")

        # --- Validate the lookup query ---
        # [Ticker validation logic remains the same]
        validated_ticker = None; primary_ticker = None
        if lookup_query:
            placeholder.markdown(f"⏳ Verifying '{lookup_query}'...");
            try: validated_ticker = lookup_ticker_by_company_name(lookup_query)
            except Exception as lookup_err: logging.error(f"Ticker lookup failed: {lookup_err}", exc_info=True); validated_ticker = None
            if validated_ticker:
                primary_ticker = validated_ticker; logging.info(f"Lookup success. Using primary ticker: {primary_ticker}")
                is_query_like_ticker = re.fullmatch(r'[\$\.A-Z0-9\-]{1,10}', lookup_query.upper().replace('$',''))
                if not is_query_like_ticker and lookup_query.upper() != primary_ticker:
                     if user_input.strip().upper().replace('$','') != primary_ticker:
                         st.toast(f"Found data for **{primary_ticker}** (based on '{lookup_query}')", icon="💡")
            else:
                logging.info(f"Could not validate/find equity/ETF ticker for '{lookup_query}'.")
                if len(lookup_query) > 1:
                     if is_potential_ticker or len(lookup_query.split()) > 1 or len(lookup_query) > 4: # Adjusted check
                         st.toast(f"Couldn't find stock/ETF matching '{lookup_query}'.", icon="⚠️")
                primary_ticker = None
        else: logging.info("No lookup query."); primary_ticker = None


        # --- Initialize Context Variables ---
        stock_data = None;
        close_prices_history = None # Store close prices for ETS
        full_history_df = None # Store full df for risk
        # Removed: strategy_scan_summary, scan_has_signals
        news_sentiment_summary = "News Sentiment: Not calculated."
        news_sentiment_counts_dict = {'positive': None, 'negative': None, 'neutral': None, 'total': None, 'avg_score': None}
        ets_forecast_summary = "ETS Price Forecast: Not calculated."
        stock_data_for_prompt = "No specific ticker identified or data failed."
        risk_score_final = None; risk_score_description = "Risk Score (Model): Not Calculated"

        # --- Fetch Data, Calculate Risk, News & Forecast if ticker identified ---
        if primary_ticker:
            with st.spinner(f"Gathering data & insights for **{primary_ticker}**..."):
                logging.info(f"--- Processing Data for Ticker: {primary_ticker} ---")

                # 1. Fetch Yahoo Summary Data
                placeholder.markdown(f"⏳ Fetching Yahoo Finance summary for **{primary_ticker}**...")
                stock_data = get_stock_data(primary_ticker)
                if stock_data: stock_data_for_prompt = format_stock_data_for_prompt(stock_data); logging.info("Yahoo summary fetch OK.")
                else: stock_data_for_prompt = f"Could not retrieve summary data from Yahoo for {primary_ticker}."; logging.warning("Yahoo summary fetch failed.")

                # 2. Fetch Unified History (Needed for Risk & Forecast)
                placeholder.markdown(f"⏳ Fetching Yahoo Finance history for **{primary_ticker}**...")
                close_prices_history, full_history_df = get_unified_yfinance_history(primary_ticker, period="3y")

                # 3. Calculate Risk Score (Uses full history df)
                placeholder.markdown(f"⏳ Calculating Dynamic Risk Score for **{primary_ticker}**...")
                intermediate_risk_scores = {}; factors_weight_sum = 0.0
                if full_history_df is not None and not full_history_df.empty:
                    try: risk_score_final, intermediate_risk_scores, factors_weight_sum = calculate_dynamic_risk_score(primary_ticker, full_history_df, weights=DEFAULT_WEIGHTS, cov_method='hist')
                    except Exception as risk_err: logging.error(f"Error calling risk calc: {risk_err}", exc_info=True); risk_score_final = None
                if risk_score_final is not None: risk_category = get_risk_category(risk_score_final); risk_score_description = f"Risk Score (Model): {risk_score_final:.2f}/100 ({risk_category})"; logging.info(f"Risk score OK: {risk_score_description}.")
                else: risk_score_description = f"Risk Score (Model): Could not calculate for {primary_ticker} (Insufficient history or data error)."; logging.warning("Risk score calc returned None.")

                # 4. Run Strategy Scan REMOVED
                # placeholder.markdown(f"⏳ Running Technical Strategy Scan for **{primary_ticker}**...") REMOVED

                # 5. Fetch Multi-Source News Sentiment
                placeholder.markdown(f"⏳ Fetching Recent News Sentiment (Multi-Source/VADER) for **{primary_ticker}**...")
                try:
                    news_sentiment_summary, news_sentiment_counts_dict = get_multi_source_news_sentiment(primary_ticker, NEWS_API_KEY, FMP_API_KEY)
                    logging.info(f"Multi-source news sentiment fetch completed for {primary_ticker}.")
                except Exception as news_err:
                    logging.error(f"Error calling multi-source news sentiment: {news_err}", exc_info=True)
                    news_sentiment_summary = f"News Sentiment: Error during analysis for {primary_ticker}."
                    news_sentiment_counts_dict = {'positive': None, 'negative': None, 'neutral': None, 'total': None, 'avg_score': None}

                # 6. Generate ETS Forecast (Uses close_prices series)
                placeholder.markdown(f"⏳ Generating ETS Price Forecast for **{primary_ticker}**...")
                forecast_days = 7 # Define forecast horizon
                if close_prices_history is not None and not close_prices_history.empty:
                    try:
                        forecast_values, eval_metric_str, model_desc = forecast_stock_ets_advanced(primary_ticker, close_prices_history, forecast_days=forecast_days)
                        if forecast_values is not None:
                            forecast_lines = [f"- Forecast Period: Next {forecast_days} business days"]
                            forecast_lines.append(f"- Model Used: {model_desc}")
                            if eval_metric_str and "Error" not in eval_metric_str: forecast_lines.append(f"- {eval_metric_str}")
                            forecast_lines.append("- Forecasted Prices:")
                            for date, value in forecast_values.items(): date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date); forecast_lines.append(f"  - {date_str}: {value:.2f}")
                            ets_forecast_summary = "\n".join(forecast_lines); logging.info(f"ETS Forecast generated successfully for {primary_ticker}.")
                        else: ets_forecast_summary = f"ETS Price Forecast ({model_desc}): Could not generate forecast for {primary_ticker}. Reason: {eval_metric_str or 'Model fitting error'}"; logging.warning(f"ETS forecast generation failed for {primary_ticker}. Reason: {eval_metric_str or 'Model fitting error'}")
                    except Exception as forecast_err: logging.error(f"Error calling ETS forecast function: {forecast_err}", exc_info=True); ets_forecast_summary = f"ETS Price Forecast: Error during calculation for {primary_ticker}."
                else: ets_forecast_summary = f"ETS Price Forecast: Unavailable due to missing historical price data for {primary_ticker}."; logging.warning(f"ETS forecast skipped for {primary_ticker} due to missing history.")

            placeholder.markdown(f"⏳ Compiling info & generating response for **{primary_ticker}**...")
        else:
            logging.info("No ticker identified. Skipping data steps."); placeholder.markdown("⏳ Generating general response...")
            context_message_content = "\n--- Start of Context ---\nNo specific ticker could be identified from the user's request, or data fetching failed. Please provide a general response or indicate the ticker was not found/data is unavailable.\n--- End of Context---\n"
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": context_message_content}]
            history_limit = 6; start_index = max(0, len(st.session_state.messages) - (history_limit * 2)); relevant_history = st.session_state.messages[start_index:]
            full_messages.extend(relevant_history);
            if not full_messages or full_messages[-1]["role"] != "user": full_messages.append({"role": "user", "content": user_input})


        # --- Prepare messages for OpenAI (if ticker identified, context already built) ---
        if primary_ticker:
             full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
             history_limit = 6; start_index = max(0, len(st.session_state.messages) - (history_limit * 2)); relevant_history = st.session_state.messages[start_index:]
             full_messages.extend(relevant_history); logging.info(f"Included {len(relevant_history)} messages from history.")

             # --- Add the Specific Context Block (REMOVED Strategy Scan) ---
             context_message_content = ""
             context_message_content += f"\n--- Start of Context for {primary_ticker} ---\n"
             context_message_content += f"Yahoo Finance Summary:\n{stock_data_for_prompt}\n\n"
             # Removed: context_message_content += f"Technical Strategy Scan Summary (Yahoo Finance Daily Data):\n{strategy_scan_summary}\n\n"
             context_message_content += f"Recent News Sentiment Summary (Multi-Source / VADER Analysis):\n{news_sentiment_summary}\n\n"
             context_message_content += f"Dynamic Risk Score Calculation Result:\n{risk_score_description}\n\n"
             context_message_content += f"ETS Price Forecast (Calculated from Yahoo History):\n{ets_forecast_summary}\n" # Kept Forecast
             context_message_content += f"--- End of Context for {primary_ticker} ---\n"
             context_message_content += f"Please prioritize info in context. Follow guidelines (missing data, analyst format, state sources: Yahoo Finance, Multi-Source News/VADER, ETS Model, etc.). Note news covers last {NEWS_DAYS_BACK} days. ETS Forecast is short-term. Technical strategy signals are NOT available." # Added note about no signals

             full_messages.insert(-1, {"role": "system", "content": context_message_content})
             logging.info(f"Prepared context for {primary_ticker}."); logging.debug(f"Context (start): {context_message_content[:800]}...")

             if not full_messages or full_messages[-1]["role"] != "user":
                 full_messages.append({"role": "user", "content": user_input})
                 logging.warning("Appended user message manually.")


        # --- Call OpenAI API ---
        try:
            logging.info(f"Sending request to {MODEL_NAME}.");
            if not client: raise ValueError("OpenAI client error.")
            stream = client.chat.completions.create(model=MODEL_NAME, messages=full_messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stream=True )
            response_content = placeholder.write_stream(stream); gpt_reply = response_content
            st.session_state.messages.append({"role": "assistant", "content": gpt_reply}); logging.info(f"Streamed response OK.")

            # Display Data Dashboard AFTER response generation
            if primary_ticker and stock_data:
                 risk_category_display = get_risk_category(risk_score_final)
                 display_stock_data_dashboard(
                     stock_data,
                     risk_score_final,
                     risk_category_display,
                     news_sentiment_counts_dict
                 )
            elif primary_ticker:
                 st.warning(f"Could not display data snapshot for {primary_ticker} (Yahoo summary data missing).")

        except AuthenticationError as e:
            error_message = f"😥 OpenAI API Authentication Error: Invalid API Key or configuration issue. Please check your key. Error: {e}";
            placeholder.error(error_message); logging.error(f"OpenAI API Authentication Error: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": error_message}); st.stop()
        except OpenAIError as e:
            error_message = f"😥 OpenAI API Error: {e}. Try again later."; placeholder.error(error_message); logging.error(f"OpenAI API Error: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": error_message})
        except Exception as e: error_message = f"😥 Unexpected error: {e}. Please check logs."; placeholder.error(error_message); logging.error(f"Unexpected Error: {e}", exc_info=True); st.session_state.messages.append({"role": "assistant", "content": f"Internal Error: {e}"})


# --- Footer ---
st.divider()
# Updated footer disclaimer
st.caption(f"*Data: Yahoo Finance (via yfinance), News (NewsAPI, FMP, RSS - {NEWS_DAYS_BACK}d / VADER), Wikipedia, Model Calculations, ETS Forecast. May be delayed. Not financial advice.*")
