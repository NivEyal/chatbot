# chatbot
# 📈 Financial Chatbot with Risk, News & ETS Forecasting

This Streamlit application provides a conversational interface for financial information, incorporating:

*   **Stock Data:** Fetches real-time summary data from Yahoo Finance.
*   **Company Lookup:** Identify tickers from company names (using Wikipedia lists and Yahoo Search).
*   **Risk Scoring:** Calculates a dynamic risk score based on multiple weighted factors (volatility, valuation, liquidity, beta, Piotroski F-score, downside risk metrics via Riskfolio-Lib if installed).
*   **Technical Strategy Signals:** Scans historical data (Yahoo Finance) for signals from common technical strategies (Golden Cross, MACD crossovers, etc.).
*   **News Sentiment Analysis:** Aggregates recent news from NewsAPI, Financial Modeling Prep (FMP), and RSS feeds, performing VADER sentiment analysis.
*   **ETS Price Forecasting:** Generates a short-term (e.g., 7-day) price forecast using an Exponential Smoothing (ETS) model based on historical price data.

**Disclaimer:** This tool is for educational and informational purposes only. It does **not** provide financial advice. Data may be delayed or contain errors. Always conduct your own thorough research or consult a qualified financial advisor before making investment decisions. Statistical forecasts (like ETS) are based on past data and are not guarantees of future performance.

## Features

*   **Conversational Interface:** Ask questions about stocks, markets, technical concepts, risk, news sentiment, or forecasts in natural language.
*   **Ticker/Company Recognition:** Use ticker symbols (e.g., `$AAPL`) or company names (e.g., `Microsoft`).
*   **Data Dashboard:** Displays key metrics, analyst consensus, and news sentiment counts for identified tickers.
*   **Multi-Source News:** Combines news from various APIs and RSS feeds for broader coverage.
*   **VADER Sentiment:** Provides a quick gauge of the tone of recent news coverage.
*   **Configurable Risk Model:** Uses weighted factors (customizable in the code) for risk assessment.
*   **Technical Scan:** Identifies potential entry/exit signals based on pre-defined strategies.
*   **ETS Forecasting:** Provides a statistical short-term price projection.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd financial-chatbot
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `riskfolio-lib` is optional for the advanced risk factors (SemiDeviation, CDaR, GMD). The chatbot will fall back to simpler calculations if it's not installed.*

4.  **Configure API Keys:**

    🚨 **SECURITY WARNING:** The current `chatbot_optimized.py` script hardcodes API keys directly. This is **highly insecure** and not recommended for any environment, especially if sharing the code.

    **Strongly Recommended Method: Streamlit Secrets (`secrets.toml`)**

    *   Create a folder named `.streamlit` in the project root.
    *   Inside `.streamlit`, create a file named `secrets.toml`.
    *   Add your keys to `secrets.toml` like this:
        ```toml
        # .streamlit/secrets.toml
        OPENAI_API_KEY = "sk-..."
        NEWS_API_KEY = "your_newsapi_key_here"
        FMP_API_KEY = "your_fmp_api_key_here"
        ```
    *   **Modify `chatbot_optimized.py`** to load keys from secrets:
        *   Remove the hardcoded key assignments near the top.
        *   Load keys using `st.secrets`:
            ```python
            # --- API Key Loading (Using Streamlit Secrets - RECOMMENDED) ---
            try:
                OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
                NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
                FMP_API_KEY = st.secrets["FMP_API_KEY"]
                logging.info("Loaded API keys from Streamlit secrets.")
            except KeyError as e:
                st.error(f"Missing API Key in Streamlit secrets: {e}. Please check your .streamlit/secrets.toml file.")
                logging.error(f"Missing API Key in Streamlit secrets: {e}")
                st.stop()
            except FileNotFoundError:
                 st.error("Streamlit secrets file (.streamlit/secrets.toml) not found. Please create it with your API keys.")
                 logging.error("Streamlit secrets file not found.")
                 st.stop()

            # --- API Key Validation (Adjust for loaded keys) ---
            missing_keys = []
            if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"): missing_keys.append("OpenAI")
            if not NEWS_API_KEY or len(NEWS_API_KEY) < 20: missing_keys.append("NewsAPI")
            if not FMP_API_KEY or len(FMP_API_KEY) < 20: missing_keys.append("FMP")

            if missing_keys:
                st.error(f"Invalid API Key(s) format found in secrets: {', '.join(missing_keys)}")
                logging.error(f"Invalid API Key(s) format in secrets: {', '.join(missing_keys)}")
                st.stop()
            else:
                logging.info("API keys loaded from secrets appear valid.")
            # --- End of API Key Loading ---

            # Remove the old hardcoded key assignments and validation logic related to placeholders.
            # The rest of the script uses these variables (e.g., OPENAI_API_KEY).
            ```
        *   Make sure `.streamlit/secrets.toml` is listed in your `.gitignore` file!

    **Alternative (Less Secure): Environment Variables**

    *   Set environment variables before running the script:
        ```bash
        export OPENAI_API_KEY="sk-..."
        export NEWS_API_KEY="your_newsapi_key_here"
        export FMP_API_KEY="your_fmp_api_key_here"
        # On Windows use 'set' instead of 'export'
        ```
    *   Modify the script to load from environment variables using `os.getenv()`.

    **(Not Recommended) Hardcoding (Current State):** If you *must* keep keys hardcoded for testing, ensure you replace the placeholders in `chatbot_optimized.py` with your actual keys. **Do not commit real keys to a public GitHub repository.**

    **Required Keys:**
    *   **OpenAI:** For the LLM interaction.
    *   **NewsAPI:** For fetching news articles (newsapi.org).
    *   **Financial Modeling Prep (FMP):** For fetching news articles (financialmodelingprep.com). Free tier has limitations.

## Running the App

1.  Activate your virtual environment (if created).
2.  Ensure API keys are configured (preferably via `secrets.toml`).
3.  Run the Streamlit app from your terminal:
    ```bash
    streamlit run chatbot_optimized.py
    ```
4.  The application should open in your web browser.

## Dependencies

All required Python packages are listed in `requirements.txt`.

## License

(Optional) Specify your chosen license here (e.g., This project is licensed under the MIT License - see the LICENSE file for details).
