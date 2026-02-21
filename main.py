# ============================================================================
# IMPORTS & SETUP
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# DATA LOADING & PROCESSING (WITH CACHING FOR PERFORMANCE)
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data():
    indian_tickers = [
        "RELIANCE.NS",  # Reliance Industries
        "HDFCBANK.NS",  # HDFC Bank
        "BHARTIARTL.NS",# Bharti Airtel
        "TCS.NS",       # Tata Consultancy Services
        "ICICIBANK.NS", # ICICI Bank
        "SBIN.NS",      # State Bank of India
        "INFY.NS",      # Infosys
        "BAJFINANCE.NS",# Bajaj Finance
        "LICI.NS",      # LIC India
        "HINDUNILVR.NS",# Hindustan Unilever
        "LT.NS",        # Larsen & Toubro
        "HCLTECH.NS",   # HCL Technologies
        "KOTAKBANK.NS", # Kotak Mahindra Bank
        "AXISBANK.NS",  # Axis Bank
        "MARUTI.NS",    # Maruti Suzuki
        "ASIANPAINT.NS",# Asian Paints
        "TITAN.NS",     # Titan Company
        "SUNPHARMA.NS", # Sun Pharma
        "BAJAJFINSV.NS",# Bajaj Finserv
        "NTPC.NS"       # NTPC
    ]
    
    global_tickers = [
        "NVDA", # Nvidia
        "AAPL", # Apple
        "MSFT", # Microsoft
        "GOOGL",# Alphabet (Google)
        "AMZN", # Amazon
        "META", # Meta Platforms (Facebook)
        "AVGO", # Broadcom
        "TSLA", # Tesla
        "TSM",  # Taiwan Semiconductor (TSM)
        "2222.SR", # Saudi Aramco (Saudi Exchange)
        "JPM",  # JP Morgan Chase
        "WMT",  # Walmart
        "V",    # Visa
        "JNJ",  # Johnson & Johnson
        "UNH",  # UnitedHealth Group
        "PG",   # Procter & Gamble
        "BAC",  # Bank of America
        "XOM",  # Exxon Mobil
        "CVX",  # Chevron
        "NFLX", # Netflix
        "ORCL", # Oracle
        "PEP",  # PepsiCo
        "KO",   # Coca-Cola
        "IBM",  # IBM
        "SAP",  # SAP (Germany)
        "BABA", # Alibaba (China, USA ADR)
        "TSM",  # TSMC
        "NKE",  # Nike
        "DIS"   # Disney
    ]
    
    df1 = yf.download(indian_tickers, start="2019-01-01", end="2026-01-01", progress=False)
    df2 = yf.download(global_tickers, start="2019-01-01", end="2026-01-01", progress=False)
    
    close_india = df1["Close"]
    close_global = df2["Close"]
    close_price = pd.concat([close_india, close_global], axis = 1)
    close_price = close_price.sort_index()
    close_price = close_price.ffill()
    close_price = close_price.dropna()
    
    return close_price, indian_tickers, global_tickers

# Load data with caching - this will only download once and then use cached data
close_price, indian_tickers, global_tickers = load_stock_data()

# Calculate returns
returns_df = close_price.pct_change()
log_returns = np.log(close_price / close_price.shift(1)).dropna()

annual_return = returns_df.mean()*252
annual_volatility = returns_df.std()*np.sqrt(252)
sharpe_ratio = annual_return/annual_volatility

risk_metrics = pd.DataFrame({
    "Annual Return": annual_return,
    "Annual Volatility": annual_volatility,
    "Sharpe Ratio": sharpe_ratio
})
top_risk_metrics = risk_metrics.sort_values("Annual Return", ascending = False)

# All tickers combined
all_tickers = list(indian_tickers) + list(global_tickers)

# ============================================================================
# COMPANY NAME MAPPINGS
# ============================================================================
company_names = {
    # Indian Stocks (NSE)
    "RELIANCE.NS": "Reliance Industries Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "BHARTIARTL.NS": "Bharti Airtel Ltd",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "ICICIBANK.NS": "ICICI Bank Ltd",
    "SBIN.NS": "State Bank of India",
    "INFY.NS": "Infosys Ltd",
    "BAJFINANCE.NS": "Bajaj Finance Ltd",
    "LICI.NS": "Life Insurance Corporation of India",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd",
    "LT.NS": "Larsen & Toubro Ltd",
    "HCLTECH.NS": "HCL Technologies Ltd",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd",
    "AXISBANK.NS": "Axis Bank Ltd",
    "MARUTI.NS": "Maruti Suzuki India Ltd",
    "ASIANPAINT.NS": "Asian Paints Ltd",
    "TITAN.NS": "Titan Company Ltd",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd",
    "BAJAJFINSV.NS": "Bajaj Finserv Ltd",
    "NTPC.NS": "NTPC Ltd",
    # Global Stocks
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc (Google)",
    "AMZN": "Amazon.com Inc",
    "META": "Meta Platforms Inc",
    "AVGO": "Broadcom Inc",
    "TSLA": "Tesla Inc",
    "TSM": "Taiwan Semiconductor Manufacturing Co",
    "2222.SR": "Saudi Arabian Oil Company (Aramco)",
    "JPM": "JPMorgan Chase & Co",
    "WMT": "Walmart Inc",
    "V": "Visa Inc",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc",
    "PG": "Procter & Gamble Co",
    "BAC": "Bank of America Corp",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "NFLX": "Netflix Inc",
    "ORCL": "Oracle Corporation",
    "PEP": "PepsiCo Inc",
    "KO": "The Coca-Cola Company",
    "IBM": "International Business Machines (IBM)",
    "SAP": "SAP SE",
    "BABA": "Alibaba Group Holding Ltd",
    "NKE": "Nike Inc",
    "DIS": "The Walt Disney Company"
}

# ============================================================================
# CURRENCY DETECTION FUNCTION
# ============================================================================
def get_currency_info(ticker):
    """
    Returns currency symbol and code based on ticker
    Indian stocks (.NS suffix) -> INR (₹)
    Saudi stocks (.SR suffix) -> SAR (﷼)
    Global stocks -> USD ($)
    """
    if ticker.endswith('.NS'):
        return "₹", "INR"
    elif ticker.endswith('.SR'):
        return "﷼", "SAR"
    else:
        return "$", "USD"

def format_price(price, ticker):
    """Format price with correct currency symbol"""
    currency_symbol, _ = get_currency_info(ticker)
    return f"{currency_symbol}{price:,.2f}"

# Remove old class definitions and use Streamlit directly

# ============================================================================
# PAGE STYLING & CONFIGURATION
# ============================================================================
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Advanced Quant Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ADVANCED CUSTOM CSS & THEMING
# ============================================================================
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary: #00D4FF;
        --secondary: #FF006E;
        --accent: #8338EC;
        --success: #06FFA5;
        --warning: #FFBE0B;
        --danger: #FB5607;
        --dark-bg: #0A0E27;
        --card-bg: #1B1F3A;
        --text-primary: #E8EAED;
        --text-secondary: #A0A3B8;
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0A0E27 0%, #1B1F3A 100%);
        color: #E8EAED;
    }
    
    /* Main Container */
    .main {
        padding: 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1624 0%, #1B1F3A 100%);
        border-right: 2px solid #00D4FF;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00D4FF !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Modern Cards */
    .metric-card {
        background: linear-gradient(135deg, #1B1F3A 0%, #2D2150 100%);
        border: 2px solid #00D4FF;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 0 50px rgba(0, 212, 255, 0.4);
        border-color: #FF006E;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00D4FF 0%, #8338EC 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #A0A3B8;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4FF 0%, #8338EC 100%);
        color: white !important;
        border-radius: 10px;
    }
    
    /* Selectbox & Input */
    .stSelectbox>div>div, .stTextInput>div>div {
        background: #1B1F3A !important;
        border: 2px solid #00D4FF !important;
        border-radius: 12px !important;
        color: #E8EAED !important;
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        accent-color: #00D4FF;
    }
    
    /* Info & Warning Boxes */
    .info-box, .stInfo {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(131, 56, 236, 0.1) 100%) !important;
        border: 2px solid #00D4FF !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    /* Success Box */
    .success-box, .stSuccess {
        background: linear-gradient(135deg, rgba(6, 255, 165, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
        border: 2px solid #06FFA5 !important;
        border-radius: 15px !important;
    }
    
    /* Custom Divider */
    .gradient-divider {
        background: linear-gradient(90deg, #00D4FF, #8338EC, #FF006E);
        height: 3px;
        border: none;
        margin: 25px 0;
        border-radius: 2px;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: #1B1F3A !important;
        border-radius: 15px !important;
        overflow: hidden;
    }
    
    /* Animation */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.6); }
    }
    
    .glow-text {
        animation: glow 2s ease-in-out infinite;
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 25px 0; border-bottom: 2px solid #00D4FF;">
    <h1 style="color: #00D4FF; margin: 0; font-size: 2em;">📈 QUANT</h1>
    <p style="color: #A0A3B8; margin: 5px 0 0 0; font-size: 0.9em;">Advanced Trading System v2.0</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Risk Metrics", "🎲 Monte Carlo", "ℹ️ System Info"],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="padding: 20px; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(131, 56, 236, 0.1) 100%); border-radius: 15px; border: 2px solid #00D4FF;">
    <p style="color: #06FFA5; font-size: 0.85rem; margin: 8px 0;">
        <strong>📊 Total Stocks:</strong> {len(close_price.columns)}<br>
        <strong>📅 Data Range:</strong> 2019-2026<br>
        <strong>💹 Indian Stocks:</strong> 20<br>
        <strong>🌍 Global Stocks:</strong> {len(close_price.columns) - 20}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if menu == "🏠 Dashboard":
    st.markdown("<h1 style='text-align: center; color: #00D4FF;'>📈 Stock Market Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #A0A3B8; margin: 0; font-size: 0.9rem;'>Total Stocks</p>
            <h2 style='color: #00D4FF; margin: 10px 0; font-size: 2.5rem;'>{len(close_price.columns)}</h2>
            <p style='color: #06FFA5; margin: 5px 0 0 0; font-size: 0.85rem;'>✓ Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #A0A3B8; margin: 0; font-size: 0.9rem;'>Avg Annual Return</p>
            <h2 style='color: #06FFA5; margin: 10px 0; font-size: 2.5rem;'>{risk_metrics['Annual Return'].mean():.2%}</h2>
            <p style='color: #00D4FF; margin: 5px 0 0 0; font-size: 0.85rem;'>Portfolio Wide</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #A0A3B8; margin: 0; font-size: 0.9rem;'>Avg Volatility</p>
            <h2 style='color: #FF006E; margin: 10px 0; font-size: 2.5rem;'>{risk_metrics['Annual Volatility'].mean():.2%}</h2>
            <p style='color: #FFBE0B; margin: 5px 0 0 0; font-size: 0.85rem;'>Risk Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #A0A3B8; margin: 0; font-size: 0.9rem;'>Avg Sharpe Ratio</p>
            <h2 style='color: #8338EC; margin: 10px 0; font-size: 2.5rem;'>{risk_metrics['Sharpe Ratio'].mean():.2f}</h2>
            <p style='color: #00D4FF; margin: 5px 0 0 0; font-size: 0.85rem;'>Risk-Adjusted</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Performers")
        top_10 = top_risk_metrics["Annual Return"].head(10)
        fig = px.bar(
            x=top_10.values,
            y=top_10.index,
            orientation='h',
            color=top_10.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Annual Return', 'y': 'Ticker'},
            title='Best Performing Stocks'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(27, 31, 58, 0.5)',
            font_color='#E8EAED',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Volatility Analysis")
        vol_data = risk_metrics['Annual Volatility'].nlargest(10)
        fig = px.bar(
            x=vol_data.values,
            y=vol_data.index,
            orientation='h',
            color=vol_data.values,
            color_continuous_scale='Reds',
            labels={'x': 'Annual Volatility', 'y': 'Ticker'},
            title='Highest Risk (Volatility) Stocks'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(27, 31, 58, 0.5)',
            font_color='#E8EAED',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

elif menu == "📊 Risk Metrics":
    st.markdown("<h1 style='text-align: center; color: #00D4FF;'>📊 Risk Metrics Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Top Returns", "📉 Low Returns", "⚖️ Sharpe Ratio", "📊 Volatility", "⛑️ Drawdown"])
    
    with tab1:
        n = st.slider("Number of Stocks", 1, 30, 10, key="top_returns")
        top_data = top_risk_metrics["Annual Return"].head(n)
        fig = px.bar(x=top_data.values, y=top_data.index, orientation='h', color=top_data.values,
                     color_continuous_scale='Greens', title='✨ Top Performers by Annual Return')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(27, 31, 58, 0.5)',
                         font_color='#E8EAED', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        n = st.slider("Number of Stocks", 1, 30, 10, key="low_returns")
        low_data = top_risk_metrics["Annual Return"].tail(n)
        fig = px.bar(x=low_data.values, y=low_data.index, orientation='h', color=low_data.values,
                     color_continuous_scale='Reds', title='⚠️ Underperformers by Annual Return')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(27, 31, 58, 0.5)',
                         font_color='#E8EAED', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        n = st.slider("Number of Stocks", 1, 30, 10, key="sharpe")
        sharpe_data = risk_metrics["Sharpe Ratio"].nlargest(n)
        fig = px.bar(x=sharpe_data.values, y=sharpe_data.index, orientation='h', color=sharpe_data.values,
                     color_continuous_scale='Blues', title='⭐ Best Risk-Adjusted Returns (Sharpe Ratio)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(27, 31, 58, 0.5)',
                         font_color='#E8EAED', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        n = st.slider("Number of Stocks", 1, 30, 10, key="volatility")
        vol_data = risk_metrics["Annual Volatility"].nlargest(n)
        fig = px.bar(x=vol_data.values, y=vol_data.index, orientation='h', color=vol_data.values,
                     color_continuous_scale='Purples', title='🚨 Highest Risk (Volatility)')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(27, 31, 58, 0.5)',
                         font_color='#E8EAED', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        n = st.slider("Number of Stocks", 1, 30, 10, key="drawdown")
        max_drawdowns = {}
        for ticker in close_price.columns:
            price = close_price[ticker]
            rolling_max = price.cummax()
            drawdown = (price - rolling_max) / rolling_max
            max_drawdowns[ticker] = drawdown.min()
        
        drawdown_data = pd.Series(max_drawdowns).nsmallest(n)
        fig = px.bar(x=drawdown_data.values, y=drawdown_data.index, orientation='h', color=drawdown_data.values,
                     color_continuous_scale='Oranges', title='📉 Maximum Drawdown Analysis')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(27, 31, 58, 0.5)',
                         font_color='#E8EAED', height=600)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🎲 Monte Carlo":
    st.markdown("<h1 style='text-align: center; color: #00D4FF;'>🎲 Monte Carlo Simulation</h1>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.selectbox("Select Stock Ticker", sorted(close_price.columns))
    with col2:
        simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Forecast Days", 1, 365, 252)
    with col2:
        show_percentiles = st.checkbox("Show Percentiles", value=True)
    
    if st.button("🚀 Run Simulation", use_container_width=True):
        with st.spinner("Computing Monte Carlo simulation..."):
            data = close_price[ticker].dropna()
            log_returns = np.log(data / data.shift(1)).dropna()
            mu = log_returns.mean()
            sigma = log_returns.std()
            last_price = data.iloc[-1]
            
            simulation_df = pd.DataFrame()
            for i in range(simulations):
                prices = [last_price]
                for _ in range(days):
                    shock = np.random.normal(mu, sigma)
                    price = prices[-1] * np.exp(shock)
                    prices.append(price)
                simulation_df[i] = prices
            
            fig = go.Figure()
            
            sample_size = min(100, simulations)
            for i in range(sample_size):
                fig.add_trace(go.Scatter(
                    y=simulation_df[i],
                    mode='lines',
                    line=dict(color=f'rgba(131, 56, 236, 0.1)'),
                    showlegend=False
                ))
            
            mean_path = simulation_df.mean(axis=1)
            fig.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                name='Expected Path',
                line=dict(color='#00D4FF', width=3)
            ))
            
            if show_percentiles:
                p5 = simulation_df.quantile(0.05, axis=1)
                p95 = simulation_df.quantile(0.95, axis=1)
                
                fig.add_trace(go.Scatter(y=p95, name='95th Percentile', 
                                        line=dict(color='#06FFA5', dash='dash')))
                fig.add_trace(go.Scatter(y=p5, name='5th Percentile',
                                        line=dict(color='#FF006E', dash='dash')))
            
            # Get currency info for the selected ticker
            currency_symbol, currency_code = get_currency_info(ticker)
            
            fig.update_layout(
                title=f"Monte Carlo Simulation: {ticker} - {company_names.get(ticker, ticker)} ({simulations} paths, {days} days)",
                xaxis_title="Days",
                yaxis_title=f"Price ({currency_symbol})",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(27, 31, 58, 0.5)',
                font_color='#E8EAED',
                height=600,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            final_prices = simulation_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><p style='color: #A0A3B8;'>Current Price</p><h3 style='color: #00D4FF;'>{currency_symbol}{last_price:.2f}</h3></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><p style='color: #A0A3B8;'>Expected Price</p><h3 style='color: #06FFA5;'>{currency_symbol}{final_prices.mean():.2f}</h3></div>", unsafe_allow_html=True)
            with col3:
                worst = np.percentile(final_prices, 5)
                st.markdown(f"<div class='metric-card'><p style='color: #A0A3B8;'>5% Worst Case</p><h3 style='color: #FF006E;'>{currency_symbol}{worst:.2f}</h3></div>", unsafe_allow_html=True)
            with col4:
                best = np.percentile(final_prices, 95)
                st.markdown(f"<div class='metric-card'><p style='color: #A0A3B8;'>95% Best Case</p><h3 style='color: #06FFA5;'>{currency_symbol}{best:.2f}</h3></div>", unsafe_allow_html=True)

else:  # System Info
    st.markdown("<h1 style='text-align: center; color: #00D4FF;'>ℹ️ System Information</h1>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 About This System</h3>
            <p>Advanced Quant Trading System v2.0 - A comprehensive stock market analysis platform with:</p>
            <ul style='color: #00D4FF;'>
                <li>Real-time Stock Data via Yahoo Finance</li>
                <li>Risk Metrics Analysis (Sharpe, Volatility)</li>
                <li>Monte Carlo Simulations</li>
                <li>Advanced Visualizations</li>
                <li>Multi-market Coverage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📈 Portfolio Statistics</h3>
            <p style='color: #A0A3B8;'>
                <strong style='color: #00D4FF;'>Total Assets:</strong> {len(close_price.columns)}<br>
                <strong style='color: #06FFA5;'>Avg Return:</strong> {risk_metrics['Annual Return'].mean():.2%}<br>
                <strong style='color: #FF006E;'>Avg Volatility:</strong> {risk_metrics['Annual Volatility'].mean():.2%}<br>
                <strong style='color: #8338EC;'>Avg Sharpe Ratio:</strong> {risk_metrics['Sharpe Ratio'].mean():.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Indian Stocks (NSE)")
        indian = [t for t in all_tickers if '.NS' in t]
        st.write(f"**Count:** {len(indian)}")
        for t in indian:
            currency_symbol, currency_code = get_currency_info(t)
            st.write(f"• **{t}** - {company_names.get(t, t)} ({currency_symbol})")
    
    with col2:
        st.subheader("🌎 Global Stocks")
        global_st = [t for t in all_tickers if '.NS' not in t and '.SR' not in t]
        st.write(f"**Count:** {len(global_st)}")
        for t in global_st:
            currency_symbol, currency_code = get_currency_info(t)
            st.write(f"• **{t}** - {company_names.get(t, t)} ({currency_symbol})")
    
    st.markdown("<div class='gradient-divider'></div>", unsafe_allow_html=True)
    
    st.subheader("📊 Complete Risk Metrics")
    st.dataframe(risk_metrics.head(20), use_container_width=True)







