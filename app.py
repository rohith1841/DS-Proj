import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Hyperliquid Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data and models
@st.cache_resource
def load_models():
    with open('pnl_bucket_model.pkl', 'rb') as f:
        pnl_model = pickle.load(f)
    with open('volatility_model.pkl', 'rb') as f:
        vol_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler_cluster.pkl', 'rb') as f:
        scaler_cluster = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return pnl_model, vol_model, scaler, kmeans, scaler_cluster, le

@st.cache_data
def load_data():
    trader_archetypes = pd.read_csv('trader_archetypes.csv')
    daily_metrics = pd.read_csv('daily_metrics.csv')
    model_data = pd.read_csv('model_data.csv')
    return trader_archetypes, daily_metrics, model_data

# Load everything
pnl_model, vol_model, scaler, kmeans, scaler_cluster, le = load_models()
trader_archetypes, daily_metrics, model_data = load_data()

# Archetype names
archetype_names = {
    0: "Steady Retail",
    1: "High-Frequency Pro",
    2: "Whale Trader", 
    3: "Consistent Winner"
}

# ==================== PAGE FUNCTIONS ====================

def show_overview():
    st.title("📈 Hyperliquid Sentiment Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes the relationship between Bitcoin Fear/Greed sentiment and 
    trader behavior on Hyperliquid. Explore how market sentiment affects trading performance,
    discover trader behavioral archetypes, and use predictive models.

    ### Key Metrics
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", f"{184263:,}")
    with col2:
        st.metric("Unique Traders", "32")
    with col3:
        st.metric("Date Range", "Mar 2023 - Feb 2025")
    with col4:
        st.metric("Avg Win Rate", "85.2%")

    st.markdown("---")

    # Summary insights
    st.subheader("🔍 Key Insights")

    insights = [
        "**Trading Activity Surges During Fear**: Traders execute 2.8x more trades during Fear days (4,183 avg) vs Greed days (1,491 avg)",
        "**Higher Per-Trade Profitability in Greed**: Average PnL/trade is 55% higher during Greed ($77.84) vs Fear ($50.05)",
        "**Consistent Winners Maintain Edge**: 90%+ win rate traders keep 95-97% win rates across ALL sentiment conditions",
        "**Long Bias Increases in Greed**: 69.8% long trades in Greed vs 67.3% in Fear"
    ]

    for insight in insights:
        st.markdown(f"- {insight}")

def show_sentiment_analysis():
    st.title("📊 Sentiment Analysis")

    # Sentiment comparison table
    st.subheader("Performance by Sentiment")

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Fear', 'Greed', 'Neutral'],
        'Total Trades': [133871, 43251, 7141],
        'Avg Trades/Trader': [4183.47, 1491.41, 892.63],
        'Avg PnL/Trade': [50.05, 77.84, 22.23],
        'Win Rate (%)': [86.12, 83.80, 80.55],
        'Total PnL ($)': [6699925, 3366582, 158742],
        'Long Ratio (%)': [67.32, 69.75, 40.79]
    })

    st.dataframe(sentiment_data.style.format({
        'Avg Trades/Trader': '{:.0f}',
        'Avg PnL/Trade': '${:.2f}',
        'Win Rate (%)': '{:.1f}%',
        'Total PnL ($)': '${:,.0f}',
        'Long Ratio (%)': '{:.1f}%'
    }), use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(sentiment_data, x='Sentiment', y='Total PnL ($)',
                      title='Total PnL by Sentiment',
                      color='Sentiment',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#27ae60', 'Neutral': '#95a5a6'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(sentiment_data, x='Sentiment', y='Win Rate (%)',
                      title='Win Rate by Sentiment',
                      color='Sentiment',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#27ae60', 'Neutral': '#95a5a6'})
        st.plotly_chart(fig2, use_container_width=True)

    # Trade frequency chart
    fig3 = px.bar(sentiment_data, x='Sentiment', y='Avg Trades/Trader',
                  title='Average Trades per Trader by Sentiment',
                  color='Sentiment',
                  color_discrete_map={'Fear': '#e74c3c', 'Greed': '#27ae60', 'Neutral': '#95a5a6'})
    st.plotly_chart(fig3, use_container_width=True)

def show_trader_archetypes():
    st.title("👥 Trader Behavioral Archetypes")

    st.markdown("""
    Using K-Means clustering, we've identified 4 distinct trader behavioral archetypes based on 
    trading patterns, profitability, and risk characteristics.
    """)

    # Archetype selector
    selected_archetype = st.selectbox(
        "Select Archetype to Explore",
        options=list(archetype_names.keys()),
        format_func=lambda x: f"{x}: {archetype_names[x]}"
    )

    # Filter data for selected archetype
    archetype_data = trader_archetypes[trader_archetypes['cluster'] == selected_archetype]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📌 {archetype_names[selected_archetype]}")
        st.markdown(f"**Trader Count:** {len(archetype_data)}")

        metrics = {
            "Avg Total PnL": f"${archetype_data['total_pnl'].mean():,.2f}",
            "Avg Win Rate": f"{archetype_data['win_rate'].mean():.1%}",
            "Avg Total Trades": f"{archetype_data['total_trades'].mean():,.0f}",
            "Avg Trade Size": f"${archetype_data['avg_trade_size'].mean():,.2f}",
            "Avg Trades/Day": f"{archetype_data['trades_per_day'].mean():,.0f}",
            "PnL Volatility": f"${archetype_data['pnl_volatility'].mean():,.2f}"
        }

        for metric, value in metrics.items():
            st.metric(metric, value)

    with col2:
        # Comparison radar chart
        all_profiles = trader_archetypes.groupby('cluster')[['total_pnl', 'win_rate', 'total_trades', 
                                                              'avg_trade_size', 'trades_per_day']].mean()

        # Normalize for radar chart
        normalized = all_profiles.copy()
        for col in normalized.columns:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

        categories = ['Total PnL', 'Win Rate', 'Total Trades', 'Avg Trade Size', 'Trades/Day']

        fig = go.Figure()

        for cluster_id in archetype_names.keys():
            values = normalized.loc[cluster_id].tolist()
            values += values[:1]  # Complete the circle

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=archetype_names[cluster_id],
                opacity=0.6 if cluster_id == selected_archetype else 0.2
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Archetype Comparison (Normalized)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # All archetypes comparison table
    st.subheader("All Archetypes Comparison")

    comparison_df = trader_archetypes.groupby('cluster').agg({
        'total_pnl': 'mean',
        'win_rate': 'mean',
        'total_trades': 'mean',
        'avg_trade_size': 'mean',
        'trades_per_day': 'mean',
        'pnl_volatility': 'mean'
    }).round(2)

    comparison_df['Archetype'] = comparison_df.index.map(archetype_names)
    comparison_df = comparison_df[['Archetype', 'total_pnl', 'win_rate', 'total_trades', 
                                   'avg_trade_size', 'trades_per_day', 'pnl_volatility']]

    st.dataframe(comparison_df.style.format({
        'total_pnl': '${:,.2f}',
        'win_rate': '{:.1%}',
        'total_trades': '{:.0f}',
        'avg_trade_size': '${:,.2f}',
        'trades_per_day': '{:.0f}',
        'pnl_volatility': '${:,.2f}'
    }), use_container_width=True)

def show_predictive_models():
    st.title("🔮 Predictive Models")

    st.markdown("""
    Use our trained machine learning models to predict next-day trader profitability and volatility.

    ### Model Performance
    - **PnL Bucket Classifier**: Random Forest with feature importance analysis
    - **Volatility Predictor**: Gradient Boosting Regressor
    """)

    # Prediction interface
    st.subheader("Make Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Features**")
        sentiment_value = st.slider("Sentiment Value", 0, 100, 50)
        sentiment_group = st.selectbox("Sentiment Group", ['Fear', 'Greed', 'Neutral'])
        total_pnl = st.number_input("Today's Total PnL ($)", value=10000.0)
        avg_pnl = st.number_input("Today's Avg PnL ($)", value=50.0)
        trade_count = st.number_input("Today's Trade Count", value=100, min_value=1)
        total_volume = st.number_input("Today's Total Volume ($)", value=500000.0)
        avg_trade_size = st.number_input("Today's Avg Trade Size ($)", value=5000.0)
        win_rate = st.slider("Today's Win Rate", 0.0, 1.0, 0.85)
        total_fees = st.number_input("Today's Total Fees ($)", value=100.0)
        pnl_volatility = st.number_input("Today's PnL Volatility ($)", value=500.0)

    with col2:
        st.markdown("**Predictions**")

        # Prepare input
        sentiment_encoded = 0 if sentiment_group == 'Fear' else (1 if sentiment_group == 'Greed' else 2)

        input_features = np.array([[sentiment_value, sentiment_encoded, total_pnl, avg_pnl,
                                    trade_count, total_volume, avg_trade_size, win_rate,
                                    total_fees, pnl_volatility]])

        input_scaled = scaler.transform(input_features)

        # Predict PnL bucket
        pnl_bucket_pred = pnl_model.predict(input_scaled)[0]

        # Predict volatility
        vol_pred = vol_model.predict(input_scaled)[0]

        st.metric("Predicted Next-Day PnL Bucket", pnl_bucket_pred)
        st.metric("Predicted Next-Day Volatility", f"${vol_pred:,.2f}")

        # Feature importance
        st.markdown("**Top Influential Features**")
        feature_importance = pd.DataFrame({
            'Feature': ['avg_trade_size', 'trade_count', 'total_volume', 'pnl_volatility', 
                       'win_rate', 'total_fees', 'total_pnl', 'avg_pnl', 'sentiment', 'sentiment_val'],
            'Importance': pnl_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(5)

        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance (PnL Bucket Model)')
        st.plotly_chart(fig, use_container_width=True)

def show_raw_data():
    st.title("📋 Raw Data")

    tab1, tab2, tab3 = st.tabs(["Daily Metrics", "Trader Archetypes", "Model Data"])

    with tab1:
        st.subheader("Daily Aggregate Metrics")
        st.dataframe(daily_metrics, use_container_width=True)

        csv1 = daily_metrics.to_csv(index=False)
        st.download_button("Download Daily Metrics", csv1, "daily_metrics.csv", "text/csv")

    with tab2:
        st.subheader("Trader Archetypes Data")
        st.dataframe(trader_archetypes, use_container_width=True)

        csv2 = trader_archetypes.to_csv(index=False)
        st.download_button("Download Trader Archetypes", csv2, "trader_archetypes.csv", "text/csv")

    with tab3:
        st.subheader("Model Training Data")
        st.dataframe(model_data, use_container_width=True)

        csv3 = model_data.to_csv(index=False)
        st.download_button("Download Model Data", csv3, "model_data.csv", "text/csv")

# ==================== NAVIGATION ====================

pages = [
    st.Page(show_overview, title="Overview", icon="📈", default=True),
    st.Page(show_sentiment_analysis, title="Sentiment Analysis", icon="📊"),
    st.Page(show_trader_archetypes, title="Trader Archetypes", icon="👥"),
    st.Page(show_predictive_models, title="Predictive Models", icon="🔮"),
    st.Page(show_raw_data, title="Raw Data", icon="📋"),
]

pg = st.navigation(pages)

# Run navigation
pg.run()

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit**")
st.sidebar.markdown("Data: Hyperliquid & Alternative.me Fear/Greed Index")

