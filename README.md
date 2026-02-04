# Hyperliquid Sentiment Analysis Dashboard

A lightweight Streamlit dashboard to explore the relationship between Bitcoin Fear/Greed sentiment and trader behavior on Hyperliquid.

## Features

- **Overview**: Key metrics and insights summary
- **Sentiment Analysis**: Performance comparison across Fear/Greed/Neutral periods
- **Trader Archetypes**: Interactive exploration of 4 behavioral clusters
- **Predictive Models**: Make predictions using trained ML models
- **Raw Data**: Download all analysis data

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

## Models Included

1. **PnL Bucket Classifier** (Random Forest): Predicts next-day profitability category
2. **Volatility Predictor** (Gradient Boosting): Predicts next-day PnL volatility
3. **K-Means Clustering**: Identifies 4 trader behavioral archetypes

## Data Sources

- Hyperliquid trading data (32 traders, 184K+ trades)
- Alternative.me Bitcoin Fear & Greed Index
