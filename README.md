# 📈 TRIFUSION Forecasting Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trifusion-forecasting.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/github/last-commit/YOUR_USERNAME/trifusion-forecasting)](https://github.com/YOUR_USERNAME/trifusion-forecasting)

## 🎯 Overview

**TRIFUSION** is a production-ready, hybrid time series forecasting framework that intelligently combines three modeling paradigms:
- **🔢 Statistical Models** (ARIMA/SARIMAX)
- **🧠 Deep Learning** (Transformer/LSTM)
- **🤖 Large Language Models** (GPT-4 with RAG)

Designed specifically for forecasting Malaysia's Consumer Price Index (CPI) using Department of Statistics Malaysia (DOSM) open data, with automatic fallback mechanisms for robust cloud deployment.

### Key Features
- ✅ **Automatic Model Fusion**: Dynamic weighting based on uncertainty and performance
- ✅ **Regime Shift Detection**: Identifies COVID-19, policy shocks, and structural breaks
- ✅ **Interactive Visualizations**: Plotly-based dashboards with real-time metrics
- ✅ **Cloud-Ready**: CPU-only PyTorch, no local dependencies required
- ✅ **Fault Tolerant**: Graceful degradation when components fail
- ✅ **Federated Learning**: Cross-state model aggregation support
- ✅ **RAG-Enhanced LLM**: Retrieval-Augmented Generation for domain knowledge

---

## 🚀 Live Demo

**Launch the app instantly**:  
👉 **[https://trifusion-forecasting.streamlit.app](https://trifusion-forecasting.streamlit.app)**

---

## 📦 Quick Start

### **Option A: Deploy on Streamlit Cloud (Recommended - No Installation)**

1. **Fork this repository** on GitHub
2. **Sign in to [Streamlit Cloud](https://share.streamlit.io)** with your GitHub account
3. Click **"Deploy an app"** → Select your forked repo
4. Set **Main file path** to `dosm_simulation.py`
5. In **Advanced Settings**, select **Python 3.11**
6. Click **Deploy** and wait 5-10 minutes

### **Option B: Run Locally**

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/trifusion-forecasting.git
cd trifusion-forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run dosm_simulation.py
