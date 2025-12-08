import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# =============================================================================
# TRIFUSION FORECASTING FRAMEWORK v3.2
# Python 3.13 & Streamlit Cloud Compatible
# =============================================================================

@dataclass
class ForecastConfig:
    """Configuration with validation"""
    statistical_order: Tuple[int, int, int] = (1, 1, 1)
    lookback: int = 36
    epochs: int = 80
    llm_model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 300
    temperature: float = 0.2
    use_rag: bool = True
    rag_top_k: int = 10
    rag_hybrid_weight: float = 0.5
    window_size: int = 50
    alpha: float = 2.5
    guardrail_threshold: float = 0.4
    uncertainty_weighting: bool = True
    
    def validate(self):
        assert self.lookback > 0, "Lookback must be positive"
        assert 0 <= self.temperature <= 2, "Temperature must be in [0, 2]"

class StatisticalForecaster:
    """ARIMA/SARIMAX forecaster"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model_fit = None
        self.history = None
        
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            self.history = y.copy()
            
            if exog is not None:
                self.model_fit = SARIMAX(y, exog=exog, order=self.config.statistical_order).fit(disp=False)
            else:
                self.model_fit = ARIMA(y, order=self.config.statistical_order).fit(disp=False)
            
            st.success(f"✅ Statistical model fitted (AIC: {self.model_fit.aic:.2f})")
        except Exception as e:
            st.error(f"Statistical model failed: {str(e)}")
            self.model_fit = None
        return self
    
    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_fit is None:
            return np.full(steps, self.history[-1] if self.history is not None else 0)
        
        try:
            forecast = self.model_fit.get_forecast(steps=steps, exog=exog_future)
            return forecast.predicted_mean
        except:
            return np.full(steps, self.history[-1])
    
    def get_confidence_interval(self) -> Optional[np.ndarray]:
        if hasattr(self, 'model_fit') and self.model_fit is not None:
            try:
                return self.model_fit.get_forecast(steps=1).conf_int()
            except:
                pass
        return None

class DeepLearningForecaster:
    """PURE PYTHON FALLBACK - No PyTorch"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.history = None
        self.is_available = False  # Explicitly disabled for Python 3.13
        
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        """No-op: Store data only"""
        self.history = y.copy()
        st.warning("⚠️ Deep learning model disabled for Python 3.13 compatibility")
        return self
    
    def predict(self, y: np.ndarray, exog: Optional[np.ndarray] = None,
                steps: int = 1, uncertainty_quantification: bool = False) -> np.ndarray:
        """
        Fallback: Simple trend extrapolation with seasonal adjustment
        This replaces the PyTorch model with pure Python logic
        """
        if len(y) == 0:
            return np.full(steps, 0)
        
        last_value = y[-1]
        
        # Calculate trend (if enough data)
        if len(y) >= 20:
            recent_trend = np.mean(y[-10:]) - np.mean(y[-20:-10])
            long_trend = (y[-1] - y[0]) / len(y)
            trend = 0.7 * recent_trend + 0.3 * long_trend
        elif len(y) >= 5:
            trend = np.mean(y[-3:]) - np.mean(y[:-3])
        else:
            trend = 0
        
        # Add seasonal pattern (annual for monthly data)
        seasonal_pattern = np.sin(np.arange(steps) * 2 * np.pi / 12) * 0.02 * last_value
        
        # Generate predictions
        predictions = last_value + np.arange(1, steps + 1) * trend * 0.5 + seasonal_pattern
        
        # Add small noise for uncertainty quantification
        if uncertainty_quantification:
            noise = np.random.normal(0, 0.01 * np.std(y), steps)
            predictions += noise
        
        # Ensure predictions stay within reasonable bounds
        return np.maximum(predictions, last_value * 0.9)

class RAGPipeline:
    """BM25-based RAG (no semantic search dependency)"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.corpus = []
        self.bm25 = None
        self.cache = {}
    
    def build_corpus(self, corpus: List[str]):
        """Build BM25 index"""
        if not corpus:
            st.warning("Empty corpus provided")
            return
        
        from rank_bm25 import BM25Okapi
        
        self.corpus = corpus
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        st.success(f"✅ RAG corpus loaded ({len(corpus)} documents)")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float], List[bool]]:
        """BM25 retrieval only"""
        if not self.bm25:
            return [], [], []
        
        cache_key = f"query:{query}:k:{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        top_k = top_k or self.config.rag_top_k
        tokenized_query = query.lower().split()
        scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize scores
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved_docs = [self.corpus[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]
        is_reliable = [s > 0.3 for s in retrieved_scores]
        
        self.cache[cache_key] = (retrieved_docs, retrieved_scores, is_reliable)
        return retrieved_docs, retrieved_scores, is_reliable

class LLMForecaster:
    """LLM forecaster with robust fallback"""
    def __init__(self, config: ForecastConfig, rag_pipeline: Optional[RAGPipeline] = None):
        self.config = config
        self.rag = rag_pipeline
        self.history = None
    
    def predict(self, y: np.ndarray, steps: int = 1, context: Optional[str] = None) -> Tuple[np.ndarray, str, float]:
        """Generate LLM forecast"""
        self.history = y
        retrieved_docs, scores, reliability = [], [], []
        
        if self.config.use_rag and self.rag and context:
            retrieved_docs, scores, reliability = self.rag.retrieve(context)
        
        prompt = f"""You are an expert economic forecaster. 
        
TASK: Forecast the next {steps} values for the time series: {y[-10:].tolist()}

OUTPUT FORMAT (JSON ONLY):
{{"forecast": [number1, number2, ...], "reasoning": "Your analysis here", "confidence": 0.0-1.0}}

CONTEXT: {context or "No additional context provided"}

RELEVANT KNOWLEDGE: {' | '.join(retrieved_docs[:3]) if retrieved_docs else "None"}

RELIABILITY SCORES: {reliability[:3] if reliability else "N/A"}

GUIDELINES:
- Consider trend, seasonality, and recent anomalies
- Factor in economic context if provided
- Be conservative with confidence for volatile periods"""
        
        return self._call_llm(prompt, steps)
    
    def _call_llm(self, prompt: str, steps: int) -> Tuple[np.ndarray, str, float]:
        """Call OpenAI API with fallback"""
        try:
            import openai
            openai.api_key = self.config.api_key
            
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a forecasting assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            # Clean up potential markdown wrapping
            content = content.strip().strip('`').replace('```json', '').replace('```', '')
            
            data = json.loads(content)
            forecast = np.array(data["forecast"][:steps], dtype=float)
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = float(data.get("confidence", 0.5))
            
            return forecast, reasoning, confidence
            
        except Exception as e:
            st.warning(f"LLM unavailable: {str(e)[:100]}")
            return self._fallback_forecast(steps)
    
    def _fallback_forecast(self, steps: int) -> Tuple[np.ndarray, str, float]:
        """Robust trend-based fallback"""
        if len(self.history) >= 10:
            trend = np.mean(self.history[-5:]) - np.mean(self.history[-10:-5])
        elif len(self.history) >= 5:
            trend = np.mean(self.history[-3:]) - np.mean(self.history[:-3])
        else:
            trend = 0
        
        last_value = self.history[-1] if len(self.history) > 0 else 100
        forecast = last_value + np.arange(1, steps + 1) * trend * 0.5
        
        # Add small noise
        forecast += np.random.normal(0, abs(trend) * 0.1 + 0.01, steps)
        
        return forecast, "Trend-based fallback (LLM unavailable)", 0.3

class MetaController:
    """Simple meta-controller"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.loss_history = {'statistical': [], 'deep_learning': [], 'llm': []}
        self.weights_history = []
    
    def compute_loss(self, y_true: float, y_pred: float, uncertainty: float, model_type: str):
        base_loss = (y_true - y_pred) ** 2
        adjusted_loss = base_loss + self.config.alpha * uncertainty
        self.loss_history[model_type].append(adjusted_loss)
        
        if len(self.loss_history[model_type]) > self.config.window_size:
            self.loss_history[model_type].pop(0)
    
    def update_weights(self, uncertainties: Dict[str, float]) -> np.ndarray:
        avg_losses = [np.mean(self.loss_history[t]) if self.loss_history[t] else 0 for t in ['statistical', 'deep_learning', 'llm']]
        
        for i, model_type in enumerate(['statistical', 'deep_learning', 'llm']):
            if uncertainties and model_type in uncertainties:
                avg_losses[i] += uncertainties[model_type] * 2
        
        if avg_losses[2] > self.config.guardrail_threshold:
            avg_losses[2] = np.inf
        
        exp_terms = np.exp(-self.config.alpha * np.array(avg_losses))
        exp_terms = np.where(np.isinf(avg_losses), 0, exp_terms)
        
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp if sum_exp > 0 else np.array([1/3, 1/3, 1/3])
        self.weights_history.append(weights.copy())
        return weights

class TRIFUSIONFramework:
    """Main orchestrator"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.statistical = StatisticalForecaster(config)
        self.deep_learning = DeepLearningForecaster(config)  # Now pure Python fallback
        self.rag = RAGPipeline(config)
        self.llm = LLMForecaster(config, self.rag)
        self.meta_controller = MetaController(config)
        self.history = None
        self.exog_history = None
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None, context: Optional[List[str]] = None):
        """Fit all components"""
        self.history = y.copy()
        self.exog_history = exog.copy() if exog is not None else None
        
        with st.spinner("🔧 Training Statistical Model..."):
            self.statistical.fit(y, exog)
        
        with st.spinner("🧠 Training Deep Learning Model..."):
            self.deep_learning.fit(y, exog)  # Now instant
        
        if context and self.config.use_rag:
            with st.spinner("📚 Building RAG Corpus..."):
                self.rag.build_corpus(context)
        
        st.success("🎯 All models trained!")
        return self
    
    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None,
                context: Optional[str] = None) -> Dict[str, Any]:
        """Generate hybrid forecast"""
        pred_stat = self.statistical.predict(steps, exog_future)
        pred_deep = self.deep_learning.predict(self.history, self.exog_history, steps)
        pred_llm, reasoning, confidence = self.llm.predict(self.history, steps, context)
        
        # Handle different lengths
        min_len = min(len(pred_stat), len(pred_deep), len(pred_llm))
        pred_stat, pred_deep, pred_llm = pred_stat[:min_len], pred_deep[:min_len], pred_llm[:min_len]
        
        uncertainties = {
            'statistical': 0.1,
            'deep_learning': 0.1,  # Fixed uncertainty for fallback
            'llm': 1.0 - confidence
        }
        
        weights = self.meta_controller.update_weights(uncertainties)
        
        # Hybrid forecast
        hybrid = weights[0] * pred_stat + weights[1] * pred_deep + weights[2] * pred_llm
        
        return {
            'forecast': hybrid,
            'components': {
                'statistical': pred_stat,
                'deep_learning': pred_deep,
                'llm': pred_llm
            },
            'weights': {
                'statistical': weights[0],
                'deep_learning': weights[1],
                'llm': weights[2]
            },
            'uncertainties': uncertainties,
            'explanation': reasoning,
            'confidence_interval': self.statistical.get_confidence_interval(),
            'overall_confidence': 1.0 - np.dot(weights, list(uncertainties.values()))
        }
    
    def update_with_new_data(self, y_new: float, exog_new: Optional[np.ndarray] = None):
        """Online update"""
        if self.history is not None:
            self.history = np.append(self.history, y_new)
            if self.exog_history is not None and exog_new is not None:
                self.exog_history = np.vstack([self.exog_history, exog_new])

class DOSMDataLoader:
    """Robust data loader with fallback"""
    def __init__(self):
        self.base_url = "https://storage.dosm.gov.my"
        self.datasets = {"cpi_state": "/timeseries/cpi/cpi_2d_state.parquet"}
    
    @st.cache_data(ttl=3600)
    def load_cpi_data(self, state: str = "Malaysia", start_date: str = "2015-01-01") -> pd.DataFrame:
        """Load CPI data with caching"""
        try:
            url = f"{self.base_url}{self.datasets['cpi_state']}"
            df = pd.read_parquet(url)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            state_col = next((col for col in ['state', 'state_name'] if col in df.columns), None)
            if state_col:
                if state == "Malaysia":
                    df_filtered = df[df[state_col].isin(["Malaysia", "SEA_MALAYSIA", "MALAYSIA"])]
                else:
                    df_filtered = df[df[state_col] == state]
            else:
                df_filtered = df
            
            df_filtered = df_filtered[
                (df_filtered['date'] >= pd.to_datetime(start_date)) &
                (df_filtered['date'] <= pd.Timestamp.now())
            ].sort_values('date').reset_index(drop=True)
            
            if df_filtered.empty:
                raise ValueError("No data after filtering")
            
            st.success(f"✅ Loaded {len(df_filtered)} CPI records for {state}")
            return df_filtered
            
        except Exception as e:
            st.error(f"❌ Failed to load DOSM data: {str(e)}")
            return self._generate_synthetic_cpi(state, start_date)
    
    def _generate_synthetic_cpi(self, state: str, start_date: str) -> pd.DataFrame:
        """Generate realistic synthetic CPI data"""
        dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
        
        # Realistic CPI generation with trends and breaks
        base_cpi = 100
        values = []
        
        for i, date in enumerate(dates):
            # Long-term trend
            trend = 1 + (date.year - 2015) * 0.015 + i * 0.0003
            
            # Seasonality
            seasonal = 1 + 0.02 * np.sin(2 * np.pi * date.month / 12)
            
            # Structural breaks
            breaks = 1.0
            if pd.Timestamp('2018-09-01') <= date <= pd.Timestamp('2018-12-01'):
                breaks *= 1.02  # SST
            if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-06-01'):
                breaks *= 1.04  # COVID
            if pd.Timestamp('2022-06-01') <= date <= pd.Timestamp('2022-09-01'):
                breaks *= 1.03  # Subsidy reform
            
            noise = np.random.normal(0, 0.25)
            cpi = base_cpi * trend * seasonal * breaks + noise
            values.append(max(95, min(135, cpi)))
        
        return pd.DataFrame({'date': dates, 'state': state, 'index': values})
    
    @st.cache_data
    def load_exogenous_data(_self, start_date: str = "2015-01-01") -> pd.DataFrame:
        """Generate synthetic exogenous variables"""
        dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
        
        data = []
        for date in dates:
            # Oil price: realistic fluctuations
            oil_trend = 60 + (date.year - 2020) * 2
            oil_cycle = 15 * np.sin(2 * np.pi * date.year / 3)
            oil_price = max(40, min(120, oil_trend + oil_cycle + np.random.normal(0, 5)))
            
            # USD/MYR: realistic exchange rate
            usd_myr_trend = 4.2 + 0.1 * np.sin(2 * np.pi * date.year / 5)
            usd_myr = max(3.8, min(4.8, usd_myr_trend + np.random.normal(0, 0.08)))
            
            # Policy shocks (binary events)
            policy_shock = 1.0 if date in [pd.Timestamp('2018-09-01'), pd.Timestamp('2022-06-01')] else 0.0
            
            # COVID impact (graduated)
            if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-03-01'):
                covid_impact = 1.5
            elif pd.Timestamp('2021-04-01') <= date <= pd.Timestamp('2021-12-01'):
                covid_impact = 1.2
            else:
                covid_impact = 0.0
            
            data.append({
                'date': date,
                'oil_price': oil_price,
                'usd_myr': usd_myr,
                'policy_shock': policy_shock,
                'covid_impact': covid_impact
            })
        
        return pd.DataFrame(data)

class TRIFUSIONApp:
    """Streamlit app with professional UI"""
    def __init__(self):
        self.config = None
        self.framework = None
        self.loader = DOSMDataLoader()
        self.state_options = ["Malaysia", "Selangor", "Johor", "Kedah", "Sabah", "Sarawak", "Penang"]
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="TRIFUSION Forecasting Framework",
            page_icon="📈",
            layout="wide"
        )
        
        # Custom styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>📈 TRIFUSION Forecasting Framework</h1>
            <p>Advanced Hybrid Time Series Forecasting with LLM Integration</p>
            <p style="color: #eee; font-size: 0.9em;">Powered by DOSM Malaysia Open Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        self._render_sidebar()
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()
    
    def _render_sidebar(self):
        """Render configuration sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            st.subheader("Model Architecture")
            # Deep learning option disabled for compatibility
            st.info("Deep Learning: **Fallback Mode** (PyTorch-free)")
            
            st.subheader("Hyperparameters")
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 30, 200, 80)
            lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            
            st.subheader("Data Settings")
            state = st.selectbox("Select State", self.state_options, index=0)
            start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
            
            st.subheader("LLM Settings")
            api_key = st.text_input("OpenAI API Key (optional)", type="password")
            use_rag = st.checkbox("Enable RAG", value=True)
            
            st.subheader("Advanced")
            uncertainty_weighting = st.checkbox("Uncertainty Weighting", value=True)
            
            self.config = ForecastConfig(
                lookback=lookback,
                epochs=epochs,
                learning_rate=lr,
                api_key=api_key or None,
                use_rag=use_rag,
                uncertainty_weighting=uncertainty_weighting
            )
            
            st.markdown("---")
            st.info("""
            **Components:**
            - 🔢 **ARIMA/SARIMAX** (Statistical)
            - 🧠 **Pure Python** (Deep Learning Fallback)
            - 🤖 **GPT-4** (LLM with RAG)
            - ⚖️ **Dynamic Fusion**
            """)
    
    def _run_analysis(self):
        """Execute full forecasting analysis"""
        if not self.config.api_key:
            st.warning("⚠️ No OpenAI API key provided. LLM will use fallback logic.")
        
        # Load data
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = self.loader.load_cpi_data(state=self.config.state, start_date="2015-01-01")
            exog_data = self.loader.load_exogenous_data(start_date="2015-01-01")
        
        # Merge datasets
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        
        # Fill missing values
        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        full_data[numeric_cols] = full_data[numeric_cols].fillna(method='ffill').fillna(0)
        full_data = full_data.dropna(subset=['index', 'date'])
        
        if full_data.empty:
            st.error("❌ No valid data available after merging. Please check data sources.")
            return
        
        # Display data overview
        self._render_data_overview(full_data)
        
        # Prepare data
        y = full_data['index'].values
        exog = full_data[numeric_cols].values
        
        # Train-test split
        train_size = len(y) - 24
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        # Initialize framework
        self.framework = TRIFUSIONFramework(self.config)
        
        # Fit
        with st.spinner("🎯 Training components..."):
            context_docs = [
                "Malaysia CPI is influenced by oil prices due to fuel subsidies",
                "USD/MYR exchange rate affects import costs and inflation",
                "COVID-19 caused supply chain disruptions in 2020-2021",
                "SST implementation in September 2018 increased prices temporarily",
                "Fuel subsidy rationalization in June 2022 caused inflation spike"
            ]
            self.framework.fit(y_train, exog_train, context=context_docs)
        
        # Rolling forecast
        with st.spinner("🔮 Generating forecasts..."):
            results = self._rolling_forecast(y_test, exog_test, full_data.iloc[train_size:])
        
        # Performance dashboard
        self._render_performance_dashboard(results)
        
        # Export
        self._render_export_section(results)
    
    def _render_data_overview(self, data: pd.DataFrame):
        """Display data summary metrics"""
        st.markdown("### 📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Data Points", len(data))
        with col2: st.metric("CPI Range", f"{data['index'].min():.1f} - {data['index'].max():.1f}")
        with col3: st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")
        with col4: st.metric("Missing Values", f"{data['index'].isna().sum()} ({data['index'].isna().mean():.1%})")
        
        # Interactive line plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['index'],
            mode='lines+markers',
            name='CPI Index',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title="Consumer Price Index Over Time",
            xaxis_title="Date",
            yaxis_title="CPI Index",
            hovermode='x unified',
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight regime shifts
        self._highlight_regime_shifts(data)
    
    def _highlight_regime_shifts(self, data: pd.DataFrame):
        """Identify and display major economic events"""
        st.markdown("#### 🔔 Identified Regime Shifts")
        
        shifts = []
        
        # COVID period
        covid_start = data[(data['date'] >= '2020-03-01') & (data['date'] <= '2020-04-01')]
        if not covid_start.empty:
            shifts.append({
                'date': covid_start.iloc[0]['date'],
                'name': '🦠 COVID-19 Pandemic',
                'desc': 'Supply chain disruption and demand shock'
            })
        
        # Policy events
        if 'policy_shock' in data.columns and data['policy_shock'].sum() > 0:
            policy_shocks = data[data['policy_shock'] > 0]
            for _, row in policy_shocks.iterrows():
                shifts.append({
                    'date': row['date'],
                    'name': '📋 Policy Shock',
                    'desc': 'Tax or subsidy policy change'
                })
        
        for shift in shifts:
            st.info(f"**{shift['name']}** ({shift['date'].strftime('%Y-%m')}) - {shift['desc']}")
    
    def _rolling_forecast(self, y_test: np.ndarray, exog_test: np.ndarray, 
                         test_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform rolling window forecast simulation"""
        predictions = []
        actuals = []
        weights_history = []
        uncertainties_history = []
        
        progress_bar = st.progress(0)
        
        for i in range(len(y_test) - 3):
            # Prepare current history
            history_end = len(self.framework.history) - len(y_test) + i + 1
            current_history = self.framework.history[:history_end]
            
            # Prepare exogenous future
            exog_future = exog_test[i:i+3] if i + 3 <= len(exog_test) else exog_test[i:]
            future_date = test_data.iloc[i]['date']
            
            # Generate context
            context = self._generate_context(future_date)
            
            # Generate forecast
            result = self.framework.predict(steps=3, exog_future=exog_future, context=context)
            
            # Store results
            if i < len(y_test) - 1:
                predictions.append(result['forecast'][0])
                actuals.append(y_test[i + 1])
                weights_history.append(result['weights'])
                uncertainties_history.append(result['uncertainties'])
            
            # Update model with new data
            self.framework.update_with_new_data(y_test[i], exog_test[i] if exog_test is not None else None)
            
            # Update progress
            progress_bar.progress((i + 1) / len(y_test))
        
        progress_bar.empty()
        
        return {
            'predictions': np.array(predictions),
            'actuals': np.array(actuals),
            'weights_history': weights_history,
            'uncertainties_history': uncertainties_history,
            'dates': test_data['date'].values[1:len(predictions)+1]
        }
    
    def _generate_context(self, date: pd.Timestamp) -> str:
        """Generate contextual text for LLM based on date"""
        contexts = []
        
        if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-12-01'):
            contexts.append("COVID-19 pandemic ongoing with supply chain disruptions")
        
        if date.month in [11, 12]:
            contexts.append("Year-end holiday season typically sees higher consumer spending")
        
        if date.year == 2018 and date.month == 9:
            contexts.append("Sales and Service Tax (SST) implementation impact")
        
        if date.year == 2022 and date.month == 6:
            contexts.append("Fuel subsidy rationalization policy announced")
        
        return " | ".join(contexts) if contexts else "Normal economic conditions"
    
    def _render_performance_dashboard(self, results: Dict[str, Any]):
        """Display comprehensive performance analysis"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        st.markdown("### 📊 Performance Dashboard")
        
        # Calculate metrics
        mae = mean_absolute_error(results['actuals'], results['predictions'])
        rmse = np.sqrt(mean_squared_error(results['actuals'], results['predictions']))
        mape = np.mean(np.abs((results['actuals'] - results['predictions']) / results['actuals'])) * 100
        r2 = 1 - (np.sum((results['actuals'] - results['predictions'])**2) / 
                  np.sum((results['actuals'] - np.mean(results['actuals']))**2))
        
        # Display metrics
        metrics_cols = st.columns(5)
        metrics = [
            ("RMSE", f"{rmse:.4f}"),
            ("MAE", f"{mae:.4f}"),
            ("MAPE", f"{mape:.2f}%"),
            ("R² Score", f"{r2:.4f}"),
            ("Observations", len(results['actuals']))
        ]
        
        for col, (label, value) in zip(metrics_cols, metrics):
            with col:
                st.metric(label, value)
        
        # Interactive forecast vs actual plot
        self._render_interactive_plot(results)
        
        # Component analysis
        self._render_component_analysis(results)
        
        # Error analysis
        self._render_error_analysis(results)
    
    def _render_interactive_plot(self, results: Dict[str, Any]):
        """Create interactive Plotly chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Forecast vs Actual", "Model Weights Over Time", "Model Uncertainties"),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        dates = results['dates']
        
        # Forecast vs Actual
        fig.add_trace(
            go.Scatter(x=dates, y=results['actuals'], mode='lines+markers', name='Actual CPI',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=results['predictions'], mode='lines+markers', name='TRIFUSION Forecast',
                      line=dict(color='#ff7f0e', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Model weights
        weights_arr = np.array([list(w.values()) for w in results['weights_history']])
        for i, name in enumerate(['Statistical', 'Deep Learning', 'LLM']):
            fig.add_trace(
                go.Scatter(x=dates, y=weights_arr[:, i], mode='lines', name=f'{name} Weight'),
                row=2, col=1
            )
        
        # Uncertainties
        unc_df = pd.DataFrame(results['uncertainties_history'])
        for col in unc_df.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=unc_df[col], mode='lines', name=f'{col} Uncertainty'),
                row=3, col=1
            )
        
        fig.update_layout(height=800, showlegend=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_component_analysis(self, results: Dict[str, Any]):
        """Analyze component contributions"""
        st.markdown("#### 🧩 Component Analysis")
        
        weights_arr = np.array([list(w.values()) for w in results['weights_history']])
        avg_weights = weights_arr.mean(axis=0)
        
        weight_df = pd.DataFrame({
            'Component': ['Statistical', 'Deep Learning', 'LLM'],
            'Avg Weight': avg_weights,
            'Std Dev': weights_arr.std(axis=0)
        })
        
        # Bar chart
        fig = go.Figure(data=[
            go.Bar(x=weight_df['Component'], y=weight_df['Avg Weight'],
                   error_y=dict(type='data', array=weight_df['Std Dev']),
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig.update_layout(title="Average Model Weights", xaxis_title="Component", yaxis_title="Weight")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_error_analysis(self, results: Dict[str, Any]):
        """Analyze forecast errors"""
        st.markdown("#### 📉 Error Analysis")
        
        errors = results['actuals'] - results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution histogram
            fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=30, marker_color='#9467bd')])
            fig.update_layout(title="Error Distribution", xaxis_title="Error", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot
            from scipy import stats
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(errors)))
            sample_quantiles = np.quantile(errors, np.linspace(0.01, 0.99, len(errors)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Q-Q'))
            fig.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines', name='Ideal'))
            fig.update_layout(title="Q-Q Plot of Errors", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
            st.plotly_chart(fig, use_container_width=True)
        
        # Error statistics table
        error_stats = pd.DataFrame({
            'Statistic': ['Mean Error', 'MAE', 'RMSE', 'MAPE (%)'],
            'Value': [
                np.mean(errors),
                np.mean(np.abs(errors)),
                np.sqrt(np.mean(errors**2)),
                np.mean(np.abs(errors / results['actuals'])) * 100
            ]
        })
        st.dataframe(error_stats.style.format({'Value': '{:.4f}'}), use_container_width=True)
    
    def _render_export_section(self, results: Dict[str, Any]):
        """Export results to CSV and JSON"""
        st.markdown("### 💾 Export Results")
        
        export_df = pd.DataFrame({
            'date': results['dates'],
            'actual_cpi': results['actuals'],
            'forecast_cpi': results['predictions']
        })
        
        # Add components if available
        if 'components' in results:
            for component, values in results['components'].items():
                if len(values) >= len(results['dates']):
                    export_df[f'{component}_forecast'] = values[:len(results['dates'])]
        
        # CSV Export
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"trifusion_forecast_{self.config.state}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # JSON Export with metadata
        json_export = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'state': self.config.state,
                'configuration': self.config.__dict__
            },
            'results': {
                'dates': [d.isoformat() for d in results['dates']],
                'actuals': results['actuals'].tolist(),
                'predictions': results['predictions'].tolist(),
                'weights': [list(w.values()) for w in results['weights_history']],
                'uncertainties': results['uncertainties_history']
            },
            'metrics': {
                'rmse': float(np.sqrt(np.mean((results['actuals'] - results['predictions'])**2))),
                'mae': float(np.mean(np.abs(results['actuals'] - results['predictions']))),
                'mape': float(np.mean(np.abs((results['actuals'] - results['predictions']) / results['actuals'])) * 100)
            }
        }
        
        json_str = json.dumps(json_export, indent=2)
        b64_json = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64_json}" download="trifusion_full_results_{datetime.now().strftime("%Y%m%d")}.json">📥 Download Full Results (JSON)</a>'
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Application entry point"""
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
