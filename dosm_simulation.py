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
# TRIFUSION FORECASTING FRAMEWORK v3.1
# CPU-Only & Cloud-Ready
# =============================================================================

@dataclass
class ForecastConfig:
    """Configuration with validation"""
    statistical_order: Tuple[int, int, int] = (1, 1, 1)
    dl_architecture: str = "transformer"
    lookback: int = 36
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    epochs: int = 80
    learning_rate: float = 0.001
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
        assert 0 <= self.dropout <= 1, "Dropout must be in [0, 1]"

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

class DeepLearningForecaster:
    """Minimal DL forecaster with CPU fallback"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.history = None
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if PyTorch can be imported"""
        try:
            import torch
            return True
        except:
            st.warning("⚠️ PyTorch not available. Deep learning model disabled.")
            return False
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        if not self.is_available:
            return self
        
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
            
            self.history = y.copy()
            self.scaler = StandardScaler()
            y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Simple LSTM implementation
            class SimpleLSTM(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(1, 50, 2, batch_first=True)
                    self.linear = torch.nn.Linear(50, 1)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.linear(out[:, -1, :])
            
            self.model = SimpleLSTM()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Create sequences
            X, Y = [], []
            for i in range(len(y_scaled) - self.config.lookback):
                X.append(y_scaled[i:i + self.config.lookback])
                Y.append(y_scaled[i + self.config.lookback])
            
            X = torch.tensor(np.array(X), dtype=torch.float32).reshape(-1, self.config.lookback, 1)
            Y = torch.tensor(np.array(Y), dtype=torch.float32).reshape(-1, 1)
            
            # Train
            for epoch in range(self.config.epochs):
                optimizer.zero_grad()
                output = self.model(X)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
            
            st.success(f"✅ Deep learning model trained")
        except Exception as e:
            st.error(f"DL training failed: {str(e)}. Disabling DL component.")
            self.is_available = False
        return self
    
    def predict(self, y: np.ndarray, exog: Optional[np.ndarray] = None,
                steps: int = 1, uncertainty_quantification: bool = False) -> np.ndarray:
        if not self.is_available or self.model is None:
            return np.full(steps, y[-1] if len(y) > 0 else 0)
        
        try:
            import torch
            self.model.eval()
            
            y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
            last_seq = torch.tensor(y_scaled[-self.config.lookback:], dtype=torch.float32).reshape(1, self.config.lookback, 1)
            
            predictions = []
            for _ in range(steps):
                with torch.no_grad():
                    pred = self.model(last_seq).cpu().numpy()[0, 0]
                    predictions.append(pred)
                new_seq = torch.cat([last_seq[:, 1:, :], torch.tensor([[[pred]]], dtype=torch.float32)], dim=1)
                last_seq = new_seq
            
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            return predictions
        except:
            return np.full(steps, y[-1] if len(y) > 0 else 0)

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
        
        cache_key = f"{query}_{top_k}"
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
        
        prompt = f"""Forecast next {steps} values for: {y[-10:].tolist()}
Return JSON: {{"forecast": [...], "reasoning": "...", "confidence": 0.0-1.0}}
Context: {context or "None"}"""
        
        return self._call_llm(prompt, steps)
    
    def _call_llm(self, prompt: str, steps: int) -> Tuple[np.ndarray, str, float]:
        """Call OpenAI API with fallback"""
        try:
            import openai
            openai.api_key = self.config.api_key
            
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return np.array(data["forecast"][:steps]), data.get("reasoning", ""), float(data.get("confidence", 0.5))
        except:
            # Fallback to trend-based forecast
            if len(self.history) >= 5:
                trend = np.mean(self.history[-3:]) - np.mean(self.history[-5:-2])
            else:
                trend = 0
            forecast = self.history[-1] + np.arange(1, steps+1) * trend * 0.5
            return forecast, "LLM fallback - trend-based", 0.3

class MetaController:
    """Simple meta-controller"""
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.loss_history = {'statistical': [], 'deep_learning': [], 'llm': []}
        self.weights_history = []
    
    def compute_loss(self, y_true: float, y_pred: float, uncertainty: float, model_type: str):
        self.loss_history[model_type].append((y_true - y_pred) ** 2 + self.config.alpha * uncertainty)
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
        self.deep_learning = DeepLearningForecaster(config)
        self.rag = RAGPipeline(config)
        self.llm = LLMForecaster(config, self.rag)
        self.meta_controller = MetaController(config)
        self.history = None
        self.exog_history = None
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None, context: Optional[List[str]] = None):
        self.history = y.copy()
        self.exog_history = exog.copy() if exog is not None else None
        
        with st.spinner("🔧 Training Statistical Model..."):
            self.statistical.fit(y, exog)
        
        with st.spinner("🧠 Training Deep Learning Model..."):
            self.deep_learning.fit(y, exog)
        
        if context and self.config.use_rag:
            with st.spinner("📚 Building RAG Corpus..."):
                self.rag.build_corpus(context)
        
        st.success("🎯 All models trained!")
        return self
    
    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None,
                context: Optional[str] = None) -> Dict[str, Any]:
        pred_stat = self.statistical.predict(steps, exog_future)
        pred_deep = self.deep_learning.predict(self.history, self.exog_history, steps, self.config.uncertainty_weighting)
        pred_llm, reasoning, confidence = self.llm.predict(self.history, steps, context)
        
        uncertainties = {
            'statistical': 0.1,
            'deep_learning': 0.1 if not hasattr(self.deep_learning, 'uncertainty') else self.deep_learning.uncertainty,
            'llm': 1.0 - confidence
        }
        
        weights = self.meta_controller.update_weights(uncertainties)
        
        # Ensure all predictions are same length
        min_len = min(len(pred_stat), len(pred_deep), len(pred_llm))
        pred_stat, pred_deep, pred_llm = pred_stat[:min_len], pred_deep[:min_len], pred_llm[:min_len]
        
        hybrid = weights[0] * pred_stat + weights[1] * pred_deep + weights[2] * pred_llm
        
        return {
            'forecast': hybrid,
            'components': {'statistical': pred_stat, 'deep_learning': pred_deep, 'llm': pred_llm},
            'weights': {'statistical': weights[0], 'deep_learning': weights[1], 'llm': weights[2]},
            'uncertainties': uncertainties,
            'explanation': reasoning,
            'confidence_interval': self.statistical.get_confidence_interval(),
            'overall_confidence': 1.0 - np.dot(weights, list(uncertainties.values()))
        }
    
    def update_with_new_data(self, y_new: float, exog_new: Optional[np.ndarray] = None):
        if self.history is not None:
            self.history = np.append(self.history, y_new)
            if self.exog_history is not None and exog_new is not None:
                self.exog_history = np.vstack([self.exog_history, exog_new])

class DOSMDataLoader:
    """Robust data loader"""
    def __init__(self):
        self.base_url = "https://storage.dosm.gov.my"
        self.datasets = {"cpi_state": "/timeseries/cpi/cpi_2d_state.parquet"}
    
    @st.cache_data(ttl=3600)
    def load_cpi_data(self, state: str = "Malaysia", start_date: str = "2015-01-01") -> pd.DataFrame:
        try:
            url = f"{self.base_url}{self.datasets['cpi_state']}"
            df = pd.read_parquet(url)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            state_col = next((col for col in ['state', 'state_name'] if col in df.columns), None)
            if state_col:
                if state == "Malaysia":
                    df = df[df[state_col].isin(["Malaysia", "SEA_MALAYSIA", "MALAYSIA"])]
                else:
                    df = df[df[state_col] == state]
            
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.Timestamp.now())]
            df = df.sort_values('date').reset_index(drop=True)
            
            if df.empty:
                raise ValueError("No data")
            
            st.success(f"✅ Loaded {len(df)} CPI records for {state}")
            return df
        except Exception as e:
            st.error(f"❌ Failed to load DOSM data: {str(e)}")
            return self._generate_synthetic_cpi(state, start_date)
    
    def _generate_synthetic_cpi(self, state: str, start_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
        values = 100 + np.arange(len(dates)) * 0.1 + 2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
        return pd.DataFrame({'date': dates, 'state': state, 'index': values})
    
    @st.cache_data
    def load_exogenous_data(_self, start_date: str = "2015-01-01") -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
        data = []
        for date in dates:
            data.append({
                'date': date,
                'oil_price': 60 + np.random.normal(0, 10),
                'usd_myr': 4.2 + np.random.normal(0, 0.1),
                'policy_shock': 1.0 if date.month == 9 else 0.0,
                'covid_impact': 1.0 if 2020 <= date.year <= 2021 else 0.0
            })
        return pd.DataFrame(data)

class TRIFUSIONApp:
    """Streamlit app"""
    def __init__(self):
        self.config = None
        self.framework = None
        self.loader = DOSMDataLoader()
        self.state_options = ["Malaysia", "Selangor", "Johor", "Kedah", "Sabah", "Sarawak", "Penang"]
    
    def run(self):
        st.set_page_config(page_title="TRIFUSION Framework", layout="wide")
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #1f77b4, #ff7f0e); color: white; border-radius: 10px;">
            <h1>📈 TRIFUSION Forecasting Framework</h1>
            <p>Advanced Hybrid Forecasting with LLM Integration</p>
        </div>
        """, unsafe_allow_html=True)
        
        self._render_sidebar()
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()
    
    def _render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuration")
            dl_arch = st.selectbox("Deep Learning", ["transformer", "lstm"], index=0)
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 30, 200, 80)
            lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            state = st.selectbox("Select State", self.state_options, index=0)
            api_key = st.text_input("OpenAI API Key (optional)", type="password")
            
            self.config = ForecastConfig(
                dl_architecture=dl_arch,
                lookback=lookback,
                epochs=epochs,
                learning_rate=lr,
                api_key=api_key or None
            )
            
            st.markdown("---")
            st.info("Components: 🔢 ARIMA | 🧠 Transformer | 🤖 GPT-4")
    
    def _run_analysis(self):
        if not self.config.api_key:
            st.warning("⚠️ No API key provided. LLM will use fallback logic.")
        
        # Load data
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = self.loader.load_cpi_data(state=self.config.state, start_date="2015-01-01")
            exog_data = self.loader.load_exogenous_data(start_date="2015-01-01")
        
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True).dropna(subset=['index', 'date'])
        
        if full_data.empty:
            st.error("❌ No valid data")
            return
        
        # Overview
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Data Points", len(full_data))
        with col2: st.metric("CPI Range", f"{full_data['index'].min():.1f} - {full_data['index'].max():.1f}")
        with col3: st.metric("Date Range", f"{full_data['date'].min().strftime('%Y-%m')} to {full_data['date'].max().strftime('%Y-%m')}")
        
        # Prepare data
        y = full_data['index'].values
        exog = full_data[['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']].values
        
        train_size = len(y) - 24
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        # Initialize and fit framework
        self.framework = TRIFUSIONFramework(self.config)
        
        with st.spinner("🎯 Training components..."):
            context_docs = [
                "Malaysia CPI influenced by oil prices and fuel subsidies",
                "USD/MYR exchange rate affects import costs",
                "COVID-19 caused supply chain disruptions in 2020-2021",
                "SST implementation in September 2018 increased prices"
            ]
            self.framework.fit(y_train, exog_train, context=context_docs)
        
        # Rolling forecast
        with st.spinner("🔮 Generating forecasts..."):
            results = self._rolling_forecast(y_test, exog_test, full_data.iloc[train_size:])
        
        # Performance
        self._render_performance(results)
        
        # Export
        self._render_export_section(results)
    
    def _rolling_forecast(self, y_test: np.ndarray, exog_test: np.ndarray, test_data: pd.DataFrame) -> Dict[str, Any]:
        predictions = []
        actuals = []
        weights_history = []
        
        for i in range(len(y_test) - 3):
            history_end = len(self.framework.history) - len(y_test) + i + 1
            current_history = self.framework.history[:history_end]
            
            exog_future = exog_test[i:i+3]
            future_date = test_data.iloc[i]['date']
            context = self._generate_context(future_date)
            
            result = self.framework.predict(steps=3, exog_future=exog_future, context=context)
            
            if i < len(y_test) - 1:
                predictions.append(result['forecast'][0])
                actuals.append(y_test[i + 1])
                weights_history.append(result['weights'])
            
            self.framework.update_with_new_data(y_test[i], exog_test[i])
        
        return {
            'predictions': np.array(predictions),
            'actuals': np.array(actuals),
            'weights_history': weights_history,
            'dates': test_data['date'].values[1:len(predictions)+1]
        }
    
    def _generate_context(self, date: pd.Timestamp) -> str:
        if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-12-01'):
            return "COVID-19 pandemic ongoing"
        return "Normal economic conditions"
    
    def _render_performance(self, results: Dict[str, Any]):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        st.markdown("### 📈 Performance Analysis")
        
        mae = mean_absolute_error(results['actuals'], results['predictions'])
        rmse = np.sqrt(mean_squared_error(results['actuals'], results['predictions']))
        mape = np.mean(np.abs((results['actuals'] - results['predictions']) / results['actuals'])) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("RMSE", f"{rmse:.4f}")
        with col2: st.metric("MAE", f"{mae:.4f}")
        with col3: st.metric("MAPE", f"{mape:.2f}%")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['dates'], y=results['actuals'], mode='lines+markers', name='Actual CPI'))
        fig.add_trace(go.Scatter(x=results['dates'], y=results['predictions'], mode='lines+markers', name='TRIFUSION Forecast'))
        fig.update_layout(title="Forecast vs Actual", xaxis_title="Date", yaxis_title="CPI Index")
        st.plotly_chart(fig, use_container_width=True)
        
        # Weights
        weights_arr = np.array([list(w.values()) for w in results['weights_history']])
        fig2 = go.Figure()
        for i, name in enumerate(['Statistical', 'Deep Learning', 'LLM']):
            fig2.add_trace(go.Scatter(x=results['dates'], y=weights_arr[:, i], mode='lines', name=name))
        fig2.update_layout(title="Model Weights Over Time", xaxis_title="Date", yaxis_title="Weight")
        st.plotly_chart(fig2, use_container_width=True)
    
    def _render_export_section(self, results: Dict[str, Any]):
        st.markdown("### 💾 Export Results")
        
        export_df = pd.DataFrame({
            'date': results['dates'],
            'actual_cpi': results['actuals'],
            'forecast_cpi': results['predictions']
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"trifusion_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def main():
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
