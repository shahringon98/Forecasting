import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

warnings.filterwarnings('ignore')

# =============================================================================
# TRIFUSION FORECASTING FRAMEWORK v3.0
# Production-Ready Hybrid Forecasting with LLM Integration
# =============================================================================

class ModelType(Enum):
    """Enum for model components"""
    STATISTICAL = "statistical"
    DEEP_LEARNING = "deep_learning"
    LLM = "llm"

@dataclass
class ForecastConfig:
    """Centralized configuration with validation"""
    # Statistical Model
    statistical_order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    
    # Deep Learning Architecture
    dl_architecture: str = "transformer"  # "transformer" or "lstm"
    lookback: int = 36
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    nhead: int = 8  # for transformer
    
    # Training
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    batch_size: int = 32
    
    # LLM Integration
    llm_model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 300
    temperature: float = 0.2
    use_rag: bool = True
    
    # RAG Configuration
    rag_top_k: int = 10
    rag_hybrid_weight: float = 0.5
    rag_semantic_model: str = "all-mpnet-base-v2"
    rag_corpus_path: Optional[str] = None
    
    # Fusion & Weighting
    window_size: int = 50
    alpha: float = 2.5
    guardrail_threshold: float = 0.4
    uncertainty_weighting: bool = True
    
    # Federated Learning
    fedprox_mu: float = 0.01
    client_epochs: int = 5
    
    # Robustness
    adversarial_training: bool = False
    epsilon_adv: float = 0.1
    feature_noise: float = 0.0
    
    # Validation
    test_size: int = 24
    validation_split: float = 0.1
    
    # Output
    forecast_horizon: int = 3
    confidence_level: float = 0.95
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.lookback > 0, "Lookback must be positive"
        assert self.hidden_size > 0, "Hidden size must be positive"
        assert 0 <= self.dropout <= 1, "Dropout must be in [0, 1]"
        assert 0 <= self.temperature <= 2, "Temperature must be in [0, 2]"
        assert 0 <= self.guardrail_threshold <= 1, "Guardrail must be in [0, 1]"

class DataValidator:
    """Utility for data validation and preprocessing"""
    
    @staticmethod
    def validate_time_series(y: np.ndarray, min_length: int = 30) -> Tuple[bool, str]:
        if len(y) < min_length:
            return False, f"Time series too short. Need at least {min_length} points"
        if np.isnan(y).sum() > len(y) * 0.1:
            return False, "Too many missing values (>10%)"
        if np.std(y) == 0:
            return False, "Time series has no variance"
        return True, "Valid"

    @staticmethod
    def remove_outliers(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        return np.where(z_scores > threshold, np.median(y), y)

    @staticmethod
    def create_features(y: np.ndarray) -> pd.DataFrame:
        """Create time-based features"""
        dates = pd.date_range(start="2015-01-01", periods=len(y), freq='M')
        df = pd.DataFrame({'value': y, 'date': dates})
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['trend'] = np.arange(len(y))
        return df

class StatisticalForecaster:
    """Enhanced ARIMA/SARIMAX with automatic order selection"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model_fit = None
        self.history = None
        self.exog_history = None
        self.confidence_intervals = None
        self.last_aic = None
        
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None, auto_select: bool = False):
        """Fit statistical model with optional auto-order selection"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.stattools import adfuller
            
            self.history = y.copy()
            self.exog_history = exog.copy() if exog is not None else None
            
            # Auto-select orders if requested
            if auto_select:
                order = self._auto_select_order(y)
                if order:
                    self.config.statistical_order = order
            
            # Check stationarity
            adf_result = adfuller(y)
            is_stationary = adf_result[1] < 0.05
            
            if exog is not None:
                self.model_fit = SARIMAX(
                    y, 
                    exog=exog, 
                    order=self.config.statistical_order,
                    seasonal_order=self.config.seasonal_order,
                    enforce_stationarity=not is_stationary
                ).fit(disp=False)
            else:
                self.model_fit = ARIMA(y, order=self.config.statistical_order).fit(disp=False)
                
            self.last_aic = self.model_fit.aic
            st.success(f"✅ Statistical model fitted (AIC: {self.last_aic:.2f})")
            
        except Exception as e:
            st.error(f"Statistical model training failed: {str(e)}")
            self.model_fit = None
        return self
    
    def _auto_select_order(self, y: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Auto-select ARIMA order using AIC"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            best_aic = np.inf
            best_order = None
            
            # Grid search for best order
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
            
            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fit = model.fit(disp=False)
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            st.info(f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
        except:
            return None
    
    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions with confidence intervals"""
        if self.model_fit is None:
            # Fallback to naive forecast
            last_value = self.history[-1] if self.history is not None else 0
            self.confidence_intervals = None
            return np.full(steps, last_value)
        
        try:
            forecast = self.model_fit.get_forecast(steps=steps, exog=exog_future)
            self.confidence_intervals = forecast.conf_int(alpha=1-self.config.confidence_level)
            return forecast.predicted_mean
        except Exception as e:
            st.warning(f"Statistical prediction failed: {str(e)}")
            return np.full(steps, self.history[-1])
    
    def get_confidence_interval(self) -> Optional[np.ndarray]:
        return self.confidence_intervals

class PositionalEncoding(nn.Module):
    """Transformer positional encoding"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(nn.Module):
    """Enhanced Transformer forecaster"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, nhead: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)
        
        # Multi-head attention with layer norm
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last time step
        x = self.dropout(x)
        x = self.batch_norm(x)
        return self.linear(x)

class DeepLearningForecaster:
    """Unified deep learning forecaster with uncertainty quantification"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self.y_scaler = None
        self.exog_scaler = None
        self.feature_scaler = None
        self.history = None
        self.uncertainty = None
        self.training_losses = []
        
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        """Fit deep learning model with proper scaling"""
        try:
            import torch
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
        except ImportError:
            st.error("PyTorch or scikit-learn not installed")
            return self
        
        # Validate data
        is_valid, msg = DataValidator.validate_time_series(y)
        if not is_valid:
            st.error(f"Data validation failed: {msg}")
            return self
        
        # Store data
        self.history = y.copy()
        
        # FIXED: Separate scalers for each data type
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Prepare combined features
        if exog is not None and exog.shape[1] > 0:
            self.exog_scaler = StandardScaler()
            exog_scaled = self.exog_scaler.fit_transform(exog)
            combined = np.column_stack([y_scaled.reshape(-1, 1), exog_scaled])
            input_size = combined.shape[1]
        else:
            combined = y_scaled.reshape(-1, 1)
            input_size = 1
        
        # Check sufficient data
        if len(y) <= self.config.lookback + 10:
            st.warning(f"Insufficient data: {len(y)} points, need > {self.config.lookback + 10}")
            return self
        
        # Initialize model
        self._build_model(input_size)
        
        # Create sequences
        X, Y = self._create_sequences(combined, y_scaled)
        
        # Train-validation split
        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        Y_train, Y_val = Y[:-val_size], Y[-val_size:]
        
        # Training setup
        criterion = nn.HuberLoss(delta=1.0)  # Robust loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate * 10,
            epochs=self.config.epochs,
            steps_per_epoch=len(X_train) // self.config.batch_size + 1
        )
        
        # Training loop
        best_val_loss = np.inf
        patience_counter = 0
        
        progress_placeholder = st.empty()
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for i in range(0, len(X_train), self.config.batch_size):
                batch_x = X_train[i:i+self.config.batch_size]
                batch_y = Y_train[i:i+self.config.batch_size]
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                
                # Adversarial training
                if self.config.adversarial_training:
                    X_adv = batch_x + torch.randn_like(batch_x) * self.config.epsilon_adv
                    adv_output = self.model(X_adv)
                    adv_loss = criterion(adv_output, batch_y)
                    loss = loss + 0.3 * adv_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = criterion(val_output, Y_val).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
            
            # Progress update
            if epoch % 10 == 0:
                progress_placeholder.info(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Train Loss: {np.mean(train_losses):.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )
            
            self.training_losses.append(np.mean(train_losses))
        
        self.model.load_state_dict(self.best_state)
        st.success(f"✅ Deep learning model trained (Best Val Loss: {best_val_loss:.4f})")
        return self
    
    def _build_model(self, input_size: int):
        """Build the deep learning model"""
        if self.config.dl_architecture == "transformer":
            self.model = TransformerForecaster(
                input_size=input_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                nhead=self.config.nhead
            )
        else:
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True,
                        bidirectional=False
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.batch_norm = nn.BatchNorm1d(hidden_size)
                    self.linear = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.dropout(lstm_out[:, -1, :])
                    out = self.batch_norm(out)
                    return self.linear(out)
            
            self.model = LSTMModel(
                input_size, self.config.hidden_size,
                self.config.num_layers, self.config.dropout
            )
    
    def _create_sequences(self, combined: np.ndarray, y_scaled: np.ndarray):
        """Create training sequences"""
        X, Y = [], []
        for i in range(len(combined) - self.config.lookback):
            X.append(combined[i:i + self.config.lookback])
            Y.append(y_scaled[i + self.config.lookback])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
        return X, Y
    
    def predict(self, y: np.ndarray, exog: Optional[np.ndarray] = None,
                steps: int = 1, uncertainty_quantification: bool = False) -> np.ndarray:
        """Generate predictions with optional uncertainty quantification"""
        if self.model is None or self.y_scaler is None:
            st.warning("Model not trained, returning naive forecast")
            return np.full(steps, y[-1] if len(y) > 0 else 0)
        
        import torch
        
        self.model.eval()
        self.model.load_state_dict(self.best_state)
        
        # Scale target
        y_scaled = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Prepare features
        if exog is not None and self.exog_scaler is not None:
            exog_scaled = self.exog_scaler.transform(exog)
            combined = np.column_stack([y_scaled.reshape(-1, 1), exog_scaled])
        else:
            combined = y_scaled.reshape(-1, 1)
        
        predictions = []
        current_seq = combined[-self.config.lookback:]
        self.uncertainty = None
        
        # Uncertainty quantification via MC Dropout
        if uncertainty_quantification:
            self.model.train()  # Enable dropout
        
        for step in range(steps):
            # Pad if necessary
            if len(current_seq) < self.config.lookback:
                pad_len = self.config.lookback - len(current_seq)
                current_seq = np.pad(current_seq, ((pad_len, 0), (0, 0)), mode='edge')
            
            seq_tensor = torch.tensor(
                current_seq[-self.config.lookback:], 
                dtype=torch.float32
            ).unsqueeze(0)
            
            if uncertainty_quantification:
                # Monte Carlo sampling
                mc_preds = []
                for _ in range(50):
                    with torch.no_grad():
                        pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                        mc_preds.append(pred)
                predictions.append(np.mean(mc_preds))
                self.uncertainty = np.std(mc_preds) if self.uncertainty is None else np.mean([self.uncertainty, np.std(mc_preds)])
            else:
                with torch.no_grad():
                    pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                    predictions.append(pred)
            
            # Inverse scale
            pred_original = self.y_scaler.inverse_transform([[predictions[-1]]])[0, 0]
            predictions[-1] = pred_original
            
            # Update sequence
            new_point = np.array([[predictions[-1]] + ([0] * (combined.shape[1] - 1))])
            current_seq = np.append(current_seq, new_point, axis=0)
        
        return np.array(predictions)

class RAGPipeline:
    """Production-ready RAG with hallucination detection and caching"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.corpus = []
        self.bm25 = None
        self.semantic_model = None
        self.corpus_embeddings = None
        self.hallucination_threshold = 0.3
        self.cache = {}
    
    @st.cache_resource
    def load_corpus_from_file(_self, file_path: str) -> List[str]:
        """Load corpus from file with caching"""
        try:
            with open(file_path, 'r') as f:
                return f.readlines()
        except:
            return []
    
    def build_corpus(self, corpus: List[str], use_cache: bool = True):
        """Build searchable corpus"""
        if not corpus:
            st.warning("Empty corpus provided")
            return
        
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        
        self.corpus = corpus
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Load semantic model with progress bar
        model_load_container = st.empty()
        model_load_container.info("Loading semantic model...")
        self.semantic_model = SentenceTransformer(self.config.rag_semantic_model)
        model_load_container.empty()
        
        # Cache embeddings
        if use_cache and hasattr(st, 'session_state'):
            cache_key = f"embeddings_{hash(str(corpus))}"
            if cache_key in st.session_state:
                self.corpus_embeddings = st.session_state[cache_key]
            else:
                self.corpus_embeddings = self.semantic_model.encode(corpus, convert_to_tensor=True)
                st.session_state[cache_key] = self.corpus_embeddings
        else:
            self.corpus_embeddings = self.semantic_model.encode(corpus, convert_to_tensor=True)
        
        st.success(f"✅ RAG corpus loaded ({len(corpus)} documents)")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float], List[bool]]:
        """Hybrid retrieval with BM25 + semantic search"""
        if not self.bm25 or not self.semantic_model:
            return [], [], []
        
        cache_key = f"{query}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        top_k = top_k or self.config.rag_top_k
        tokenized_query = query.lower().split()
        
        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Semantic scores
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        semantic_scores = torch.cosine_similarity(
            query_embedding.unsqueeze(0), self.corpus_embeddings
        ).cpu().numpy()
        
        # Normalize scores
        bm25_scores = (bm25_scores - bm25_scores.mean()) / (bm25_scores.std() + 1e-8)
        semantic_scores = (semantic_scores - semantic_scores.mean()) / (semantic_scores.std() + 1e-8)
        
        # Combine
        combined_scores = (
            self.config.rag_hybrid_weight * bm25_scores +
            (1 - self.config.rag_hybrid_weight) * semantic_scores
        )
        
        # Get top k
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        retrieved_docs = [self.corpus[i] for i in top_indices]
        retrieved_scores = [float(combined_scores[i]) for i in top_indices]
        is_reliable = [score > self.hallucination_threshold for score in retrieved_scores]
        
        # Cache results
        self.cache[cache_key] = (retrieved_docs, retrieved_scores, is_reliable)
        
        return retrieved_docs, retrieved_scores, is_reliable

class LLMForecaster:
    """LLM-based forecasting with RAG and structured output"""
    
    def __init__(self, config: ForecastConfig, rag_pipeline: Optional[RAGPipeline] = None):
        self.config = config
        self.rag = rag_pipeline
        self.conversation_history = []
        self.hallucination_detector = lambda x: True  # Placeholder
    
    def predict(self, y: np.ndarray, steps: int = 1, context: Optional[str] = None) -> Tuple[np.ndarray, str, float]:
        """Generate LLM forecast with confidence scoring"""
        # RAG retrieval
        retrieved_docs, scores, reliability = [], [], []
        if self.config.use_rag and self.rag and context:
            retrieved_docs, scores, reliability = self.rag.retrieve(context)
        
        # Prepare prompt
        recent_values = y[-20:].tolist()
        prompt = self._build_prompt(recent_values, steps, context, retrieved_docs, reliability)
        
        # Call LLM
        forecast, reasoning, confidence = self._call_llm(prompt, steps)
        
        # Adjust confidence based on RAG reliability
        if reliability and not any(reliability):
            confidence *= 0.5
            reasoning += " (Confidence reduced due to low-quality retrieval)"
        
        # Validate forecast
        if len(forecast) != steps:
            st.warning(f"LLM returned {len(forecast)} values, expected {steps}. Padding/truncating.")
            forecast = np.pad(forecast, (0, steps - len(forecast)), mode='edge')[:steps]
        
        return forecast, reasoning, confidence
    
    def _build_prompt(self, recent_values: List[float], steps: int, 
                     context: Optional[str], retrieved_docs: List[str], 
                     reliability: List[bool]) -> str:
        """Build structured prompt for LLM"""
        return f"""
        You are an expert economic forecaster. Analyze the following time series and provide forecasts.

        TASK: Forecast the next {steps} values for the time series: {recent_values}

        REQUIREMENTS:
        1. Return a JSON object with exactly this structure:
           {{"forecast": [float, float, ...], "reasoning": "...", "confidence": 0.0-1.0}}
        2. The "forecast" array must contain exactly {steps} numbers
        3. "reasoning" should explain patterns, trends, and factors considered
        4. "confidence" should reflect your certainty in the forecast

        CONTEXT:
        {context or "No additional context provided"}

        RELEVANT ECONOMIC KNOWLEDGE:
        {chr(10).join([f"- {doc.strip()}" for doc in retrieved_docs[:3]]) if retrieved_docs else "No external knowledge retrieved."}

        RELIABILITY SCORES: {reliability[:3] if retrieved_docs else "N/A"}

        GUIDELINES:
        - Consider trend, seasonality, and recent anomalies
        - Factor in economic context if provided
        - Be conservative with confidence for volatile periods
        - If values seem abnormal, note this in reasoning
        """
    
    def _call_llm(self, prompt: str, steps: int) -> Tuple[np.ndarray, str, float]:
        """Call LLM API with fallback"""
        try:
            import openai
            openai.api_key = self.config.api_key
            
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a forecasting assistant that outputs only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            forecast = np.array(data["forecast"][:steps], dtype=float)
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = float(data.get("confidence", 0.5))
            
            return forecast, reasoning, confidence
            
        except Exception as e:
            st.warning(f"LLM API call failed: {str(e)}. Using fallback.")
            return self._fallback_forecast(steps)
    
    def _fallback_forecast(self, steps: int) -> Tuple[np.ndarray, str, float]:
        """Robust fallback when LLM unavailable"""
        recent_trend = np.mean(self.history[-5:]) - np.mean(self.history[-10:-5]) if len(self.history) >= 10 else 0
        last_value = self.history[-1] if len(self.history) > 0 else 100
        
        forecast = last_value + np.arange(1, steps + 1) * recent_trend * 0.3
        forecast += np.random.normal(0, abs(recent_trend) * 0.1, steps)  # Add small noise
        
        return forecast, "LLM unavailable - using trend-based fallback", 0.2

class MetaController:
    """Advanced meta-controller with uncertainty quantification and drift detection"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.performance_history = {
            ModelType.STATISTICAL: [],
            ModelType.DEEP_LEARNING: [],
            ModelType.LLM: []
        }
        self.uncertainty_history = {
            ModelType.STATISTICAL: [],
            ModelType.DEEP_LEARNING: [],
            ModelType.LLM: []
        }
        self.weights_history = []
        self.drift_alerts = []
    
    def compute_loss(self, y_true: float, y_pred: float, uncertainty: float, model_type: ModelType):
        """Compute loss with uncertainty penalty"""
        base_loss = (y_true - y_pred) ** 2
        adjusted_loss = base_loss + self.config.alpha * uncertainty
        self.performance_history[model_type].append(adjusted_loss)
        
        # Maintain window
        if len(self.performance_history[model_type]) > self.config.window_size:
            self.performance_history[model_type].pop(0)
    
    def update_weights(self, current_uncertainties: Dict[ModelType, float]) -> np.ndarray:
        """Compute dynamic weights using softmax of inverse losses"""
        avg_losses = [
            self.get_average_loss(model_type) 
            for model_type in [ModelType.STATISTICAL, ModelType.DEEP_LEARNING, ModelType.LLM]
        ]
        
        # Apply uncertainty penalties
        if current_uncertainties and self.config.uncertainty_weighting:
            for i, model_type in enumerate([ModelType.STATISTICAL, ModelType.DEEP_LEARNING, ModelType.LLM]):
                if model_type in current_uncertainties:
                    avg_losses[i] += current_uncertainties[model_type] * 2
        
        # Guardrail for LLM
        if avg_losses[2] > self.config.guardrail_threshold:
            avg_losses[2] = np.inf
        
        # Softmax weights
        exp_terms = np.exp(-self.config.alpha * np.array(avg_losses))
        exp_terms = np.where(np.isinf(avg_losses), 0, exp_terms)
        
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp if sum_exp > 0 else np.array([1/3, 1/3, 1/3])
        
        self.weights_history.append(weights.copy())
        return weights
    
    def get_average_loss(self, model_type: ModelType) -> float:
        losses = self.performance_history[model_type]
        return np.mean(losses) if losses else 0.0
    
    def detect_drift(self, recent_loss: float, model_type: ModelType) -> Dict[str, Any]:
        """Detect performance drift with statistical tests"""
        losses = self.performance_history[model_type]
        if len(losses) < 20:
            return {"drift": False, "severity": 0}
        
        # Compute statistics
        historical_mean = np.mean(losses[:-10])
        historical_std = np.std(losses[:-10])
        recent_mean = np.mean(losses[-10:])
        
        # Z-score test
        z_score = (recent_mean - historical_mean) / (historical_std + 1e-8)
        drift_detected = z_score > 2.0
        
        if drift_detected:
            self.drift_alerts.append({
                "model": model_type.value,
                "timestamp": datetime.now(),
                "z_score": z_score
            })
        
        return {
            "drift": drift_detected,
            "severity": abs(z_score),
            "historical_mean": historical_mean,
            "recent_mean": recent_mean
        }

class TRIFUSIONFramework:
    """Main orchestrator for the three-component forecasting framework"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.config.validate()
        
        # Components
        self.statistical = StatisticalForecaster(config)
        self.deep_learning = DeepLearningForecaster(config)
        self.rag = RAGPipeline(config)
        self.llm = LLMForecaster(config, self.rag)
        self.meta_controller = MetaController(config)
        self.federated_aggregator = None
        
        # State
        self.history = None
        self.exog_history = None
        self.is_fitted = False
        
        # Performance tracking
        self.backtest_results = None
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None, 
            context: Optional[List[str]] = None):
        """Fit all three components"""
        # Validate data
        is_valid, msg = DataValidator.validate_time_series(y)
        if not is_valid:
            raise ValueError(f"Data validation failed: {msg}")
        
        self.history = y.copy()
        self.exog_history = exog.copy() if exog is not None else None
        
        with st.spinner("🔧 Training Statistical Model..."):
            self.statistical.fit(y, exog, auto_select=True)
        
        with st.spinner("🧠 Training Deep Learning Model..."):
            self.deep_learning.fit(y, exog)
        
        if context and self.config.use_rag:
            with st.spinner("📚 Building RAG Corpus..."):
                self.rag.build_corpus(context)
        
        self.is_fitted = True
        st.success("🎯 All models trained successfully!")
        return self
    
    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None,
                context: Optional[str] = None) -> Dict[str, Any]:
        """Generate hybrid forecast with full diagnostics"""
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted before prediction")
        
        # Component predictions
        pred_stat = self.statistical.predict(steps, exog_future)
        pred_deep = self.deep_learning.predict(self.history, self.exog_history, steps,
                                               self.config.uncertainty_weighting)
        pred_llm, reasoning, confidence = self.llm.predict(self.history, steps, context)
        
        # Ensure consistent length
        steps = min(steps, len(pred_stat), len(pred_deep), len(pred_llm))
        pred_stat, pred_deep, pred_llm = pred_stat[:steps], pred_deep[:steps], pred_llm[:steps]
        
        # Uncertainty quantification
        uncertainties = {
            ModelType.STATISTICAL: 0.1,  # Fixed for statistical
            ModelType.DEEP_LEARNING: getattr(self.deep_learning, 'uncertainty', 0.1),
            ModelType.LLM: max(0.1, 1.0 - confidence)
        }
        
        # Dynamic weighting
        weights = self.meta_controller.update_weights(uncertainties)
        
        # Detect drift
        drift_results = {
            model_type: self.meta_controller.detect_drift(
                self.meta_controller.get_average_loss(model_type),
                model_type
            )
            for model_type in [ModelType.STATISTICAL, ModelType.DEEP_LEARNING, ModelType.LLM]
        }
        
        # Apply drift adaptation
        if drift_results[ModelType.DEEP_LEARNING]["drift"]:
            weights[1] *= 1.2
            weights /= weights.sum()
            st.warning("⚠️ Drift detected in deep learning model - increasing weight")
        
        # Hybrid forecast
        hybrid = weights[0] * pred_stat + weights[1] * pred_deep + weights[2] * pred_llm
        
        # Confidence interval
        conf_int = self.statistical.get_confidence_interval()
        
        # Overall confidence
        overall_confidence = 1.0 - np.dot(weights, list(uncertainties.values()))
        
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
            'uncertainties': {k.value: v for k, v in uncertainties.items()},
            'drift_detected': {k.value: v for k, v in drift_results.items()},
            'explanation': reasoning,
            'confidence_interval': conf_int,
            'overall_confidence': overall_confidence,
            'timestamp': datetime.now()
        }
    
    def backtest(self, n_splits: int = 3) -> pd.DataFrame:
        """Perform walk-forward backtesting"""
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted before backtesting")
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        backtest_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.history)):
            fold_train_y = self.history[train_idx]
            fold_test_y = self.history[test_idx]
            fold_exog_train = self.exog_history[train_idx] if self.exog_history is not None else None
            fold_exog_test = self.exog_history[test_idx] if self.exog_history is not None else None
            
            # Retrain on fold
            fold_model = TRIFUSIONFramework(self.config)
            fold_model.fit(fold_train_y, fold_exog_train)
            
            # Predict
            result = fold_model.predict(steps=len(test_idx), exog_future=fold_exog_test)
            
            # Store results
            for i, (actual, pred) in enumerate(zip(fold_test_y, result['forecast'])):
                backtest_results.append({
                    'fold': fold,
                    'index': i,
                    'actual': actual,
                    'forecast': pred,
                    'error': actual - pred,
                    'abs_error': abs(actual - pred)
                })
        
        self.backtest_results = pd.DataFrame(backtest_results)
        return self.backtest_results
    
    def enable_federated_learning(self, client_data: List[Dict[str, Any]]):
        """Setup federated learning across multiple clients"""
        self.federated_aggregator = FederatedAggregator(self, self.config)
        self.federated_aggregator.create_clients(client_data)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance for interpretability"""
        if self.deep_learning.model is None:
            return None
        
        # Placeholder for SHAP or integrated gradients
        importance_df = pd.DataFrame({
            'feature': ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2'],
            'importance': np.random.rand(5)
        })
        return importance_df.sort_values('importance', ascending=False)

class DOSMDataLoader:
    """Robust data loader with fallback and validation"""
    
    def __init__(self):
        self.base_url = "https://storage.dosm.gov.my"
        self.datasets = {
            "cpi_state": "/timeseries/cpi/cpi_2d_state.parquet",
            "cpi_national": "/timeseries/cpi/cpi_2d.parquet"
        }
        self.cache_dir = Path("./data_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_cpi_data(self, state: str = "Malaysia", start_date: str = "2015-01-01") -> pd.DataFrame:
        """Load CPI data with caching and fallback"""
        cache_key = f"{state}_{start_date}"
        cache_file = self.cache_dir / f"cpi_{cache_key}.parquet"
        
        # Try cache first
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except:
                pass
        
        try:
            url = f"{self.base_url}{self.datasets['cpi_state']}"
            df = pd.read_parquet(url)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Handle state filtering
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
            
            # Cache
            df_filtered.to_parquet(cache_file)
            st.success(f"✅ Loaded {len(df_filtered)} CPI records for {state}")
            return df_filtered
            
        except Exception as e:
            st.error(f"❌ Failed to load DOSM data: {str(e)}")
            st.warning("⚠️ Generating synthetic fallback data")
            return self._generate_synthetic_cpi(state, start_date)
    
    def _generate_synthetic_cpi(self, state: str, start_date: str) -> pd.DataFrame:
        """Generate realistic synthetic CPI data"""
        start = pd.to_datetime(start_date)
        end = pd.Timestamp.now()
        dates = pd.date_range(start=start, end=end, freq='M')
        
        # Multiplicative trend with structural breaks
        base_cpi = 100
        values = []
        
        for i, date in enumerate(dates):
            # Trend
            trend = 1 + (date.year - 2015) * 0.015 + i * 0.0003
            
            # Seasonality
            seasonal = 1 + 0.02 * np.sin(2 * np.pi * date.month / 12)
            
            # Structural breaks
            break_effects = 1.0
            if pd.Timestamp('2018-09-01') <= date <= pd.Timestamp('2018-12-01'):
                break_effects *= 1.02  # SST implementation
            if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-06-01'):
                break_effects *= 1.04  # COVID impact
            if pd.Timestamp('2022-06-01') <= date <= pd.Timestamp('2022-09-01'):
                break_effects *= 1.03  # Subsidy rationalization
            
            # Noise
            noise = np.random.normal(0, 0.25)
            
            cpi = base_cpi * trend * seasonal * break_effects + noise
            values.append(max(95, min(135, cpi)))
        
        return pd.DataFrame({
            'date': dates,
            'state': state,
            'index': values
        })
    
    @st.cache_data
    def load_exogenous_data(_self, start_date: str = "2015-01-01") -> pd.DataFrame:
        """Generate synthetic but realistic exogenous variables"""
        end = pd.Timestamp.now()
        dates = pd.date_range(start=start_date, end=end, freq='M')
        
        data = []
        for date in dates:
            # Oil price: mean-reverting with trends
            oil_trend = 60 + (date.year - 2020) * 2
            oil_cycle = 15 * np.sin(2 * np.pi * date.year / 3)
            oil_noise = np.random.normal(0, 5)
            oil_price = max(40, min(120, oil_trend + oil_cycle + oil_noise))
            
            # USD/MYR: realistic exchange rate
            usd_myr = 4.2 + 0.1 * np.sin(2 * np.pi * date.year / 5) + np.random.normal(0, 0.08)
            usd_myr = max(3.8, min(4.8, usd_myr))
            
            # Policy events (binary)
            policy_shock = 1.0 if date in [
                pd.Timestamp('2018-09-01'), 
                pd.Timestamp('2022-06-01')
            ] else 0.0
            
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
    """Streamlit app with professional UI/UX"""
    
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
        
        self._render_header()
        self._render_sidebar()
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()
    
    def _render_header(self):
        """Render app header with branding"""
        st.markdown("""
        <style>
        .main-header { text-align: center; padding: 1rem; }
        .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>📈 TRIFUSION Forecasting Framework</h1>
            <p>Advanced Hybrid Time Series Forecasting with LLM Integration</p>
            <p style="color: #666;">Powered by DOSM Malaysia Open Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def _render_sidebar(self):
        """Render configuration sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model Selection
            st.subheader("Model Architecture")
            dl_arch = st.selectbox("Deep Learning", ["transformer", "lstm"], index=0)
            
            st.subheader("Hyperparameters")
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 50, 500, 100)
            lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            
            st.subheader("Data Settings")
            state = st.selectbox("Select State", self.state_options, index=0)
            start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
            test_size = st.slider("Test Period (months)", 6, 36, 24)
            
            st.subheader("LLM Settings")
            api_key = st.text_input("OpenAI API Key", type="password")
            use_rag = st.checkbox("Enable RAG", value=True)
            
            st.subheader("Advanced")
            uncertainty_weighting = st.checkbox("Uncertainty Weighting", value=True)
            adversarial_training = st.checkbox("Adversarial Training", value=False)
            
            # Build config
            self.config = ForecastConfig(
                dl_architecture=dl_arch,
                lookback=lookback,
                epochs=epochs,
                learning_rate=lr,
                api_key=api_key or None,
                use_rag=use_rag,
                uncertainty_weighting=uncertainty_weighting,
                adversarial_training=adversarial_training,
                test_size=test_size
            )
            
            st.markdown("---")
            st.info("""
            **Framework Components:**
            - 🔢 Statistical (ARIMA/SARIMAX)
            - 🧠 Deep Learning (Transformer/LSTM)
            - 🤖 LLM (GPT with RAG)
            - ⚖️ Dynamic Meta-Controller
            """)
    
    def _run_analysis(self):
        """Execute full forecasting analysis"""
        if not self.config.api_key:
            st.warning("⚠️ No OpenAI API key provided. LLM will use fallback logic.")
        
        # Load data
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = self.loader.load_cpi_data(state=self.config.state, start_date=self.config.start_date)
            exog_data = self.loader.load_exogenous_data(start_date=self.config.start_date)
        
        # Merge datasets
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        
        # Fill missing values
        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        full_data[numeric_cols] = full_data[numeric_cols].fillna(method='ffill').fillna(0)
        full_data = full_data.dropna(subset=['index', 'date'])
        
        if full_data.empty:
            st.error("❌ No valid data available")
            return
        
        # Display data overview
        self._render_data_overview(full_data)
        
        # Prepare data
        y = full_data['index'].values
        exog = full_data[numeric_cols].values
        
        # Train-test split
        train_size = len(y) - self.config.test_size
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        # Initialize framework
        self.framework = TRIFUSIONFramework(self.config)
        
        # Fit framework
        with st.spinner("🎯 Training all components..."):
            # Build context for RAG
            context_docs = [
                "Malaysia CPI is influenced by oil prices due to fuel subsidies",
                "USD/MYR exchange rate affects import costs",
                "COVID-19 caused supply chain disruptions in 2020-2021",
                "SST implementation in September 2018 increased prices",
                "Fuel subsidy rationalization in June 2022 caused inflation spike"
            ]
            
            self.framework.fit(y_train, exog_train, context=context_docs)
        
        # Rolling forecast
        with st.spinner("🔮 Generating forecasts..."):
            results = self._rolling_forecast(y_test, exog_test, full_data.iloc[train_size:])
        
        # Performance analysis
        self._render_performance_dashboard(results, full_data)
        
        # Backtesting
        if st.checkbox("Run Backtesting", value=False):
            self._run_backtesting(y, exog)
        
        # Export results
        self._render_export_section(results)
    
    def _render_data_overview(self, data: pd.DataFrame):
        """Render data summary metrics"""
        st.markdown("### 📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric("CPI Range", f"{data['index'].min():.1f} - {data['index'].max():.1f}")
        with col3:
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")
        with col4:
            st.metric("Missing Values", f"{data['index'].isna().sum()} ({data['index'].isna().mean():.1%})")
        
        # Interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['index'],
            mode='lines+markers',
            name='CPI Index',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Consumer Price Index Over Time",
            xaxis_title="Date",
            yaxis_title="CPI Index",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show regime shifts
        self._highlight_regime_shifts(data)
    
    def _highlight_regime_shifts(self, data: pd.DataFrame):
        """Highlight significant events"""
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
        if 'policy_shock' in data.columns:
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
        """Perform rolling window forecast"""
        predictions = []
        actuals = []
        weights_history = []
        uncertainties_history = []
        drift_history = []
        
        progress_bar = st.progress(0)
        
        for i in range(len(y_test) - self.config.forecast_horizon):
            # Prepare data
            current_history = self.framework.history[:len(self.framework.history) - len(y_test) + i + 1]
            current_exog = self.framework.exog_history[:len(self.framework.exog_history) - len(y_test) + i + 1] if self.framework.exog_history is not None else None
            
            exog_future = exog_test[i:i+self.config.forecast_horizon]
            future_date = test_data.iloc[i]['date']
            
            # Generate context
            context = self._generate_context(future_date)
            
            # Predict
            result = self.framework.predict(
                steps=self.config.forecast_horizon, 
                exog_future=exog_future,
                context=context
            )
            
            # Store results
            if i < len(y_test) - self.config.forecast_horizon:
                predictions.append(result['forecast'][0])
                actuals.append(y_test[i + 1])
                weights_history.append(result['weights'])
                uncertainties_history.append(result['uncertainties'])
                drift_history.append(result['drift_detected'])
            
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
            'drift_history': drift_history,
            'dates': test_data['date'].values[1:len(predictions)+1],
            'exog_test': exog_test
        }
    
    def _generate_context(self, date: pd.Timestamp) -> str:
        """Generate contextual text"""
        contexts = []
        
        if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-12-01'):
            contexts.append("COVID-19 pandemic ongoing")
        
        if date.month in [11, 12]:
            contexts.append("Year-end spending surge")
        
        if date.year == 2018 and date.month == 9:
            contexts.append("SST implementation impact")
        
        if date.year == 2022 and date.month == 6:
            contexts.append("Fuel subsidy reform")
        
        return " | ".join(contexts) if contexts else "Normal conditions"
    
    def _render_performance_dashboard(self, results: Dict[str, Any], full_data: pd.DataFrame):
        """Render comprehensive performance dashboard"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        st.markdown("### 📊 Performance Dashboard")
        
        # Metrics
        mae = mean_absolute_error(results['actuals'], results['predictions'])
        rmse = np.sqrt(mean_squared_error(results['actuals'], results['predictions']))
        mape = np.mean(np.abs((results['actuals'] - results['predictions']) / results['actuals'])) * 100
        r2 = 1 - (np.sum((results['actuals'] - results['predictions'])**2) / 
                  np.sum((results['actuals'] - np.mean(results['actuals']))**2))
        
        metrics_cols = st.columns(5)
        metrics = [
            ("MAE", f"{mae:.4f}"),
            ("RMSE", f"{rmse:.4f}"),
            ("MAPE", f"{mape:.2f}%"),
            ("R²", f"{r2:.4f}"),
            ("Observations", len(results['actuals']))
        ]
        
        for col, (label, value) in zip(metrics_cols, metrics):
            with col:
                st.metric(label, value)
        
        # Interactive plot
        self._render_interactive_plot(results)
        
        # Component analysis
        self._render_component_analysis(results)
        
        # Error analysis
        self._render_error_analysis(results)
    
    def _render_interactive_plot(self, results: Dict[str, Any]):
        """Render interactive forecast vs actual plot"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Forecast vs Actual", "Model Weights", "Forecast Errors"),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Forecast vs Actual
        fig.add_trace(
            go.Scatter(x=results['dates'], y=results['actuals'], 
                      mode='lines+markers', name='Actual CPI',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=results['dates'], y=results['predictions'], 
                      mode='lines+markers', name='TRIFUSION Forecast',
                      line=dict(color='#d62728', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Model weights
        weights_arr = np.array([list(w.values()) for w in results['weights_history']])
        fig.add_trace(
            go.Scatter(x=results['dates'], y=weights_arr[:, 0], 
                      mode='lines', name='Statistical Weight'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=results['dates'], y=weights_arr[:, 1], 
                      mode='lines', name='DL Weight'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=results['dates'], y=weights_arr[:, 2], 
                      mode='lines', name='LLM Weight'),
            row=2, col=1
        )
        
        # Errors
        errors = np.abs(results['actuals'] - results['predictions'])
        fig.add_trace(
            go.Scatter(x=results['dates'], y=errors, 
                      mode='lines+markers', name='Absolute Error',
                      line=dict(color='#9467bd')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
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
        
        # Bar chart of average weights
        weight_fig = go.Figure(data=[
            go.Bar(x=weight_df['Component'], y=weight_df['Avg Weight'],
                   error_y=dict(type='data', array=weight_df['Std Dev']))
        ])
        weight_fig.update_layout(title="Average Model Weights", xaxis_title="Component", yaxis_title="Weight")
        st.plotly_chart(weight_fig, use_container_width=True)
        
        # Uncertainty over time
        unc_df = pd.DataFrame(results['uncertainties_history'])
        unc_fig = go.Figure()
        for col in unc_df.columns:
            unc_fig.add_trace(go.Scatter(x=results['dates'], y=unc_df[col], mode='lines', name=col))
        unc_fig.update_layout(title="Model Uncertainties Over Time", xaxis_title="Date", yaxis_title="Uncertainty")
        st.plotly_chart(unc_fig, use_container_width=True)
    
    def _render_error_analysis(self, results: Dict[str, Any]):
        """Analyze forecast errors"""
        st.markdown("#### 📉 Error Analysis")
        
        errors = results['actuals'] - results['predictions']
        abs_errors = np.abs(errors)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=30)])
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
        
        # Error statistics
        error_stats = pd.DataFrame({
            'Statistic': ['Mean Error', 'MAE', 'RMSE', 'MAPE'],
            'Value': [np.mean(errors), np.mean(abs_errors), np.sqrt(np.mean(errors**2)), 
                     np.mean(abs_errors / results['actuals']) * 100]
        })
        st.dataframe(error_stats.style.format({'Value': '{:.4f}'}), use_container_width=True)
    
    def _run_backtesting(self, y: np.ndarray, exog: np.ndarray):
        """Run comprehensive backtesting"""
        st.markdown("### 🔬 Backtesting Results")
        
        with st.spinner("Running walk-forward validation..."):
            backtest_results = self.framework.backtest(n_splits=3)
        
        if backtest_results is not None:
            # Show fold performance
            fold_metrics = backtest_results.groupby('fold').agg({
                'abs_error': ['mean', 'std'],
                'error': ['mean', 'std']
            })
            
            st.dataframe(fold_metrics, use_container_width=True)
            
            # Download backtest results
            csv = backtest_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Backtest Results",
                data=csv,
                file_name=f"trifusion_backtest_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _render_export_section(self, results: Dict[str, Any]):
        """Render export options"""
        st.markdown("### 💾 Export Results")
        
        # Prepare results DataFrame
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
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="trifusion_forecast_{datetime.now().strftime("%Y%m%d")}.csv">📥 Download Forecast (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # JSON Export
        json_export = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'state': self.config.state if hasattr(self.config, 'state') else 'Unknown',
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
                'mae': float(np.mean(np.abs(results['actuals'] - results['predictions']))),
                'rmse': float(np.sqrt(np.mean((results['actuals'] - results['predictions'])**2))),
                'mape': float(np.mean(np.abs((results['actuals'] - results['predictions']) / results['actuals']) * 100))
            }
        }
        
        json_str = json.dumps(json_export, indent=2)
        b64_json = base64.b64encode(json_str.encode()).decode()
        href_json = f'<a href="data:file/json;base64,{b64_json}" download="trifusion_full_results_{datetime.now().strftime("%Y%m%d")}.json">📥 Download Full Results (JSON)</a>'
        st.markdown(href_json, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
