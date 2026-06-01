import numpy as np
import pandas as pd
import logging

log = logging.getLogger("alpaca_day_bot.risk.garch")

# Try to import the 'arch' package; if missing, fall back to a simpler estimator
try:
    from arch import arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except Exception:
    arch_model = None
    _ARCH_AVAILABLE = False
    log.warning("Optional dependency 'arch' not available; GARCH forecasting will use a rolling-std fallback.")

def predict_garch_volatility(prices: np.ndarray) -> float:
    """
    Fits a GARCH(1,1) model on the log returns of trailing prices to predict next-period variance.
    Returns predicted volatility proxy in decimals (e.g. 0.03 = 3% next bar expectation).
    Falls back to standard deviation on convergence errors or small sample size.
    """
    if len(prices) < 20:
        return 0.0025  # Default 0.25% floor proxy
        
    try:
        # Calculate log returns
        returns = np.diff(np.log(prices))
        returns = returns[np.isfinite(returns)]
        
        # Scale standard returns to percentage returns for stability
        scaled_returns = returns * 100.0
        
        # Add microscopic random jitter (1e-6) to prevent singular covariance matrix solver failures in flat/zero-volume price regimes
        np.random.seed(42) # Ensure deterministic optimization
        scaled_returns = scaled_returns + np.random.normal(0.0, 1e-6, size=len(scaled_returns))
        
        if len(scaled_returns) < 15:
            return 0.0025
            
        # Fit GARCH(1,1) with normal distribution and quiet solver
        model = arch_model(scaled_returns, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        
        # Forecast 1 step ahead
        forecasts = res.forecast(horizon=1)
        pred_variance = float(forecasts.variance.iloc[-1, 0])
        
        # Ensure variance is positive and realistic
        if pred_variance <= 0:
            raise ValueError("Non-positive variance forecast")
            
        # Scale standard deviation back to decimal space
        pred_std = np.sqrt(pred_variance) / 100.0
        
        # Apply safety bounds to predicted standard deviation scaled for 1-minute bars (0.05% to 1.00%)
        pred_std = max(0.0005, min(0.0100, pred_std))
        
        return float(pred_std)
    except Exception as e:
        log.debug(f"GARCH model fit error, falling back to rolling return standard deviation: {e}")
        try:
            returns = np.diff(np.log(prices))
            returns = returns[np.isfinite(returns)]
            std_val = float(np.std(returns))
            return max(0.0005, min(0.0100, std_val))
        except Exception:
            return 0.0025

