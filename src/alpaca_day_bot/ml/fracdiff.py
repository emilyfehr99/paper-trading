import numpy as np
import pandas as pd

def get_weights(d, size):
    """
    Generates weights for fractional differentiation.
    d: the degree of differentiation (usually between 0.3 and 0.7 for finance)
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * ((d - k + 1) / k)
        w.append(w_k)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def fractional_diff(series, d, threshold=1e-5):
    """
    Applies fractional differentiation to a time series.
    series: pandas Series
    d: degree of differentiation
    threshold: weight threshold for window size
    """
    # 1. Generate weights
    w = get_weights(d, size=len(series))
    
    # 2. Determine window size based on threshold
    # (We only keep weights that are significant)
    w_abs = np.abs(w)
    skip = np.searchsorted(w_abs.flatten(), threshold)
    w_final = w[skip:]
    
    # 3. Apply weights to rolling window
    series_values = series.values.reshape(-1, 1)
    diff_values = []
    
    window_size = len(w_final)
    for i in range(window_size, len(series_values) + 1):
        window = series_values[i - window_size:i]
        diff_values.append(np.dot(w_final.T, window)[0, 0])
    
    # Pad with NaNs to match original length
    result = [np.nan] * (window_size - 1) + diff_values
    return pd.Series(result, index=series.index)

def apply_frac_diff_to_df(df, column='close', d=0.4):
    """
    Utility to add a frac_diff column to a dataframe.
    """
    if df is None or df.empty:
        return df
    df[f'frac_diff_{d}'] = fractional_diff(df[column], d)
    return df
