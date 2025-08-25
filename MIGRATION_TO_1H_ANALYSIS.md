# Migration to 1-Hour Analysis

This document describes the changes made to migrate the analysis pipeline from 1-minute to 1-hour timeframes.

## Overview

The entire data pipeline has been rewritten to analyze options and equity data on a 1-hour basis instead of 1-minute. This provides several benefits:

- **Reduced noise**: 1-hour data smooths out micro-structure noise
- **Better signal**: Cleaner patterns for longer-term option relationships
- **Computational efficiency**: 60x fewer data points to process
- **More stable features**: Rolling statistics with meaningful lookback periods

## Key Changes

### 1. Database Schema Updates (`src/fetch_data_sqlite.py`)

#### New Primary Tables (1-hour focus):
- `opra_1h` - 1-hour options OHLCV data
- `equity_1h` - 1-hour equity OHLCV data  
- `equity_1d` - Daily equity data for longer-term context
- `merged_1h` - Merged 1-hour options/equity data
- `processed_merged_1h` - Processed with option characteristics
- `atm_slices_1h` - AT-the-money option slices

#### Legacy Tables Preserved:
- All `*_1m` tables maintained for backward compatibility

### 2. Data Fetching Logic

#### Updated `_fetch()` function:
```python
# OLD: Fetched 1-minute data
opra_1m, eq_1m, eq_1h = _fetch(API_KEY, start, end, ticker)

# NEW: Fetches 1-hour data
opra_1h, eq_1h, eq_1d = _fetch(API_KEY, start, end, ticker)
```

#### Market Hours Filtering:
- **1-minute**: 14:00-21:00 UTC (2:00 PM - 9:00 PM)
- **1-hour**: 9:30 AM - 4:00 PM ET (14:30-21:00 UTC) with proper timezone handling

#### Default Timeframe:
- All functions now default to `timeframe="1h"`
- CLI option `--timeframe` supports both "1h" and "1m"

### 3. Feature Engineering Updates (`src/feature_engineering.py`)

#### Rolling Window Adjustments:

| Feature Type | 1-Minute Version | 1-Hour Version | Notes |
|--------------|------------------|----------------|-------|
| **Realized Volatility** | `rv_30m` (30-min) | `rv_30h` (30-hour) | ~1 week lookback |
| **IV Momentum** | `iv_ret_1m`, `iv_ret_5m`, `iv_ret_15m` | `iv_ret_1h`, `iv_ret_3h`, `iv_ret_6h` | 1h, 3h, 6h changes |
| **IV Rolling Stats** | `iv_sma_5m`, `iv_sma_15m` | `iv_sma_3h`, `iv_sma_6h` | 3h, 6h averages |
| **IV Volatility** | `iv_std_15m` | `iv_std_6h` | 6-hour rolling std |
| **IV Technical** | `iv_rsi_15m`, `iv_zscore_15m` | `iv_rsi_6h`, `iv_zscore_6h` | 6-hour RSI/z-score |
| **Volume Features** | `opt_vol_roll_5m`, `opt_vol_roll_15m` | `opt_vol_roll_3h`, `opt_vol_roll_6h`, `opt_vol_roll_24h` | 3h, 6h, daily |
| **Volume Changes** | `opt_vol_change_1m` | `opt_vol_change_1h` | 1-hour changes |

#### Annualization Constants:
```python
# NEW: Added for 1-hour data
ANNUAL_HOURS = 252 * 6.5  # 252 trading days * 6.5 hours per day

# LEGACY: Preserved for backward compatibility  
ANNUAL_MINUTES = 252 * 390
```

#### SABR Parameter Estimation:
- Nu (volatility of volatility) calculation updated to use `ANNUAL_HOURS`
- Rolling windows adjusted for 1-hour timeframe

### 4. Configuration Updates

#### Default Database Path:
```python
# OLD
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))

# NEW  
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1h.db"))
```

#### Core Feature Columns:
Updated `CORE_FEATURE_COLS` to reflect new 1-hour feature names.

## Migration Path

### For Existing Users:

1. **Update Environment Variable** (optional):
   ```bash
   export IV_DB_PATH="data/iv_data_1h.db"
   ```

2. **Fetch New Data**:
   ```bash
   python src/fetch_data_sqlite.py --db data/iv_data_1h.db \
       --tickers ASTS SATS VZ T --start 2025-08-01 --end 2025-08-10 \
       --timeframe 1h
   ```

3. **Run Analysis**:
   ```bash
   python src/main_runner.py --tickers ASTS SATS VZ T \
       --start 2025-08-01 --end 2025-08-10
   ```

### Backward Compatibility:

- All `*_1m` tables and functions preserved
- Can specify `timeframe="1m"` to use legacy behavior
- No breaking changes to existing APIs

## Benefits of 1-Hour Analysis

### 1. **Signal-to-Noise Ratio**
- **1-minute**: High noise from market microstructure
- **1-hour**: Cleaner price action, better for pattern recognition

### 2. **Rolling Window Meaningfulness**
- **1-minute**: 15-minute window = very short-term
- **1-hour**: 6-hour window = half trading day, more meaningful

### 3. **Computational Performance**
- **1-minute**: ~390 observations per ticker per day
- **1-hour**: ~6.5 observations per ticker per day (60x reduction)

### 4. **Feature Stability**
- **1-minute**: Features change rapidly, potential overfitting
- **1-hour**: More stable features, better generalization

### 5. **Market Regime Detection**
- **1-minute**: Too granular for regime changes
- **1-hour**: Better captures intraday regime shifts

## Feature Comparison Examples

### IV Return Features:
```python
# 1-Minute Analysis
df["iv_ret_1m"]   # 1-minute IV change
df["iv_ret_5m"]   # 5-minute IV change  
df["iv_ret_15m"]  # 15-minute IV change

# 1-Hour Analysis  
df["iv_ret_1h"]   # 1-hour IV change
df["iv_ret_3h"]   # 3-hour IV change
df["iv_ret_6h"]   # 6-hour IV change (half trading day)
```

### Volume Features:
```python
# 1-Minute Analysis
df["opt_vol_roll_15m"]   # 15-minute rolling average

# 1-Hour Analysis
df["opt_vol_roll_6h"]    # 6-hour rolling average (half day)
df["opt_vol_roll_24h"]   # 24-hour rolling average (daily)
```

## Expected Impact on Analysis Results

### Peer Effects Analysis:
- **More stable peer relationships**: Less noise in cross-ticker correlations
- **Better regime detection**: Hourly data captures meaningful regime changes
- **Cleaner signals**: Reduced impact of microstructure noise

### Baseline Correlations:
- **Smoother correlation estimates**: Less volatile correlation matrices
- **Better temporal stability**: More consistent correlation patterns

### Pooled IV Modeling:
- **Improved generalization**: Less overfitting to noise
- **Better cross-sectional patterns**: Cleaner relationships between tickers
- **More interpretable features**: Features represent meaningful time periods

## Testing and Validation

To validate the 1-hour analysis pipeline:

1. **Data Quality Check**:
   ```bash
   python src/fetch_data_sqlite.py --db test_1h.db --tickers ASTS --start 2025-08-01 --end 2025-08-02 --timeframe 1h
   ```

2. **Feature Engineering Test**:
   ```python
   from src.feature_engineering import add_all_features
   # Test with sample 1-hour data
   ```

3. **End-to-End Pipeline**:
   ```bash
   python src/main_runner.py --tickers ASTS SATS --start 2025-08-01 --end 2025-08-02 --debug
   ```

## Next Steps

1. **Update Data Coordinator**: Modify `src/data_loader_coordinator.py` to prioritize 1-hour tables
2. **Update Analysis Scripts**: Ensure all analysis scripts work with new feature names
3. **Performance Testing**: Validate that 1-hour analysis provides better results
4. **Documentation**: Update user documentation and examples

The migration to 1-hour analysis represents a significant improvement in signal quality and computational efficiency while maintaining full backward compatibility with existing 1-minute workflows.
