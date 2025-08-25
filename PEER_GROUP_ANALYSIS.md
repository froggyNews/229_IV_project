# Peer Group Analysis Module

This document describes the comprehensive peer group analysis functionality that extends the main runner pipeline to provide detailed analysis of relationships within and between defined peer groups.

## Overview

The peer group analysis module provides:

1. **Intra-Group Analysis**: Correlations and peer effects within each defined group
2. **Inter-Group Analysis**: Relationships and effects between different peer groups  
3. **Time-Aligned Analysis**: Synchronized analysis across time periods
4. **Comprehensive Reporting**: Organized results with statistical summaries

## Key Features

### 1. Multi-Group Definition
- Define custom peer groups based on business logic
- Support for overlapping groups
- Automatic handling of group validation

### 2. Comprehensive Correlation Analysis
- Within-group IV level correlations
- Within-group IV return correlations
- Cross-group correlation matrices
- Statistical significance testing (future enhancement)

### 3. Peer Effects Modeling
- Intra-group peer effects using XGBoost
- Inter-group peer effects analysis
- Feature importance ranking
- Model performance metrics

### 4. Results Organization
- Hierarchical result structure
- JSON and human-readable outputs
- Correlation matrices saved separately
- Statistical test results (future enhancement)

## Usage

### 1. Standalone Peer Group Analysis

```python
from src.peer_group_analyzer import PeerGroupAnalyzer, PeerGroupConfig

# Define peer groups
groups = {
    "satellite": ["ASTS", "SATS"],
    "telecom": ["VZ", "T"],
    "mixed": ["ASTS", "VZ", "T", "SATS"]
}

# Create configuration
config = PeerGroupConfig(
    groups=groups,
    start="2025-08-02", 
    end="2025-08-06",
    target_kinds=["iv_ret", "iv"],
    forward_steps=15,
    output_dir=Path("outputs/peer_analysis")
)

# Run analysis
analyzer = PeerGroupAnalyzer(config)
results = analyzer.run_full_analysis()
```

### 2. Integrated with Main Runner

```bash
# Enable peer group analysis in main runner
python src/main_runner.py \
    --enable-peer-group-analysis \
    --peer-groups "satellite:ASTS,SATS" "telecom:VZ,T" \
    --tickers ASTS SATS VZ T \
    --start 2025-08-02 --end 2025-08-06
```

### 3. Programmatic Integration

```python
from src.main_runner import RunConfig, run_pipeline

config = RunConfig(
    tickers=["ASTS", "VZ", "T", "SATS"],
    enable_peer_group_analysis=True,
    peer_groups={
        "satellite": ["ASTS", "SATS"],
        "telecom": ["VZ", "T"]
    }
)

results = run_pipeline(config)
peer_group_results = results["peer_group_results"]
```

## Configuration Options

### PeerGroupConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | Dict[str, List[str]] | {} | Group definitions (name -> tickers) |
| `start` | str | "2025-08-02" | Start date for analysis |
| `end` | str | "2025-08-06" | End date for analysis |
| `target_kinds` | List[str] | ["iv_ret", "iv"] | Types of targets to analyze |
| `forward_steps` | int | 15 | Forward prediction steps |
| `test_frac` | float | 0.2 | Test set fraction |
| `tolerance` | str | "2s" | Time alignment tolerance |
| `r` | float | 0.045 | Risk-free rate |
| `output_dir` | Path | "outputs/peer_groups" | Output directory |
| `save_detailed_results` | bool | True | Save detailed analysis files |
| `debug` | bool | False | Enable debug mode |

### Main Runner Integration

Add these parameters to `RunConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_peer_group_analysis` | bool | False | Enable peer group analysis |
| `peer_groups` | Dict[str, List[str]] | None | Group definitions |

## Results Structure

### 1. Intra-Group Correlations

```json
{
  "intra_correlations": {
    "satellite": {
      "tickers": ["ASTS", "SATS"],
      "n_tickers": 2,
      "iv_correlations": {
        "matrix": {"ASTS": {"ASTS": 1.0, "SATS": 0.75}, ...},
        "mean": 0.75,
        "std": 0.0,
        "min": 0.75,
        "max": 0.75
      },
      "iv_return_correlations": {
        "matrix": {...},
        "mean": 0.65,
        "std": 0.0,
        "min": 0.65,
        "max": 0.65
      }
    }
  }
}
```

### 2. Inter-Group Correlations

```json
{
  "inter_correlations": {
    "satellite_vs_telecom": {
      "group1": "satellite",
      "group2": "telecom", 
      "group1_tickers": ["ASTS", "SATS"],
      "group2_tickers": ["VZ", "T"],
      "iv_cross_correlations": {
        "values": [0.3, 0.4, 0.2, 0.5],
        "mean": 0.35,
        "std": 0.13,
        "min": 0.2,
        "max": 0.5
      }
    }
  }
}
```

### 3. Peer Effects Analysis

```json
{
  "intra_peer_effects": {
    "satellite": {
      "tickers": ["ASTS", "SATS"],
      "results": {
        "iv_ret": {
          "target_count": 2,
          "successful_targets": 2,
          "avg_r2": 0.45,
          "avg_rmse": 0.023,
          "peer_effect_summary": {
            "SATS": {"mean_effect": 0.15, "std_effect": 0.02, "appearances": 1},
            "ASTS": {"mean_effect": 0.12, "std_effect": 0.01, "appearances": 1}
          }
        }
      }
    }
  }
}
```

## Output Directory Structure

```
outputs/
└── peer_groups/
    └── YYYYMMDD_HHMMSS/
        ├── peer_group_analysis.json      # Main results file
        ├── analysis_summary.txt          # Human-readable summary
        ├── correlations/
        │   ├── intra_satellite_correlations.json
        │   ├── intra_telecom_correlations.json
        │   └── inter_group_correlations.json
        ├── peer_effects/
        │   ├── intra_satellite_peer_effects.json
        │   ├── intra_telecom_peer_effects.json
        │   └── inter_group_peer_effects.json
        └── statistical_tests/
            └── test_results.json
```

## Example Use Cases

### 1. Sector Analysis
```python
# Analyze relationships within and between sectors
groups = {
    "tech": ["AAPL", "MSFT", "GOOGL"],
    "finance": ["JPM", "BAC", "WFC"],
    "telecom": ["VZ", "T"]
}
```

### 2. Market Cap Analysis
```python
# Analyze by market capitalization tiers
groups = {
    "large_cap": ["AAPL", "MSFT", "GOOGL"],
    "mid_cap": ["AMD", "NVDA"],
    "small_cap": ["ASTS", "SATS"]
}
```

### 3. Geographic Analysis
```python
# Analyze by geographic exposure
groups = {
    "domestic": ["VZ", "T"],
    "global": ["ASTS", "SATS"],
    "mixed": ["VZ", "ASTS"]
}
```

## Integration with Existing Pipeline

The peer group analysis seamlessly integrates with the existing pipeline:

1. **Data Loading**: Uses the same core data loading infrastructure
2. **Feature Engineering**: Leverages existing feature engineering functions
3. **Model Training**: Builds on the peer effects modeling framework
4. **Results Storage**: Extends the current results organization system

## Command Line Examples

### Basic Analysis
```bash
python src/main_runner.py \
    --enable-peer-group-analysis \
    --tickers ASTS SATS VZ T
```

### Custom Groups
```bash
python src/main_runner.py \
    --enable-peer-group-analysis \
    --peer-groups "satellite:ASTS,SATS" "telecom:VZ,T" "mixed:ASTS,VZ" \
    --tickers ASTS SATS VZ T \
    --start 2025-08-02 --end 2025-08-06 \
    --debug
```

### Full Pipeline with Peer Groups
```bash
python src/main_runner.py \
    --enable-peer-group-analysis \
    --peer-groups "satellite:ASTS,SATS" "telecom:VZ,T" \
    --tickers ASTS SATS VZ T \
    --peer-targets ASTS VZ T SATS \
    --peer-target-kinds iv_ret iv \
    --forward-steps 15 \
    --test-frac 0.2 \
    --start 2025-08-02 --end 2025-08-06 \
    --debug
```

## Future Enhancements

1. **Statistical Significance Testing**
   - Correlation significance tests
   - Peer effect significance tests
   - Group difference testing

2. **Visualization**
   - Correlation heatmaps
   - Peer effect network graphs
   - Time series of group relationships

3. **Advanced Analytics**
   - Granger causality testing
   - Regime change detection
   - Rolling window analysis

4. **Performance Optimization**
   - Parallel processing of groups
   - Caching of intermediate results
   - Memory-efficient large group handling

## Troubleshooting

### Common Issues

1. **Insufficient Data**: Ensure each group has at least 2 tickers with valid data
2. **Memory Usage**: Large numbers of groups may require memory optimization
3. **Time Alignment**: Check tolerance settings if correlations seem low
4. **Missing Dependencies**: Ensure all required packages are installed

### Debug Mode

Enable debug mode to get detailed logging:
```python
config = PeerGroupConfig(debug=True)
```

Or via command line:
```bash
python src/main_runner.py --enable-peer-group-analysis --debug
```

## Performance Considerations

- **Group Size**: Larger groups require more computation time
- **Number of Groups**: Analysis time scales with number of group pairs
- **Time Range**: Longer date ranges increase data processing time
- **Feature Count**: More target kinds increase model training time

## Dependencies

The peer group analysis module builds on:
- `baseline_correlation.py`: For correlation computations
- `train_peer_effects.py`: For peer effects modeling
- `feature_engineering.py`: For data preparation
- `data_loader_coordinator.py`: For data loading
- Standard scientific Python stack (pandas, numpy, scikit-learn, xgboost)
