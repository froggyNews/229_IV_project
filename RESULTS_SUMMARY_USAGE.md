# Results Summary Viewer Usage Guide

This guide shows how to use the clean, high-level summary tools for peer group analysis results.

## Quick Start

### 1. View Latest Results Summary
```bash
python src/results_summary_viewer.py
```

### 2. Command-Line Interface
```bash
# Basic summary
python scripts/view_results.py

# Compact view
python scripts/view_results.py --compact

# List available groups
python scripts/view_results.py --list-groups

# Detailed analysis of specific group
python scripts/view_results.py --detailed quantum

# Export data to CSV
python scripts/view_results.py --export
```

## Features

### 📊 Executive Summary
- High-level overview of all analysis results
- Key correlations within and between groups
- Peer effects model performance
- Automatically generated insights

### 📋 Group Details
- Detailed analysis for specific groups
- Correlation matrices with individual pair values
- Peer effects rankings and importance
- Cross-group interaction effects

### 📈 Data Export
- CSV exports for further analysis
- Correlation summary tables
- Peer effects performance metrics
- Metadata and configuration details

### 🎯 Key Insights
Automatically identifies:
- Strongest and weakest correlations
- Best performing peer effects models
- Groups with negative correlations
- Significant cross-group effects

## Example Output

### Compact Summary
```
📊 PEER GROUP ANALYSIS - COMPACT SUMMARY
---------------------------------------------
Period: 2025-08-02 to 2025-08-06
Groups: 4, Tickers: 11

Top Correlations:
  satellite: -0.053
  telecom: +0.005
  quantum: +0.005

Best Models (R²):
  quantum_iv: 0.980
  satellite_iv: 0.775
  telecom_iv: 0.644
```

### Detailed Group Analysis
```
============================================================
DETAILED ANALYSIS: QUANTUM
============================================================
📊 Tickers: QUBT, QBTS, RGTI, IONQ (4 total)

🔗 Intra-Group Correlations:
  IV Correlations: mean=0.005, std=0.038
  Return Correlations: mean=-0.006, std=0.018

🎯 Peer Effects Analysis:
  IV Analysis:
    Average R²: 0.980
    Average RMSE: 0.0233
    Success Rate: 4/4
```

## Command-Line Options

### Input Options
- `--file PATH`: Analyze specific results file
- `--results-dir PATH`: Set results directory (default: outputs/peer_groups)

### View Options
- `--detailed GROUP`: Show detailed analysis for specific group
- `--list-groups`: List all available groups
- `--correlations-only`: Show only correlation analysis
- `--peer-effects-only`: Show only peer effects analysis
- `--compact`: Use compact display format

### Export Options
- `--export`: Export data to CSV files
- `--save-report`: Save comprehensive text report
- `--output-dir PATH`: Set output directory for exports

### Display Options
- `--quiet`: Suppress status messages
- `--compact`: Use compact format

## Programmatic Usage

### Python API
```python
from src.results_summary_viewer import PeerGroupSummaryViewer, view_latest_results

# Load latest results
viewer = view_latest_results()

# Print executive summary
viewer.print_executive_summary()

# Analyze specific group
viewer.print_detailed_group_analysis("quantum")

# Export to DataFrames
corr_df = viewer.get_correlation_dataframe()
effects_df = viewer.get_peer_effects_dataframe()

# Save comprehensive report
report_path = viewer.save_summary_report()
```

### Convenience Functions
```python
from src.results_summary_viewer import quick_summary, detailed_group_report

# Quick summary of latest results
quick_summary()

# Detailed analysis of specific group
detailed_group_report("quantum")
```

## Understanding the Results

### Correlation Analysis
- **Intra-Group**: Correlations within each peer group
- **Inter-Group**: Correlations between different peer groups
- **Values**: Range from -1 (perfect negative) to +1 (perfect positive)
- **Significance**: Values > |0.05| are generally considered meaningful

### Peer Effects Analysis
- **R²**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root mean square error (lower is better)
- **Success Rate**: Proportion of targets successfully modeled
- **Cross Effects**: Effects from one group on another

### Key Insights
- **Negative Correlations**: Groups that move in opposite directions
- **Best Models**: Groups with highest predictive accuracy
- **Cross-Group Effects**: Significant spillover effects between groups

## File Outputs

### Generated Files
- `correlations_summary.csv`: All correlation data in tabular format
- `peer_effects_summary.csv`: All peer effects performance metrics
- `analysis_metadata.json`: Configuration and run information
- `summary_report_[timestamp].txt`: Comprehensive text report

### CSV Structure

#### Correlations Summary
| Column | Description |
|--------|-------------|
| Group_1 | First group name |
| Group_2 | Second group name |
| Relationship | "Intra-group" or "Inter-group" |
| IV_Correlation_Mean | Average IV correlation |
| Return_Correlation_Mean | Average return correlation |
| N_Tickers | Number of tickers involved |

#### Peer Effects Summary
| Column | Description |
|--------|-------------|
| Group | Group name or relationship |
| Target_Kind | "iv_ret" or "iv" |
| Analysis_Type | "Intra-group" or "Inter-group" |
| Avg_R2 | Average R-squared |
| Avg_RMSE | Average RMSE |
| Successful_Targets | Number of successful models |

## Tips for Analysis

### What to Look For
1. **High R² Values**: Groups with strong peer effects (> 0.7)
2. **Consistent Correlations**: Groups with stable relationships
3. **Cross-Group Effects**: Spillover effects between sectors
4. **Negative Correlations**: Potential hedging opportunities

### Interpretation Guidelines
- **R² > 0.8**: Excellent peer effects model
- **R² 0.5-0.8**: Good peer effects model
- **R² < 0.3**: Weak peer effects
- **|Correlation| > 0.1**: Potentially significant relationship
- **|Correlation| > 0.3**: Strong relationship

### Common Patterns
- **Sector Groups**: Usually show positive intra-group correlations
- **Market Stress**: Can increase all correlations
- **Idiosyncratic Events**: Can break normal correlation patterns
- **Size Effects**: Larger groups tend to have more stable patterns

## Troubleshooting

### Common Issues
1. **No Results Found**: Ensure peer group analysis has been run
2. **File Not Found**: Check the results directory path
3. **Group Not Found**: Use `--list-groups` to see available groups
4. **Export Fails**: Check write permissions on output directory

### Error Messages
- `Results file not found`: Verify the file path exists
- `Group 'X' not found`: Use correct group name from `--list-groups`
- `No data loaded`: Check that analysis completed successfully
- `Export failed`: Verify output directory permissions

### Debug Tips
1. Use `--quiet` to reduce output
2. Use `--file` to specify exact results file
3. Check `outputs/peer_groups/` for available results
4. Ensure analysis completed without errors
