# Auto-Fetch Enhancement Documentation

This document describes the enhanced automatic data fetching functionality that has been integrated into the peer group analysis pipeline.

## Overview

The enhancement ensures that when tickers are missing data, the system automatically attempts to download it from the Databento API before proceeding with analysis. This eliminates the need for manual data preparation and makes the analysis pipeline more robust.

## Key Enhancements

### 1. Enhanced `fetch_data_sqlite.py`

#### New Functions Added:

**`check_data_exists(conn, ticker, start, end)`**
- Comprehensively checks multiple tables (`atm_slices_1m`, `processed_merged_1m`, `merged_1m`) for data availability
- Returns `True` if any data exists for the ticker in the specified time window
- Provides detailed logging of data availability status

**`auto_fetch_missing_data(tickers, start, end, db_path, API_KEY=None)`**
- Automatically checks all tickers and fetches missing data
- Returns a summary dictionary with `fetched`, `skipped`, and `failed` lists
- Handles database initialization if needed
- Uses environment variable for API key if not provided

**`ensure_data_availability(tickers, start, end, db_path, auto_fetch=True)`**
- High-level function that ensures all required data is available
- Returns `True` if all data is available or successfully fetched
- Provides comprehensive logging and error handling

#### Enhanced `preprocess_and_store()`:
- Now uses the new `check_data_exists()` function for more reliable duplicate detection
- Better handles edge cases and provides clearer logging

### 2. Enhanced `data_loader_coordinator.py`

#### New Methods in `DataCoordinator`:

**`ensure_all_data_available(tickers, start_ts, end_ts)`**
- Uses the enhanced fetch functions for comprehensive data availability checking
- Provides fallback to basic checking if enhanced functions aren't available
- Integrates seamlessly with the existing workflow

**`_fallback_data_check(tickers, start_ts, end_ts)`**
- Backup method when enhanced fetch functions aren't available
- Maintains compatibility with existing setups

#### Enhanced `load_cores_with_fetch()`:
- Now includes an upfront data availability check using the enhanced functions
- Provides multi-layered fallback for individual ticker fetching
- Better error reporting and status tracking

### 3. Enhanced `peer_group_analyzer.py`

#### New Methods in `PeerGroupAnalyzer`:

**`_emergency_data_fetch()`**
- Emergency data fetching when initial loading fails completely
- Uses the enhanced fetch functionality as a last resort
- Provides detailed error logging and debugging information

**`_validate_data_for_peer_analysis()`**
- Validates that loaded data is suitable for peer group analysis
- Checks each group for minimum ticker requirements
- Provides early warning of potential analysis issues

#### Enhanced `load_data()`:
- Multi-stage data loading with progressive fallback
- Emergency data fetch if initial loading fails
- Comprehensive data quality validation
- Enhanced debugging and status reporting

## Usage Examples

### Basic Auto-Fetch

```python
from fetch_data_sqlite import auto_fetch_missing_data
import pandas as pd

tickers = ["ASTS", "SATS", "VZ", "T"]
start = pd.Timestamp("2025-08-02", tz="UTC")
end = pd.Timestamp("2025-08-06", tz="UTC")
db_path = Path("data/iv_data_1m.db")

# Automatically fetch any missing data
results = auto_fetch_missing_data(tickers, start, end, db_path)
print(f"Fetched: {results['fetched']}")
print(f"Skipped: {results['skipped']}")
print(f"Failed: {results['failed']}")
```

### Data Availability Assurance

```python
from fetch_data_sqlite import ensure_data_availability

# Ensure all data is available before analysis
success = ensure_data_availability(tickers, start, end, db_path, auto_fetch=True)
if success:
    print("All data is guaranteed to be available")
else:
    print("Some data could not be fetched")
```

### Integrated Peer Group Analysis

```python
from peer_group_analyzer import PeerGroupAnalyzer, PeerGroupConfig

config = PeerGroupConfig(
    groups={
        "satellite": ["ASTS", "SATS"],
        "telecom": ["VZ", "T"],
        "quantum": ["QUBT", "QBTS", "RGTI", "IONQ"],
        "crypto_miners": ["MARA", "WULF", "IREN"],
    },
    auto_fetch=True,  # Enable automatic data fetching
    debug=True        # Enable detailed logging
)

analyzer = PeerGroupAnalyzer(config)
results = analyzer.run_full_analysis()  # Will auto-fetch missing data
```

### Manual Data Coordination

```python
from data_loader_coordinator import DataCoordinator

coordinator = DataCoordinator(db_path=Path("data/iv_data_1m.db"))

# Ensure data availability before loading
coordinator.ensure_all_data_available(tickers, start_ts, end_ts)

# Load cores with enhanced fetching
cores = coordinator.load_cores_with_fetch(
    tickers=tickers,
    start=start,
    end=end,
    auto_fetch=True
)
```

## Configuration Options

### Environment Variables

- `DATABENTO_API_KEY`: Required for automatic data fetching
- `IV_DB_PATH`: Default database path (optional, defaults to `data/iv_data_1m.db`)

### PeerGroupConfig Parameters

- `auto_fetch`: Boolean flag to enable/disable automatic data fetching (default: `True`)
- `debug`: Enable detailed logging for troubleshooting (default: `False`)
- `db_path`: Path to the SQLite database

### DataCoordinator Parameters

- `auto_fetch`: Enable automatic fetching of missing data
- `drop_zero_iv_ret`: Filter out zero IV return rows during validation

## Error Handling and Fallbacks

The enhanced system provides multiple layers of fallback:

1. **Primary**: Enhanced auto-fetch using new functions
2. **Secondary**: Individual ticker fetching for remaining missing data
3. **Tertiary**: Basic data availability checking
4. **Emergency**: Last-resort data fetching in peer group analyzer

Each layer provides comprehensive error logging and continues to the next layer if issues occur.

## Logging and Debugging

### Status Messages

- `[EXISTS]`: Data found in database
- `[MISSING]`: Data not found, will attempt fetch
- `[SKIP]`: Data already available, skipping fetch
- `📥`: Fetching data
- `✅`: Operation successful
- `❌`: Operation failed
- `⚠️`: Warning or partial success

### Debug Mode

Enable debug mode for detailed information:

```python
config = PeerGroupConfig(debug=True)
```

This provides:
- Detailed data loading steps
- Database query information
- Fetch attempt details
- Data validation results
- Error stack traces

## Performance Considerations

### Optimizations

1. **Batch Checking**: Checks all tickers upfront before any fetching
2. **Duplicate Avoidance**: Comprehensive duplicate detection across multiple tables
3. **Smart Fallbacks**: Only attempts more expensive operations when needed
4. **Connection Reuse**: Efficient database connection management

### Recommendations

1. **Use Environment Variables**: Set `DATABENTO_API_KEY` to avoid repeated prompts
2. **Enable Auto-Fetch**: Use `auto_fetch=True` for seamless operation
3. **Database Location**: Place database in a fast storage location
4. **Batch Operations**: Process multiple tickers together when possible

## Integration with Existing Workflow

The enhancements are fully backward compatible:

1. **Existing Code**: Continues to work without modification
2. **Optional Features**: Auto-fetch can be disabled if needed
3. **Graceful Degradation**: Falls back to existing methods if new functions fail
4. **Error Isolation**: Issues with one ticker don't affect others

## Testing

Use the provided test script to verify functionality:

```bash
python examples/test_auto_fetch.py
```

This script tests:
- Basic auto-fetch functionality
- Data availability assurance
- Peer group integration
- Extended ticker groups (quantum, crypto, etc.)

## Troubleshooting

### Common Issues

1. **API Key Missing**: Set `DATABENTO_API_KEY` environment variable
2. **Database Permissions**: Ensure write access to database directory
3. **Network Issues**: Check internet connectivity for API calls
4. **Ticker Not Found**: Some tickers may not be available in Databento

### Debug Steps

1. Enable debug mode: `debug=True`
2. Check API key: `echo $DATABENTO_API_KEY`
3. Verify database: Check if `data/iv_data_1m.db` exists and is writable
4. Test connectivity: Try manual fetch with `fetch_data_sqlite.py`

### Error Recovery

The system is designed to be resilient:
- Failed fetches for one ticker don't stop analysis of others
- Multiple fallback methods ensure maximum data availability
- Comprehensive error logging helps identify and resolve issues

## Future Enhancements

Potential improvements for future versions:

1. **Parallel Fetching**: Download multiple tickers simultaneously
2. **Incremental Updates**: Fetch only new data since last update
3. **Cache Management**: Intelligent cache invalidation and refresh
4. **API Rate Limiting**: Respect API rate limits and retry logic
5. **Data Validation**: Post-fetch data quality verification
