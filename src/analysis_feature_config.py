"""
Analysis Feature Configuration

This module defines which features each analysis type should access and how 
features should be filtered/selected for optimal performance of each analysis.
"""

from typing import Dict, List, Sequence, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class AnalysisFeatureConfig:
    """Configuration for feature selection by analysis type."""
    
    # Feature categories
    CORE_OPTION_FEATURES = [
        "opt_volume", "time_to_expiry", "days_to_expiry", "strike_price", "option_type_enc"
    ]
    
    GREEKS_FEATURES = [
        "delta", "gamma", "vega"
    ]
    
    TIME_FEATURES = [
        "hour", "minute", "day_of_week"
    ]
    
    SABR_FEATURES = [
        "sabr_alpha", "sabr_rho", "sabr_nu", "moneyness", "log_moneyness"
    ]
    
    VOLATILITY_FEATURES = [
        "rv_30m", "iv_ret_1m", "iv_ret_5m", "iv_ret_15m",
        "iv_sma_5m", "iv_sma_15m", "iv_std_15m", "iv_rsi_15m", "iv_zscore_15m"
    ]
    
    VOLUME_FEATURES = [
        "opt_vol_change_1m", "opt_vol_roll_5m", "opt_vol_roll_15m", 
        "opt_vol_roll_60m", "opt_vol_zscore_15m"
    ]
    
    # Peer/Panel features (dynamic based on tickers)
    PANEL_IV_PATTERN = "panel_IV_"
    PANEL_IVRET_PATTERN = "panel_IVRET_"
    SYMBOL_PATTERN = "sym_"
    
    # Features that indicate data leakage for specific targets
    LEAKAGE_FEATURES = {
        "iv_ret_fwd": ["iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
        "iv_ret_fwd_abs": ["iv_ret_fwd"],
        "iv_clip": ["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],
    }
    
    # Meta columns that should be excluded from modeling
    META_COLUMNS = ["ts_event", "expiry_date", "symbol"]


def get_features_for_analysis_type(
    analysis_type: str, 
    available_columns: List[str],
    target_column: Optional[str] = None,
    include_peer_features: bool = True,
    include_time_features: bool = True,
    include_advanced_features: bool = True
) -> List[str]:
    """
    Get appropriate feature set for specific analysis type.
    
    Parameters
    ----------
    analysis_type : str
        Type of analysis: 'pooled_iv_returns', 'peer_effects', 'baseline_correlation',
        'iv_level_modeling', 'correlation_analysis'
    available_columns : List[str]
        All available columns in the dataset
    target_column : str, optional
        Target column name for supervised learning tasks
    include_peer_features : bool
        Whether to include cross-ticker panel features
    include_time_features : bool
        Whether to include time-of-day features
    include_advanced_features : bool
        Whether to include advanced SABR/volatility features
        
    Returns
    -------
    List[str]
        List of feature column names appropriate for the analysis
    """
    
    config = AnalysisFeatureConfig()
    selected_features = []
    
    # Always include core option features
    selected_features.extend([col for col in config.CORE_OPTION_FEATURES if col in available_columns])
    
    # Always include Greeks for options analysis
    selected_features.extend([col for col in config.GREEKS_FEATURES if col in available_columns])
    
    # Analysis-specific feature selection
    if analysis_type == "pooled_iv_returns":
        # Pooled IV return prediction - needs all features for cross-sectional learning
        if include_time_features:
            selected_features.extend([col for col in config.TIME_FEATURES if col in available_columns])
        if include_advanced_features:
            selected_features.extend([col for col in config.SABR_FEATURES if col in available_columns])
            selected_features.extend([col for col in config.VOLATILITY_FEATURES if col in available_columns])
        selected_features.extend([col for col in config.VOLUME_FEATURES if col in available_columns])
        
        if include_peer_features:
            # Include panel features for cross-ticker effects
            panel_features = [col for col in available_columns 
                            if col.startswith((config.PANEL_IV_PATTERN, config.PANEL_IVRET_PATTERN))]
            selected_features.extend(panel_features)
            
            # Include symbol dummies for pooled learning
            symbol_features = [col for col in available_columns if col.startswith(config.SYMBOL_PATTERN)]
            selected_features.extend(symbol_features)
    
    elif analysis_type == "peer_effects":
        # Peer effects modeling - focus on peer relationships and momentum
        if include_time_features:
            selected_features.extend([col for col in config.TIME_FEATURES if col in available_columns])
        
        # For peer effects, volume and volatility momentum are most important
        selected_features.extend([col for col in config.VOLUME_FEATURES if col in available_columns])
        selected_features.extend([col for col in config.VOLATILITY_FEATURES if col in available_columns])
        
        # Include peer panel features (core of peer effects analysis)
        if include_peer_features:
            panel_features = [col for col in available_columns 
                            if col.startswith((config.PANEL_IV_PATTERN, config.PANEL_IVRET_PATTERN))]
            selected_features.extend(panel_features)
        
        # SABR features less critical for peer effects
        if include_advanced_features:
            selected_features.extend([col for col in config.SABR_FEATURES if col in available_columns])
    
    elif analysis_type == "iv_level_modeling":
        # IV level prediction - structural features more important than momentum
        if include_advanced_features:
            selected_features.extend([col for col in config.SABR_FEATURES if col in available_columns])
        
        # Time features important for IV level (intraday patterns)
        if include_time_features:
            selected_features.extend([col for col in config.TIME_FEATURES if col in available_columns])
        
        # Volatility features for regime detection
        vol_features = [col for col in config.VOLATILITY_FEATURES 
                       if col in available_columns and not col.startswith("iv_ret")]  # Exclude momentum
        selected_features.extend(vol_features)
        
        # Limited volume features (just basic flow)
        basic_vol_features = [col for col in config.VOLUME_FEATURES 
                             if col in available_columns and "roll" in col]  # Just rolling averages
        selected_features.extend(basic_vol_features)
        
        # Peer features for relative value
        if include_peer_features:
            panel_iv_features = [col for col in available_columns if col.startswith(config.PANEL_IV_PATTERN)]
            selected_features.extend(panel_iv_features)
    
    elif analysis_type == "baseline_correlation":
        # Correlation analysis - only needs IV and return series (handled separately)
        # This analysis doesn't use machine learning features
        return []
    
    elif analysis_type == "cross_sectional":
        # Cross-sectional analysis - relative features most important
        if include_peer_features:
            panel_features = [col for col in available_columns 
                            if col.startswith((config.PANEL_IV_PATTERN, config.PANEL_IVRET_PATTERN))]
            selected_features.extend(panel_features)
        
        # Time features for regime identification
        if include_time_features:
            selected_features.extend([col for col in config.TIME_FEATURES if col in available_columns])
        
        # Relative volatility features
        relative_vol_features = [col for col in config.VOLATILITY_FEATURES 
                               if col in available_columns and ("zscore" in col or "rsi" in col)]
        selected_features.extend(relative_vol_features)
    
    else:
        # Default: include all non-leakage features
        if include_time_features:
            selected_features.extend([col for col in config.TIME_FEATURES if col in available_columns])
        if include_advanced_features:
            selected_features.extend([col for col in config.SABR_FEATURES if col in available_columns])
            selected_features.extend([col for col in config.VOLATILITY_FEATURES if col in available_columns])
        selected_features.extend([col for col in config.VOLUME_FEATURES if col in available_columns])
        
        if include_peer_features:
            panel_features = [col for col in available_columns 
                            if col.startswith((config.PANEL_IV_PATTERN, config.PANEL_IVRET_PATTERN))]
            selected_features.extend(panel_features)
    
    # Remove duplicates while preserving order
    selected_features = list(dict.fromkeys(selected_features))
    
    # Remove meta columns
    selected_features = [col for col in selected_features if col not in config.META_COLUMNS]
    
    # Remove target-specific leakage features
    if target_column and target_column in config.LEAKAGE_FEATURES:
        leakage_cols = config.LEAKAGE_FEATURES[target_column]
        selected_features = [col for col in selected_features if col not in leakage_cols]
    
    # Remove the target column itself
    if target_column and target_column in selected_features:
        selected_features.remove(target_column)
    
    return selected_features


def filter_features_by_importance(
    features: List[str], 
    analysis_type: str,
    max_features: Optional[int] = None
) -> List[str]:
    """
    Filter features by importance for specific analysis type.
    
    Parameters
    ----------
    features : List[str]
        List of available features
    analysis_type : str
        Type of analysis
    max_features : int, optional
        Maximum number of features to return
        
    Returns
    -------
    List[str]
        Filtered and prioritized feature list
    """
    
    config = AnalysisFeatureConfig()
    
    # Define importance order by analysis type
    importance_order = {
        "pooled_iv_returns": [
            config.GREEKS_FEATURES,
            config.VOLATILITY_FEATURES,
            config.VOLUME_FEATURES,
            config.CORE_OPTION_FEATURES,
            config.TIME_FEATURES,
            config.SABR_FEATURES,
        ],
        "peer_effects": [
            config.VOLATILITY_FEATURES,
            config.VOLUME_FEATURES,
            config.GREEKS_FEATURES,
            config.TIME_FEATURES,
            config.CORE_OPTION_FEATURES,
            config.SABR_FEATURES,
        ],
        "iv_level_modeling": [
            config.SABR_FEATURES,
            config.GREEKS_FEATURES,
            config.CORE_OPTION_FEATURES,
            config.TIME_FEATURES,
            config.VOLATILITY_FEATURES,
            config.VOLUME_FEATURES,
        ]
    }
    
    # Get importance order for this analysis type
    feature_groups = importance_order.get(analysis_type, [
        config.GREEKS_FEATURES,
        config.CORE_OPTION_FEATURES,
        config.VOLATILITY_FEATURES,
        config.VOLUME_FEATURES,
        config.TIME_FEATURES,
        config.SABR_FEATURES,
    ])
    
    # Order features by importance
    ordered_features = []
    
    # Add features in order of importance groups
    for group in feature_groups:
        group_features = [f for f in features if f in group]
        ordered_features.extend(group_features)
    
    # Add any remaining features (like panel features)
    remaining_features = [f for f in features if f not in ordered_features]
    
    # For peer/panel features, prioritize IV levels over returns for most analyses
    panel_iv = [f for f in remaining_features if f.startswith("panel_IV_")]
    panel_ret = [f for f in remaining_features if f.startswith("panel_IVRET_")]
    symbol_features = [f for f in remaining_features if f.startswith("sym_")]
    other_features = [f for f in remaining_features 
                     if not f.startswith(("panel_", "sym_"))]
    
    # Add in priority order
    ordered_features.extend(panel_iv)
    ordered_features.extend(panel_ret)
    ordered_features.extend(symbol_features)
    ordered_features.extend(other_features)
    
    # Apply max_features limit if specified
    if max_features and len(ordered_features) > max_features:
        ordered_features = ordered_features[:max_features]
    
    return ordered_features


def get_feature_importance_weights(
    features: List[str], 
    analysis_type: str
) -> Dict[str, float]:
    """
    Get feature importance weights for analysis type.
    
    Parameters
    ----------
    features : List[str]
        List of features
    analysis_type : str
        Type of analysis
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to importance weights
    """
    
    config = AnalysisFeatureConfig()
    weights = {}
    
    # Base weights by feature category
    base_weights = {
        "pooled_iv_returns": {
            "greeks": 1.0,
            "volatility": 0.9,
            "volume": 0.8,
            "core_option": 0.7,
            "time": 0.6,
            "sabr": 0.5,
            "panel": 0.8,
            "symbol": 0.3,
        },
        "peer_effects": {
            "volatility": 1.0,
            "volume": 0.9,
            "panel": 0.9,
            "greeks": 0.7,
            "time": 0.6,
            "core_option": 0.5,
            "sabr": 0.4,
            "symbol": 0.2,
        },
        "iv_level_modeling": {
            "sabr": 1.0,
            "greeks": 0.9,
            "core_option": 0.8,
            "time": 0.7,
            "volatility": 0.6,
            "volume": 0.4,
            "panel": 0.6,
            "symbol": 0.2,
        }
    }
    
    # Default weights
    default_weights = {
        "greeks": 0.8,
        "core_option": 0.7,
        "volatility": 0.6,
        "volume": 0.5,
        "time": 0.4,
        "sabr": 0.4,
        "panel": 0.5,
        "symbol": 0.2,
    }
    
    analysis_weights = base_weights.get(analysis_type, default_weights)
    
    # Assign weights to each feature
    for feature in features:
        if feature in config.GREEKS_FEATURES:
            weights[feature] = analysis_weights["greeks"]
        elif feature in config.CORE_OPTION_FEATURES:
            weights[feature] = analysis_weights["core_option"]
        elif feature in config.VOLATILITY_FEATURES:
            weights[feature] = analysis_weights["volatility"]
        elif feature in config.VOLUME_FEATURES:
            weights[feature] = analysis_weights["volume"]
        elif feature in config.TIME_FEATURES:
            weights[feature] = analysis_weights["time"]
        elif feature in config.SABR_FEATURES:
            weights[feature] = analysis_weights["sabr"]
        elif feature.startswith(("panel_", "IV_", "IVRET_")):
            weights[feature] = analysis_weights["panel"]
        elif feature.startswith("sym_"):
            weights[feature] = analysis_weights["symbol"]
        else:
            weights[feature] = 0.3  # Default for unknown features
    
    return weights


# Export the main functions
__all__ = [
    "AnalysisFeatureConfig",
    "get_features_for_analysis_type",
    "filter_features_by_importance", 
    "get_feature_importance_weights"
]
