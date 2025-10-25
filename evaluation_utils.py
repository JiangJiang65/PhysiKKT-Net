# -*- coding: utf-8 -*-
"""
Evaluation Utility Functions
Contains functions for metric level assessment and display.
"""

def get_metric_level(metric_name, value):
    """
    Determines the performance level based on a metric's name and value.
    
    Parameters
    ----------
    metric_name : str
        The name of the metric.
    value : float
        The value of the metric.
        
    Returns
    -------
    tuple
        A tuple containing (Level, Color Code, Description).
    """
    
    # Define threshold standards (originally from a file like compute_score.py)
    # Format: (lower_threshold, upper_threshold, "min" or "max")
    # "min": Lower values are better.
    # "max": Higher values are better.
    thresholds = {
        "a_or": (0.02, 0.05, "min"),
        "a_ex": (0.02, 0.05, "min"),
        "p_or": (0.02, 0.05, "min"),
        "p_ex": (0.02, 0.05, "min"),
        "v_or": (0.2, 0.5, "min"),
        "v_ex": (0.2, 0.5, "min"),
        "CURRENT_POS": (1., 5., "min"),
        "VOLTAGE_POS": (1., 5., "min"),
        "DISC_LINES": (1., 5., "min"),
        "CHECK_LOSS": (1., 5., "min"),
        "CHECK_GC": (0.05, 0.10, "min"),
        "CHECK_LC": (0.05, 0.10, "min"),
    }
    
    # Check if a threshold exists for the metric
    if metric_name not in thresholds:
        return ("Unknown", "⚪", "Unknown Level")
    
    threshold_min, threshold_max, eval_type = thresholds[metric_name]
    
    if eval_type == "min":
        if value < threshold_min:
            return ("Great", "🟢", "Great")
        elif threshold_min <= value < threshold_max:
            return ("Acceptable", "🟡", "Acceptable")
        else:
            return ("Unacceptable", "🔴", "Unacceptable")
    elif eval_type == "max":
        if value < threshold_min:
            return ("Unacceptable", "🔴", "Unacceptable")
        elif threshold_min <= value < threshold_max:
            return ("Acceptable", "🟡", "Acceptable")
        else:
            return ("Great", "🟢", "Great")
    
    return ("Unknown", "⚪", "Unknown Level")

def print_metric_with_level(metric_name, value, unit=""):
    """
    Prints a metric and its assessed level.
    
    Parameters
    ----------
    metric_name : str
        The name of the metric.
    value : float
        The value of the metric.
    unit : str, optional
        The unit of the metric. The default is "".
    """
    level, color, description = get_metric_level(metric_name, value)
    print(f"{color} {metric_name}: {value:.4f}{unit} - {description} ({level})")

def evaluate_metrics_with_levels(metrics_dict):
    """
    Evaluates a dictionary of metrics and displays their performance levels.
    
    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing various metrics.
    """
    print("="*60)
    print("📊 Metric Evaluation Results (with Levels)")
    print("="*60)
    
    # Machine Learning Metrics
    if "ML" in metrics_dict:
        print("\n🤖 Machine Learning Metrics (ML Metrics):")
        print("-" * 40)
        
        ml_metrics = metrics_dict["ML"]
        
        # Current metrics (using MAPE_90_avg)
        if "MAPE_90_avg" in ml_metrics:
            print("\n📈 Current Prediction:")
            for key in ["a_or", "a_ex"]:
                if key in ml_metrics["MAPE_90_avg"]:
                    print_metric_with_level(key, ml_metrics["MAPE_90_avg"][key], "")
        
        # Power metrics (using MAPE_10_avg)
        if "MAPE_10_avg" in ml_metrics:
            print("\n⚡ Power Prediction:")
            for key in ["p_or", "p_ex"]:
                if key in ml_metrics["MAPE_10_avg"]:
                    print_metric_with_level(key, ml_metrics["MAPE_10_avg"][key], "")
        
        # Voltage metrics (using MAE_avg)
        if "MAE_avg" in ml_metrics:
            print("\n🔌 Voltage Prediction:")
            for key in ["v_or", "v_ex"]:
                if key in ml_metrics["MAE_avg"]:
                    print_metric_with_level(key, ml_metrics["MAE_avg"][key], "")
    
    # Physics Compliance Metrics
    if "Physics" in metrics_dict:
        print("\n⚛️ Physics Compliance Metrics:")
        print("-" * 40)
        
        physics_metrics = metrics_dict["Physics"]
        
        # Current Positivity
        if "CURRENT_POS" in physics_metrics:
            print("\n🔋 Current Positivity:")
            current_pos = physics_metrics["CURRENT_POS"]
            for key in ["a_or", "a_ex"]:
                if key in current_pos and "Violation_proportion" in current_pos[key]:
                    violation_pct = current_pos[key]["Violation_proportion"] * 100
                    print_metric_with_level("CURRENT_POS", violation_pct, "%")
        
        # Voltage Positivity
        if "VOLTAGE_POS" in physics_metrics:
            print("\n⚡ Voltage Positivity:")
            voltage_pos = physics_metrics["VOLTAGE_POS"]
            for key in ["v_or", "v_ex"]:
                if key in voltage_pos and "Violation_proportion" in voltage_pos[key]:
                    violation_pct = voltage_pos[key]["Violation_proportion"] * 100
                    print_metric_with_level("VOLTAGE_POS", violation_pct, "%")
        
        # Other physics metrics
        # physics_keys = ["LOSS_POS", "DISC_LINES", "CHECK_LOSS", "CHECK_GC", "CHECK_LC", "CHECK_JOULE_LAW"]
        physics_keys = ["DISC_LINES", "CHECK_LOSS", "CHECK_GC", "CHECK_LC"]
        for key in physics_keys:
            if key in physics_metrics:
                if isinstance(physics_metrics[key], dict):
                    if "violation_proportion" in physics_metrics[key]:
                        violation_pct = physics_metrics[key]["violation_proportion"] * 100
                        print_metric_with_level(key, violation_pct, "%")
                    elif "violation_percentage" in physics_metrics[key]:
                        violation_pct = physics_metrics[key]["violation_percentage"]
                        print_metric_with_level(key, violation_pct, "%")
    
    print("\n" + "="*60)
    print("📋 Legend for Levels:")
    print("🟢 Great: Metric value is below the first threshold.")
    print("🟡 Acceptable: Metric value is between the two thresholds.")
    print("🔴 Unacceptable: Metric value is above the second threshold.")
    print("="*60)

def get_overall_assessment(metrics_dict):
    """
    Gets an overall assessment by counting metrics in each performance level.
    
    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing various metrics.
        
    Returns
    -------
    dict
        A dictionary containing the count for each level.
    """
    level_counts = {"Great": 0, "Acceptable": 0, "Unacceptable": 0, "Unknown": 0}
    
    # Tally Machine Learning metrics
    if "ML" in metrics_dict:
        ml_metrics = metrics_dict["ML"]
        
        # Current metrics
        if "MAPE_90_avg" in ml_metrics:
            for key in ["a_or", "a_ex"]:
                if key in ml_metrics["MAPE_90_avg"]:
                    level, _, _ = get_metric_level(key, ml_metrics["MAPE_90_avg"][key])
                    level_counts[level] += 1
        
        # Power metrics
        if "MAPE_10_avg" in ml_metrics:
            for key in ["p_or", "p_ex"]:
                if key in ml_metrics["MAPE_10_avg"]:
                    level, _, _ = get_metric_level(key, ml_metrics["MAPE_10_avg"][key])
                    level_counts[level] += 1
        
        # Voltage metrics
        if "MAE_avg" in ml_metrics:
            for key in ["v_or", "v_ex"]:
                if key in ml_metrics["MAE_avg"]:
                    level, _, _ = get_metric_level(key, ml_metrics["MAE_avg"][key])
                    level_counts[level] += 1
    
    # Tally Physics Compliance metrics
    if "Physics" in metrics_dict:
        physics_metrics = metrics_dict["Physics"]
        
        # Current Positivity
        if "CURRENT_POS" in physics_metrics:
            current_pos = physics_metrics["CURRENT_POS"]
            for key in ["a_or", "a_ex"]:
                if key in current_pos and "Violation_proportion" in current_pos[key]:
                    violation_pct = current_pos[key]["Violation_proportion"] * 100
                    level, _, _ = get_metric_level("CURRENT_POS", violation_pct)
                    level_counts[level] += 1
        
        # Voltage Positivity
        if "VOLTAGE_POS" in physics_metrics:
            voltage_pos = physics_metrics["VOLTAGE_POS"]
            for key in ["v_or", "v_ex"]:
                if key in voltage_pos and "Violation_proportion" in voltage_pos[key]:
                    violation_pct = voltage_pos[key]["Violation_proportion"] * 100
                    level, _, _ = get_metric_level("VOLTAGE_POS", violation_pct)
                    level_counts[level] += 1
    
    return level_counts

def print_overall_summary(metrics_dict):
    """
    Prints an overall assessment summary.
    
    Parameters
    ----------
    metrics_dict : dict
        A dictionary containing various metrics.
    """
    level_counts = get_overall_assessment(metrics_dict)
    total = sum(level_counts.values())
    
    if total == 0:
        print("\nNo metrics were assessed.")
        return

    print("\n" + "="*60)
    print("📊 Overall Assessment Summary")
    print("="*60)
    print(f"🟢 Great: {level_counts['Great']}/{total} ({level_counts['Great']/total*100:.1f}%)")
    print(f"🟡 Acceptable: {level_counts['Acceptable']}/{total} ({level_counts['Acceptable']/total*100:.1f}%)")
    print(f"🔴 Unacceptable: {level_counts['Unacceptable']}/{total} ({level_counts['Unacceptable']/total*100:.1f}%)")
    
    if level_counts['Unknown'] > 0:
        print(f"⚪ Unknown: {level_counts['Unknown']}/{total} ({level_counts['Unknown']/total*100:.1f}%)")
    
    # Provide recommendations
    if level_counts['Unacceptable'] > level_counts['Great']:
        print("\n⚠️ Recommendation: The model needs improvement, especially on the unacceptable metrics.")
    elif level_counts['Great'] > level_counts['Acceptable'] + level_counts['Unacceptable']:
        print("\n✅ Recommendation: The model's performance is excellent!")
    else:
        print("\n📈 Recommendation: The model's performance is good and can be further optimized.")
    
    print("="*60)