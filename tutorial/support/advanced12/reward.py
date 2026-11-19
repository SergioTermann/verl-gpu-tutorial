def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Exact-match binary reward for the decimal parity task."""
    if data_source != 'decimal_parity':
        return 0.0
    prediction = solution_str.strip().lower()
    target = str(ground_truth).strip().lower()
    return float(prediction in {'yes', 'no'} and prediction == target)
