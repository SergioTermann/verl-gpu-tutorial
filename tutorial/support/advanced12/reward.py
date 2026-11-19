def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Exact-match binary reward for the decimal parity task."""
    # data_source 来自每道题的数据行，用来选择评分规则；题型不匹配时这里给零分。
    if data_source != 'decimal_parity':
        return 0.0
    # solution_str 是模型自由生成的文本，ground_truth 是数据中保存的答案。先统一大小写和首尾空白，再做整串比较。
    prediction = solution_str.strip().lower()
    target = str(ground_truth).strip().lower()
    return float(prediction in {'yes', 'no'} and prediction == target)
