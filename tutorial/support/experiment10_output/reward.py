def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return float(solution_str.strip() == str(ground_truth))
