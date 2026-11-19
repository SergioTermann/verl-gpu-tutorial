def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Synthetic, deterministic reward used only to exercise the training pipeline.
    return sum(solution_str.encode('utf-8')) % 101 / 100.0
