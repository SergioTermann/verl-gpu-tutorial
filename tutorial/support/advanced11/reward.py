def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Synthetic, deterministic reward used only to exercise the training pipeline.
    # solution_str 是一条解码后的回答。字节和取余再除以 100，得到 0 到 1 的分数；它只用来检查训练链路是否能传回不同奖励。
    return sum(solution_str.encode('utf-8')) % 101 / 100.0
