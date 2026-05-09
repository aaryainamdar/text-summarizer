"""
evaluate.py
-----------
Evaluation utilities for summarization quality.
Computes ROUGE scores to benchmark model output vs. reference summaries.
"""

from rouge_score import rouge_scorer


def compute_rouge(
    predictions: list,
    references: list,
    rouge_types: list = None,
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    the overlap between machine-generated and reference summaries.

    Args:
        predictions (list[str]): Model-generated summaries.
        references  (list[str]): Human-written reference summaries.
        rouge_types (list[str]): Which ROUGE variants to compute.
                                 Defaults to ['rouge1', 'rouge2', 'rougeL'].

    Returns:
        dict: {metric_name: {precision, recall, f1}} averaged across all samples.

    Example:
        >>> scores = compute_rouge(["The cat sat."], ["The cat is sitting."])
        >>> print(scores["rouge1"])
        {'precision': 0.75, 'recall': 0.6, 'fmeasure': 0.666...}
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer   = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    totals   = {rt: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for rt in rouge_types}
    n        = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for rt in rouge_types:
            totals[rt]["precision"] += scores[rt].precision
            totals[rt]["recall"]    += scores[rt].recall
            totals[rt]["fmeasure"]  += scores[rt].fmeasure

    # Average across samples
    averaged = {
        rt: {k: round(v / n, 4) for k, v in totals[rt].items()}
        for rt in rouge_types
    }
    return averaged


def print_rouge_table(scores: dict):
    """Pretty-print ROUGE scores as a formatted table."""
    print("\n" + "=" * 52)
    print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for metric, vals in scores.items():
        print(
            f"{metric:<12} "
            f"{vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} "
            f"{vals['fmeasure']:>10.4f}"
        )
    print("=" * 52 + "\n")
