"""
demo.py
-------
Command-line demo for the Text Summarizer.

Usage:
    # Summarize a sample article (built-in)
    python demo.py

    # Summarize your own text file
    python demo.py --file path/to/article.txt

    # Choose a different model
    python demo.py --model t5-base

    # Adjust summary length
    python demo.py --min_length 80 --max_length 250

    # Run evaluation on sample data
    python demo.py --evaluate
"""

import argparse
import time

from src.summarizer import SummarizerModel, preprocess_text
from src.evaluate   import compute_rouge, print_rouge_table


# ──────────────────────────────────────────────
#  Sample article for demo (CNN/DailyMail style)
# ──────────────────────────────────────────────

SAMPLE_ARTICLE = """
We presented CIS-PMP, a cloud-integrated predictive maintenance framework for industrial robot fleets that unifies high-velocity ingestion, real-time analytics, sequential modeling with attention, and a zero-trust security posture. On the AI4I 2020 dataset and a production-like AWS stack, CIS-PMP achieved 94.3% accuracy and 0.901 F1, while reducing false positives per day and realizing p95 inference of approximately 120 ms for batch-32. Against strong baselines—including a commercial PdM platform—CIS-PMP improved accuracy by approximately 5.1% and lowered monthly compute and storage costs by approximately 47%, translating into a 74% MTBF increase, a 38% maintenance cost reduction, and a greater than 50% cut in unplanned downtime in our 12-month evaluation. These results indicate that carefully engineered cloud-native sequential models with adaptive learning and uncertainty quantification can deliver reliable, cost-efficient PdM at scale.
The experimental results comprehensively address our three research questions. RQ1: CIS-PMP demonstrated superior performance compared to existing approaches, with 5.1% higher accuracy than commercial solutions (Siemens MindSphere) and 47% lower computational costs than baseline cloud implementations. RQ2: The adaptive learning system maintained prediction accuracy within 2.3% of initial performance over 12 months of operation, compared to 18.7% degradation in static models, directly enabling sustained maintenance cost reduction and equipment reliability improvement. RQ3: The security framework successfully mitigated all identified threats during independent penetration testing while maintaining compliance with IEC 62443-3-3 industrial security standards and imposing less than 5% performance overhead.
Future work will focus on: (1) privacy-preserving federated learning across sites to enable collaborative model improvement without data centralization; (2) explainability through attention maps and feature attributions for operator trust and regulatory compliance; (3) post-quantum cryptography pilots to prepare for quantum computing threats; (4) broader cross-asset generalization beyond robots to pumps, conveyors, and other industrial equipment; and (5) integration with digital twins for what-if simulation and predictive scenario analysis. We also plan to open-source a reference configuration and extend evaluations with drift stress-tests and additional ablations to further validate robustness in diverse industrial conditions.

"""

SAMPLE_REFERENCE = (
    "We presented CIS-PMP, a cloud-integrated predictive maintenance framework for industrial robot fleets that unifies high-velocity ingestion, real-time analytics, sequential modeling with attention, and a zero-trust security posture. On the AI4I 2020 dataset and a production-like AWS stack, CIS- PMP achieved 94.3 accuracy and 0.901 F1."
)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🤖 NLP Text Summarizer — BART / T5"
    )
    parser.add_argument("--model",      type=str, default="facebook/bart-large-cnn",
                        help="HuggingFace model name")
    parser.add_argument("--file",       type=str, default=None,
                        help="Path to a .txt file to summarize")
    parser.add_argument("--min_length", type=int, default=60,
                        help="Minimum summary length (tokens)")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum summary length (tokens)")
    parser.add_argument("--num_beams",  type=int, default=4,
                        help="Beam search width")
    parser.add_argument("--evaluate",   action="store_true",
                        help="Run ROUGE evaluation on sample data")
    args = parser.parse_args()

    # ── Load text ─────────────────────────────
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            article = f.read()
        print(f"[INFO] Loaded article from: {args.file}")
    else:
        article = SAMPLE_ARTICLE
        print("[INFO] Using built-in sample article.\n")

    # ── Load model ────────────────────────────
    model = SummarizerModel(model_name=args.model)

    # ── Preprocess ────────────────────────────
    print("── Preprocessing ──────────────────────────────────")
    clean = preprocess_text(article)
    print(f"Original length : {len(article.split())} words")
    print(f"Cleaned length  : {len(clean.split())} words\n")

    # ── Summarize ─────────────────────────────
    print("── Generating Summary ─────────────────────────────")
    start   = time.time()
    summary = model.summarize(
        article,
        min_summary_length = args.min_length,
        max_summary_length = args.max_length,
        num_beams          = args.num_beams,
    )
    elapsed = time.time() - start

    print(f"\n📄 SUMMARY ({elapsed:.2f}s):")
    print("─" * 55)
    print(summary)
    print("─" * 55)
    print(f"\nCompression ratio: {len(summary.split()) / len(clean.split()):.1%}\n")

    # ── ROUGE Evaluation ──────────────────────
    if args.evaluate:
        print("── ROUGE Evaluation ───────────────────────────────")
        scores = compute_rouge([summary], [SAMPLE_REFERENCE])
        print_rouge_table(scores)


if __name__ == "__main__":
    main()
