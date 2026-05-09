"""
tests/test_summarizer.py
------------------------
Unit tests for preprocessing and evaluation modules.
Model loading tests are skipped by default (require internet + GPU time).
Run with:  pytest tests/ -v
"""

import pytest
from src.summarizer import preprocess_text
from src.evaluate   import compute_rouge


# ─────────────────────────────────────────────
#  Preprocessing Tests
# ─────────────────────────────────────────────

class TestPreprocessText:

    def test_strips_leading_trailing_whitespace(self):
        raw    = "   Hello world.   "
        result = preprocess_text(raw)
        assert result == result.strip()

    def test_collapses_multiple_spaces(self):
        raw    = "Hello    world"
        result = preprocess_text(raw)
        assert "  " not in result

    def test_removes_newlines(self):
        raw    = "Line one.\n\nLine two.\nLine three."
        result = preprocess_text(raw)
        assert "\n" not in result

    def test_removes_urls(self):
        raw    = "Visit https://example.com for details."
        result = preprocess_text(raw)
        assert "http" not in result

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_preserves_basic_punctuation(self):
        raw    = "Hello, world! How are you?"
        result = preprocess_text(raw)
        assert "," in result
        assert "!" in result
        assert "?" in result


# ─────────────────────────────────────────────
#  ROUGE Evaluation Tests
# ─────────────────────────────────────────────

class TestComputeRouge:

    def test_perfect_match(self):
        text   = ["The cat sat on the mat."]
        scores = compute_rouge(text, text)
        # Perfect match → F1 = 1.0 for rouge1
        assert scores["rouge1"]["fmeasure"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_overlap(self):
        pred   = ["apple orange banana"]
        ref    = ["car truck bus"]
        scores = compute_rouge(pred, ref)
        assert scores["rouge1"]["fmeasure"] == pytest.approx(0.0, abs=1e-3)

    def test_returns_all_rouge_types(self):
        pred   = ["The quick brown fox."]
        ref    = ["The quick brown fox jumped over the lazy dog."]
        scores = compute_rouge(pred, ref)
        assert "rouge1"  in scores
        assert "rouge2"  in scores
        assert "rougeL"  in scores

    def test_scores_between_zero_and_one(self):
        pred   = ["AI is transforming the world of technology."]
        ref    = ["Artificial intelligence is changing technology rapidly."]
        scores = compute_rouge(pred, ref)
        for metric, vals in scores.items():
            for k, v in vals.items():
                assert 0.0 <= v <= 1.0, f"{metric}.{k} out of range: {v}"

    def test_batch_averaging(self):
        preds = ["The cat sat.", "Dogs are loyal."]
        refs  = ["A cat was sitting.", "Dogs show loyalty."]
        scores = compute_rouge(preds, refs)
        # Just check it runs and returns valid structure
        assert isinstance(scores["rouge1"]["fmeasure"], float)
