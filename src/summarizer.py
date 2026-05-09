"""
summarizer.py
-------------
Core summarization pipeline using HuggingFace Transformers.
Supports BART and T5 pretrained models out of the box.
"""

import re
import torch
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)


# ─────────────────────────────────────────────
#  Text Preprocessing
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw input text before tokenization.

    Steps:
        1. Strip leading/trailing whitespace
        2. Collapse multiple newlines and spaces
        3. Remove non-ASCII characters (optional – configurable)
        4. Basic sentence boundary normalization

    Args:
        text (str): Raw article / document text.

    Returns:
        str: Cleaned text ready for tokenization.
    """
    # Remove excessive whitespace
    text = text.strip()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)

    return text


# ─────────────────────────────────────────────
#  Model Loader
# ─────────────────────────────────────────────

class SummarizerModel:
    """
    Wrapper around HuggingFace pretrained models for text summarization.

    Supported architectures:
        - facebook/bart-large-cnn  (BART)
        - t5-small / t5-base / t5-large  (T5)

    Args:
        model_name (str): HuggingFace model hub identifier.
        device      (str): 'cuda' or 'cpu'. Auto-detected if not specified.

    Example:
        >>> model = SummarizerModel("facebook/bart-large-cnn")
        >>> summary = model.summarize("Long article text here...")
        >>> print(summary)
    """

    SUPPORTED_MODELS = {
        "bart": [
            "facebook/bart-large-cnn",
            "facebook/bart-base",
        ],
        "t5": [
            "t5-small",
            "t5-base",
            "t5-large",
            "google/flan-t5-base",
        ],
    }

    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type(model_name)

        print(f"[INFO] Loading model  : {model_name}")
        print(f"[INFO] Model type     : {self.model_type.upper()}")
        print(f"[INFO] Running on     : {self.device.upper()}")

        self._load_model()

    # ── private helpers ──────────────────────

    def _detect_model_type(self, name: str) -> str:
        if "bart" in name.lower():
            return "bart"
        elif "t5" in name.lower():
            return "t5"
        else:
            raise ValueError(
                f"Unknown model type for '{name}'. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

    def _load_model(self):
        if self.model_type == "bart":
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model     = BartForConditionalGeneration.from_pretrained(self.model_name)
        elif self.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()  # disable dropout for inference
        print(f"[INFO] Model loaded successfully ✓\n")

    # ── public API ───────────────────────────

    def tokenize(self, text: str, max_input_length: int = 1024) -> dict:
        """
        Tokenize input text and return encoded tensors.

        Args:
            text            (str): Preprocessed input text.
            max_input_length(int): Maximum token length (model-specific limit).

        Returns:
            dict: Tokenized output with input_ids and attention_mask.
        """
        # T5 expects a task prefix
        if self.model_type == "t5":
            text = "summarize: " + text

        inputs = self.tokenizer(
            text,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def summarize(
        self,
        text: str,
        max_input_length: int  = 1024,
        min_summary_length: int = 60,
        max_summary_length: int = 200,
        num_beams: int          = 4,
        length_penalty: float   = 2.0,
        early_stopping: bool    = True,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        """
        Generate a concise summary for the given text.

        Args:
            text               (str)  : Input article / document.
            max_input_length   (int)  : Truncation length for input tokens.
            min_summary_length (int)  : Minimum tokens in the output summary.
            max_summary_length (int)  : Maximum tokens in the output summary.
            num_beams          (int)  : Beam search width (higher = better quality, slower).
            length_penalty     (float): > 1.0 encourages longer summaries.
            early_stopping     (bool) : Stop when all beams reach EOS.
            no_repeat_ngram_size(int) : Prevents repetition of n-grams.

        Returns:
            str: Generated summary string.
        """
        # Preprocess + tokenize
        clean_text = preprocess_text(text)
        inputs     = self.tokenize(clean_text, max_input_length)

        # Inference (no gradient tracking needed)
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                min_length     = min_summary_length,
                max_length     = max_summary_length,
                num_beams      = num_beams,
                length_penalty = length_penalty,
                early_stopping = early_stopping,
                no_repeat_ngram_size = no_repeat_ngram_size,
            )

        # Decode token IDs → human-readable string
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return summary

    def batch_summarize(self, texts: list, **kwargs) -> list:
        """
        Summarize a list of texts (useful for bulk processing).

        Args:
            texts (list[str]): List of documents to summarize.
            **kwargs         : Forwarded to `summarize()`.

        Returns:
            list[str]: List of generated summaries.
        """
        return [self.summarize(t, **kwargs) for t in texts]


# ─────────────────────────────────────────────
#  Pipeline Shortcut (HuggingFace high-level)
# ─────────────────────────────────────────────

def quick_summarize(text: str, model_name: str = "facebook/bart-large-cnn") -> str:
    """
    One-liner summarization using HuggingFace pipeline API.
    Great for quick demos; use SummarizerModel for full control.

    Args:
        text       (str): Input text.
        model_name (str): Model to use.

    Returns:
        str: Summary string.
    """
    summarizer = pipeline("summarization", model=model_name)
    result     = summarizer(text, max_length=200, min_length=60, do_sample=False)
    return result[0]["summary_text"]
