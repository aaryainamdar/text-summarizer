# 🧠 NLP Text Summarizer — BART & T5

> Abstractive text summarization using pretrained Transformer models (BART, T5) from HuggingFace.  
> Built as a learning project to explore NLP pipelines, transfer learning, and sequence-to-sequence architectures.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.35%2B-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=flat-square)

---

## 📌 About the Project

This project implements an **abstractive text summarization pipeline** using state-of-the-art transformer models. Given a long article or document, the system generates a concise, human-readable summary — not by copy-pasting sentences, but by actually understanding and rephrasing the content.

I built this to get hands-on experience with:
- 🔤 **Natural Language Processing** (tokenization, text preprocessing)
- 🤗 **HuggingFace Transformers** library and the model hub
- 🧩 **Transfer Learning** — using pretrained weights for a new task
- ⚙️ **Sequence-to-Sequence architectures** (encoder-decoder models)
- 📊 **ROUGE evaluation** to measure summarization quality

---

## ✨ Features

- **Two model backends** — BART (`facebook/bart-large-cnn`) and T5 (`t5-base`, `t5-small`, etc.)
- **Full preprocessing pipeline** — URL removal, whitespace normalization, special character handling
- **Configurable inference** — control summary length, beam search width, repetition penalties
- **Batch summarization** — process multiple documents at once
- **ROUGE scoring** — evaluate output quality against reference summaries
- **CLI demo** — run directly from the terminal with flags

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| PyTorch | Deep learning backend |
| HuggingFace Transformers | Pretrained models & tokenizers |
| ROUGE Score | Evaluation metric |
| Pytest | Unit testing |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/text-summarizer.git
cd text-summarizer
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ PyTorch may need a specific install command depending on your CUDA version.  
> Check [pytorch.org](https://pytorch.org/get-started/locally/) for the right command.

---

## 💻 Usage

### Quick demo (built-in sample article)

```bash
python demo.py
```

### Summarize your own text file

```bash
python demo.py --file path/to/your/article.txt
```

### Switch to T5

```bash
python demo.py --model t5-base
```

### Tune summary length and beam width

```bash
python demo.py --min_length 80 --max_length 300 --num_beams 6
```

### Run with ROUGE evaluation

```bash
python demo.py --evaluate
```

---

## 🐍 Use as a Python Module

```python
from src.summarizer import SummarizerModel

# Load the model (downloads weights on first run)
model = SummarizerModel("facebook/bart-large-cnn")

article = """
Artificial intelligence has transformed the way we interact with technology...
[your long article here]
"""

# Generate summary
summary = model.summarize(
    article,
    min_summary_length=60,
    max_summary_length=200,
    num_beams=4,
)

print(summary)
```

**Batch processing:**

```python
articles = [article1, article2, article3]
summaries = model.batch_summarize(articles, max_summary_length=150)
```

**Quick one-liner (uses HuggingFace pipeline API):**

```python
from src.summarizer import quick_summarize
summary = quick_summarize("Your text here...")
```

---

## 📊 Sample Output

**Input** (~350 words on AI and NLP):

> *"Artificial intelligence has transformed the way we interact with technology, enabling machines to perform tasks that once required human intelligence..."*

**Generated Summary** (BART, ~60 words):

> *"Large language models like GPT, BERT, and T5 have revolutionized NLP through self-supervised training. BART and T5 excel at abstractive summarization due to their encoder-decoder architecture. Responsible development of AI is critical to address bias and societal implications."*

**ROUGE Scores:**

```
====================================================
Metric        Precision     Recall         F1
----------------------------------------------------
rouge1           0.5714       0.5714     0.5714
rouge2           0.3077       0.3077     0.3077
rougeL           0.5714       0.5714     0.5714
====================================================
```

---

## 🗂️ Project Structure

```
text-summarizer/
│
├── src/
│   ├── __init__.py         # Package init & exports
│   ├── summarizer.py       # Core model class + preprocessing
│   └── evaluate.py         # ROUGE scoring utilities
│
├── tests/
│   └── test_summarizer.py  # Unit tests (pytest)
│
├── notebooks/
│   └── exploration.ipynb   # Step-by-step walkthrough (Jupyter)
│
├── demo.py                 # CLI demo script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_summarizer.py::TestPreprocessText::test_strips_leading_trailing_whitespace  PASSED
tests/test_summarizer.py::TestPreprocessText::test_collapses_multiple_spaces           PASSED
tests/test_summarizer.py::TestPreprocessText::test_removes_newlines                    PASSED
tests/test_summarizer.py::TestPreprocessText::test_removes_urls                        PASSED
tests/test_summarizer.py::TestPreprocessText::test_empty_string                        PASSED
tests/test_summarizer.py::TestPreprocessText::test_preserves_basic_punctuation         PASSED
tests/test_summarizer.py::TestComputeRouge::test_perfect_match                         PASSED
tests/test_summarizer.py::TestComputeRouge::test_empty_overlap                         PASSED
tests/test_summarizer.py::TestComputeRouge::test_returns_all_rouge_types               PASSED
tests/test_summarizer.py::TestComputeRouge::test_scores_between_zero_and_one           PASSED
tests/test_summarizer.py::TestComputeRouge::test_batch_averaging                       PASSED
```

---

## 📚 What I Learned

This project was my first deep dive into real NLP. Key takeaways:

- **Transformer architectures** are really encoder-decoder networks where the encoder reads the full input and the decoder generates output token by token.
- **BART vs T5**: BART is pre-trained with a denoising objective (great for generation); T5 treats everything as text-to-text (more flexible but needs a task prefix like `"summarize: "`).
- **Beam search** trades speed for quality — a higher beam width explores more candidate sequences.
- **ROUGE** measures n-gram overlap and is the standard evaluation metric for summarization, though it doesn't capture semantic similarity perfectly.
- **Transfer learning** is incredibly powerful — a model pre-trained on millions of articles can be used out of the box for summarization with zero fine-tuning.

---

## 🔭 Future Plans

- [ ] Fine-tune on CNN/DailyMail dataset for better domain-specific performance
- [ ] Add a Gradio / Streamlit web interface
- [ ] Experiment with extractive methods (TextRank, BERTSum) for comparison
- [ ] Add support for PDF and web URL inputs
- [ ] Containerize with Docker for easy deployment

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) — for making state-of-the-art NLP accessible
- [BART Paper](https://arxiv.org/abs/1910.13461) — Lewis et al., 2019
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Raffel et al., 2019
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail) — the standard summarization benchmark

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ and a lot of Stack Overflow</p>
