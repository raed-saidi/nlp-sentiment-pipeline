# 📘 NLP Pipeline: Sentiment Analysis & Named Entity Recognition

### Fine-tuned Transformer Pipeline for IMDB Sentiment Classification & WikiANN NER

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Pipeline-0056b3?style=for-the-badge" alt="NLP Pipeline"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Datasets-HuggingFace-FF6F00?style=for-the-badge&logo=huggingface" alt="Datasets"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/GPU-Ready-6FDA44?style=for-the-badge&logo=nvidia" alt="GPU Ready"/>
</p>

---

## 🎯 Overview

This repository contains a comprehensive **Jupyter Notebook** implementing a complete NLP workflow with two state-of-the-art deep learning models:

✅ **Sentiment Analysis** — DistilBERT fine-tuned on IMDB movie reviews  
✅ **Named Entity Recognition (NER)** — BERT-Large fine-tuned on WikiANN dataset

The notebook includes end-to-end functionality: dataset preparation, tokenization, model training, evaluation, visualization, and inference utilities — all fully reproducible on **Kaggle** or **Google Colab**.

---

## 🚀 Features

- 🔥 **Pre-trained Transformers** — Leverages DistilBERT and BERT-Large from Hugging Face
- 📊 **Comprehensive EDA** — Token distribution, entity frequency, sequence length analysis
- 🎨 **Rich Visualizations** — Training curves, confusion matrices, dataset statistics
- 💾 **Checkpoint Management** — Automatic saving after each epoch with custom callbacks
- 🔄 **Reproducible** — Seed management for deterministic results
- ⚡ **GPU Accelerated** — Mixed precision training (FP16) support
- 📈 **Full Evaluation Suite** — Classification reports, SeqEval metrics, confidence scores

---

## 🧱 Project Structure

The notebook is organized into logical sections:

### 1. 🔧 Setup & Configuration

- Imports all required dependencies (Transformers, Datasets, PyTorch, Evaluate, Matplotlib, Seaborn)
- Auto-detects GPU (CUDA) availability
- Creates training and output directories
- Defines `set_seed()` function for reproducibility

### 2. 🧩 Utilities & Callbacks

**SaveEveryEpochCallback**
- Custom Hugging Face Trainer callback
- Saves checkpoints after each epoch
- Exports model, tokenizer, and training logs
- Ensures training continuity

**Training Visualization**
- `plot_training_history()` — Generates accuracy & loss curves
- `load_training_logs()` — Loads JSON logs into Pandas DataFrame

**Evaluation Utilities**
- `evaluate_sentiment_model()` — Returns predictions with confidence scores

### 3. 🎬 Sentiment Analysis (IMDB)

#### ✔ Dataset Preparation
- Loads IMDB movie review dataset
- Tokenizes text using DistilBERT tokenizer
- Converts to PyTorch tensors
- Saves processed dataset for reusability

#### ✔ Exploratory Data Analysis
- Token length distribution visualization
- Mean & median sequence length analysis
- Saves figure: `imdb_sequence_lengths.png`

#### ✔ Training
- Fine-tunes DistilBERT for binary classification
- Mixed precision training (FP16)
- Epoch-level logging with automatic saving
- Loads best model by accuracy

#### ✔ Evaluation
- Full classification report (precision, recall, F1)
- Confusion matrix heatmap → `sentiment_confusion_matrix.png`
- Training/validation curves
- Predictions exported to `imdb_predictions.csv`

### 4. 🏷️ Named Entity Recognition (WikiANN)

#### ✔ Dataset Preparation
- Loads WikiANN English dataset
- Tokenizes with BERT-Large NER tokenizer
- Aligns labels with wordpiece tokens
- Handles subword tokenization with `-100` labels
- Saves tokenized dataset in Arrow format

#### ✔ Exploratory Data Analysis
Four-panel visualization including:
- Sequence length distribution
- Entity frequency distribution
- Entities per sequence
- Train/validation/test split sizes

Saved as: `wikiann_ner_eda.png`

#### ✔ Training
- Freezes lower BERT layers for efficiency
- Fine-tunes top encoder layers (8–11)
- SeqEval metrics: precision, recall, F1
- Token classification data collator
- Saves best model checkpoint

### 5. 🧪 Inference & Testing

Example predictions for:
- Movie reviews → sentiment classification
- News sentences → entity extraction
- General text → combined analysis

**Sample Output:**
```json
{
  "text": "Apple released the new iPhone in California.",
  "sentiment": "positive",
  "confidence": 0.97,
  "entities": [
    {"text": "Apple", "type": "ORG"},
    {"text": "iPhone", "type": "PRODUCT"},
    {"text": "California", "type": "LOC"}
  ]
}
```

---

## 📊 Generated Artifacts

| File / Folder | Description |
|--------------|-------------|
| `imdb_predictions.csv` | All sentiment predictions with confidence scores |
| `sentiment_confusion_matrix.png` | Confusion matrix heatmap |
| `imdb_sequence_lengths.png` | Token length distribution visualization |
| `wikiann_ner_eda.png` | Four-panel NER dataset analysis |
| `train_log_history.json` | Hugging Face Trainer logs |
| `training_logs.csv` | Cleaned CSV format training logs |
| `/kaggle/working/my_sentiment_model/` | Saved DistilBERT sentiment model |
| `/kaggle/working/ner_model_final/` | Saved BERT-Large NER model |

---

## 🚀 How to Run

### Option A — Google Colab

1. Upload the `.ipynb` file to Google Colab
2. Enable GPU runtime: `Runtime → Change runtime type → GPU`
3. Run all cells sequentially
4. Models will automatically download, train, and save checkpoints

### Option B — Kaggle

1. Create a new Kaggle notebook
2. Upload the `.ipynb` file
3. Enable GPU accelerator (T4 or P100)
4. Add required datasets if needed
5. Run all cells

### Option C — Local Jupyter

```bash
git clone <repository-url>
cd nlp-pipeline
jupyter notebook
# Open the notebook and run all cells
```

---

## 📈 Model Performance

### Sentiment Analysis (IMDB)
- **Model:** DistilBERT (distilbert-base-uncased)
- **Accuracy:** ~92% on test set
- **Training Time:** ~30 minutes on GPU

### Named Entity Recognition (WikiANN)
- **Model:** BERT-Large (dslim/bert-large-NER)
- **F1 Score:** ~90% on test set
- **Entity Types:** PER, ORG, LOC, MISC
- **Training Time:** ~45 minutes on GPU

---

## 🔮 Future Improvements

Potential enhancements for contributors:

- [ ] Add inference API (FastAPI / Gradio interface)
- [ ] Combine tasks into unified pipeline (NER + sentiment per entity)
- [ ] Create interactive dashboard for error analysis
- [ ] Train domain-specific NER models
- [ ] Add support for multilingual models
- [ ] Implement model distillation for deployment
- [ ] Create Docker container for easy deployment
- [ ] Add CI/CD pipeline for automated testing

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Raed SAIDI**
- GitHub: https://github.com/raed-saidi
- LinkedIn: https://www.linkedin.com/in/saidi-raed-a368022a1/

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers library
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for sentiment analysis
- [WikiANN Dataset](https://huggingface.co/datasets/wikiann) for NER training
- The open-source NLP community

---

<p align="center">Made with ❤️ </p>
