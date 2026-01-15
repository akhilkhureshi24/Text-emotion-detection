# 🧠 Emotion Detection from Textual Comments

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)
![Transformers](https://img.shields.io/badge/Transformers-4.36.0-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-success)

**Automated Detection of Different Emotions from Textual Comments using Transformer-based Deep Learning**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Results](#-results) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a **state-of-the-art emotion detection system** that can analyze textual comments and classify them into six distinct emotional categories. Built on transformer-based architecture (DistilBERT), it achieves **93.6% accuracy** on the test dataset.

### Supported Emotions

| Emotion | Description | Example |
|---------|-------------|---------|
| 😢 **Sadness** | Feelings of grief, sorrow, or disappointment | "I feel so lonely and lost" |
| 😊 **Joy** | Expressions of happiness and contentment | "This is the best day ever!" |
| ❤️ **Love** | Affection, care, and romantic feelings | "I absolutely adore you" |
| 😠 **Anger** | Frustration, irritation, and rage | "This is so infuriating!" |
| 😨 **Fear** | Anxiety, worry, and terror | "I'm terrified of what might happen" |
| 😲 **Surprise** | Astonishment and unexpected reactions | "Wow, I didn't see that coming!" |

---

## ✨ Features

### 🚀 Core Capabilities
- **Multi-class Emotion Classification**: Detects 6 distinct emotions with high accuracy
- **Real-time Inference**: Fast prediction pipeline for instant results
- **Robust Preprocessing**: Handles slang, abbreviations, URLs, and special characters
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and performance analysis
- **Production Ready**: Optimized for deployment on cloud platforms

### 🛠️ Technical Features
- Transformer-based architecture (DistilBERT)
- GPU acceleration support
- Batch processing capabilities
- Model checkpointing and versioning
- Extensive logging and monitoring
- Export to various formats (ONNX, TorchScript)

---

## 🎬 Demo

### Web Application
```bash
streamlit run emotion_app.py
```

### Quick Prediction Example
```python
from transformers import pipeline

# Load the model
detector = pipeline("text-classification", model="./emotion_model")

# Predict emotion
text = "I am feeling absolutely wonderful today!"
result = detector(text)
print(result)  # [{'label': 'joy', 'score': 0.9876}]
```

### Sample Outputs

| Input Text | Predicted Emotion | Confidence |
|------------|-------------------|------------|
| "I can't believe I lost everything" | Sadness | 98.7% |
| "This is the happiest day of my life!" | Joy | 99.4% |
| "You make my heart skip a beat" | Love | 96.3% |
| "This is absolutely unacceptable!" | Anger | 97.2% |
| "I'm terrified of the dark" | Fear | 98.9% |
| "What?! I never expected this!" | Surprise | 94.8% |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Text                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Preprocessing                              │
│  • Lowercasing                                              │
│  • URL/Mention Removal                                      │
│  • Slang Normalization                                      │
│  • Special Character Cleaning                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DistilBERT Tokenizer                           │
│  • Max Length: 128 tokens                                   │
│  • Padding & Truncation                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         DistilBERT Model (66M parameters)                   │
│  • 6 Transformer Layers                                     │
│  • 768 Hidden Dimensions                                    │
│  • 12 Attention Heads                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Classification Head                                │
│  • Dropout (0.1)                                            │
│  • Linear Layer (768 → 6)                                   │
│  • Softmax Activation                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Emotion Prediction                              │
│  • Class: [Sadness, Joy, Love, Anger, Fear, Surprise]      │
│  • Confidence Score                                         │
└─────────────────────────────────────────────────────────────┘
```

### Model Pipeline

1. **Data Loading** → Load emotion dataset from Hugging Face
2. **Preprocessing** → Clean and normalize text data
3. **Tokenization** → Convert text to token IDs
4. **Training** → Fine-tune DistilBERT on emotion data
5. **Evaluation** → Comprehensive testing and metrics
6. **Deployment** → Save model and create inference pipeline

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n emotion-env python=3.8
conda activate emotion-env

# Or using venv
python -m venv emotion-env
source emotion-env/bin/activate  # Linux/Mac
emotion-env\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model (Optional)

```bash
# Download the pre-trained model from releases
wget https://github.com/yourusername/emotion-detection/releases/download/v1.0/emotion_model.zip
unzip emotion_model.zip
```

---

## 💻 Usage

### Training from Scratch

```python
# Run the complete training pipeline
python train.py

# Or use the Jupyter notebook
jupyter notebook Text_emotion_detection.ipynb
```

### Inference on New Data

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load model
tokenizer = AutoTokenizer.from_pretrained("./emotion_model")
model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
detector = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Predict single text
text = "I'm feeling great today!"
result = detector(text)
print(f"Emotion: {result[0]['label']}, Confidence: {result[0]['score']:.4f}")

# Batch prediction
texts = [
    "I'm so happy!",
    "This is terrible",
    "I love this so much"
]
results = detector(texts)
for text, result in zip(texts, results):
    print(f"{text} → {result['label']} ({result['score']:.2%})")
```

### Launch Web Application

```bash
# Start Streamlit app
streamlit run emotion_app.py --server.port 8501

# Access at http://localhost:8501
```

### Command-Line Interface

```bash
# Predict emotion from command line
python predict.py --text "Your input text here"

# Batch prediction from file
python predict.py --file inputs.txt --output results.csv

# With confidence threshold
python predict.py --text "Your text" --threshold 0.8
```

---

## 🤖 Model Details

### Architecture Specifications

| Component | Details |
|-----------|---------|
| **Base Model** | DistilBERT (distilbert-base-uncased) |
| **Parameters** | 66,958,086 trainable parameters |
| **Input Size** | 128 tokens (max sequence length) |
| **Output Size** | 6 classes (emotions) |
| **Hidden Size** | 768 dimensions |
| **Attention Heads** | 12 heads per layer |
| **Transformer Layers** | 6 layers |
| **Dropout** | 0.1 |

### Training Configuration

```python
TRAINING_ARGS = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'optimizer': 'AdamW',
    'scheduler': 'linear',
}
```

### Hyperparameters

- **Learning Rate**: 2e-5 with linear warmup
- **Batch Size**: 16 (train), 16 (eval)
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Cross-Entropy Loss
- **Evaluation Metric**: Weighted F1-Score

---

## 📊 Dataset

### Source
- **Dataset**: [Emotion Dataset from Hugging Face](https://huggingface.co/datasets/emotion)
- **Size**: 20,000 samples
- **Languages**: English
- **License**: Apache 2.0

### Data Distribution

| Split | Samples | Percentage |
|-------|---------|------------|
| **Train** | 14,000 | 70% |
| **Validation** | 2,000 | 10% |
| **Test** | 4,000 | 20% |

### Emotion Distribution (Training Set)

```
Sadness  : ████████████████████████████ 29.0% (4,058)
Joy      : ████████████████████████████████ 33.8% (4,733)
Love     : ████████ 8.2% (1,149)
Anger    : █████████████ 13.5% (1,896)
Fear     : ███████████ 11.9% (1,661)
Surprise : ███ 3.6% (503)
```

### Data Preprocessing Steps

1. **Text Cleaning**
   - Lowercase conversion
   - URL removal
   - Mention (@username) removal
   - Hashtag (#) removal

2. **Slang Normalization**
   - u → you
   - ur → your
   - thx → thanks
   - gr8 → great
   - [+15 more mappings]

3. **Whitespace Normalization**
   - Remove extra spaces
   - Strip leading/trailing whitespace

---

## 📈 Results

### Overall Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.60% |
| **Weighted Precision** | 93.59% |
| **Weighted Recall** | 93.60% |
| **Weighted F1-Score** | 93.58% |
| **Average Confidence** | 96.55% |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Sadness** | 96.13% | 96.46% | 96.30% | 1,159 |
| **Joy** | 94.83% | 96.38% | 95.60% | 1,352 |
| **Love** | 88.49% | 82.01% | 85.13% | 328 |
| **Anger** | 93.30% | 92.44% | 92.86% | 542 |
| **Fear** | 91.86% | 90.32% | 91.08% | 475 |
| **Surprise** | 80.00% | 86.11% | 82.94% | 144 |

### Confusion Matrix

![Confusion Matrix](./results/confusion_matrix.png)

### Training Curves

- **Training Loss**: 0.111 (final epoch)
- **Validation Loss**: 0.162 (final epoch)
- **Validation F1**: 93.26% (final epoch)

### Edge Case Performance

| Scenario | Example | Prediction | Confidence |
|----------|---------|------------|------------|
| Negation | "I'm not happy" | Joy ❌ | 99.23% |
| Sarcasm | "Yeah, great... not!" | Joy ❌ | 99.42% |
| Mixed Emotions | "I love this but scared" | Fear ✅ | 99.64% |
| Slang | "r u gud?" | Anger ⚠️ | 56.49% |

---

## 📁 Project Structure

```
emotion-detection/
│
├── 📂 emotion_model/          # Trained model files
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
│
├── 📂 results/                # Evaluation results
│   ├── confusion_matrix.png
│   ├── final_results.json
│   └── training_logs.txt
│
├── 📂 logs/                   # Training logs
│   └── checkpoint-*/
│
├── 📂 data/                   # (Optional) Local data storage
│
├── 📄 Text_emotion_detection.ipynb  # Main notebook
├── 📄 emotion_app.py          # Streamlit web app
├── 📄 train.py                # Training script
├── 📄 predict.py              # Inference script
├── 📄 requirements.txt        # Dependencies
├── 📄 README.md               # This file
├── 📄 LICENSE                 # MIT License
└── 📄 .gitignore              # Git ignore rules
```

---

## 🔌 API Reference

### Python API

```python
from transformers import pipeline

# Initialize
detector = pipeline("text-classification", model="./emotion_model")

# Single prediction
result = detector("Your text here")
# Returns: [{'label': 'LABEL_1', 'score': 0.9876}]

# Batch prediction
results = detector(["Text 1", "Text 2", "Text 3"])

# With custom parameters
result = detector(
    "Your text",
    top_k=3,           # Return top 3 predictions
    truncation=True,   # Truncate long texts
    max_length=128     # Maximum sequence length
)
```

### REST API (Flask/FastAPI)

```python
# Example FastAPI endpoint
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
detector = pipeline("text-classification", model="./emotion_model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotion(input: TextInput):
    result = detector(input.text)[0]
    return {
        "emotion": result['label'],
        "confidence": result['score']
    }
```

---

## 🌐 Deployment

### Deploy on Google Colab

1. Upload notebook to Colab
2. Run all cells
3. Use ngrok for public URL:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8501)
print(f"App URL: {public_url}")
```

### Deploy on Hugging Face Spaces

```bash
# Create new Space on Hugging Face
# Upload these files:
# - emotion_app.py
# - requirements.txt
# - emotion_model/ (directory)
```

### Deploy on AWS/GCP/Azure

See [deployment guide](DEPLOYMENT.md) for detailed instructions.

### Docker Deployment

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "emotion_app.py"]
```

```bash
# Build and run
docker build -t emotion-detection .
docker run -p 8501:8501 emotion-detection
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size
BATCH_SIZE = 8  # Instead of 16
```

**Issue**: Slow training on CPU
```bash
# Solution: Use Google Colab with GPU
# Runtime → Change runtime type → GPU
```

**Issue**: Model not loading
```bash
# Solution: Check model directory exists
ls ./emotion_model/
# Should contain: config.json, model.safetensors, etc.
```

**Issue**: Streamlit app not starting
```bash
# Solution: Kill existing processes
pkill streamlit
streamlit run emotion_app.py --server.port 8502
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and emotion dataset
- **DistilBERT** authors for the efficient transformer model
- **Streamlit** for the amazing web framework
- **PyTorch** team for the deep learning framework
- All contributors and the open-source community

---

## 📧 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

**Project Link**: [https://github.com/yourusername/emotion-detection](https://github.com/yourusername/emotion-detection)

---

## 🔗 Useful Links

- [Hugging Face Model Hub](https://huggingface.co/models)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Emotion Dataset](https://huggingface.co/datasets/emotion)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Project Demo](https://your-demo-url.com)

---

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{emotion_detection_2024,
  author = {Your Name},
  title = {Emotion Detection from Textual Comments},
  year = {2024},
  url = {https://github.com/yourusername/emotion-detection}
}
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by [Your Name]

</div>