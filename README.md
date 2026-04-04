# 🕵️ Deepfake & AI-Generated Face Detection

A deep learning system that detects deepfake and AI-generated facial images using a fine-tuned **Xception CNN** model trained on 100,000 images.

## 🏆 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **99%** |
| AUC-ROC Score | **0.9992** |
| F1-Score | **0.99** |
| Precision (Fake) | 0.99 |
| Recall (Fake) | 0.98 |

## 🏗️ Model

- **Architecture:** Xception (Transfer Learning from ImageNet)
- **Fine-tuned:** Top 20 layers unfrozen
- **Dataset:** 140k Real and Fake Faces (Kaggle)
- **Training:** Kaggle Notebooks — Tesla T4 x2 GPU
- **Epochs:** 13 (early stopping)
- **Optimizer:** Adam (lr=1e-4)

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/DetectFake.git
cd DetectFake

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

> On first run, the model (~100MB) downloads automatically from Google Drive.

## 📁 Structure

```
DetectFake/
├── app.py            # Streamlit web application
├── requirements.txt  # Dependencies
└── README.md
```

## 🛠️ Tech Stack

- Python, TensorFlow/Keras, Xception CNN
- Streamlit, OpenCV, NumPy, PIL
- Trained on Kaggle (Tesla T4 GPU)

## 📚 Honours Project

Built as part of a Bachelor of Engineering (Honours) in Computer Science, 2025-2026.
