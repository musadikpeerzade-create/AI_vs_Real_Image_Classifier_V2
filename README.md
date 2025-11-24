# 🧠 AI-Generated vs Real Image Classifier – V2.1 (96.34% Accuracy)

This project builds a deep learning model that classifies whether an image is *AI-generated* or *real, using a custom Convolutional Neural Network (CNN) built in **PyTorch*.  
It includes a full web UI interface built with *Flask*, allowing real-time predictions directly from a browser.

---

### ⭐ Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | — (varies per run, optimized with scheduler) |
| *Validation Accuracy* | *96.34%* |
| *Test Accuracy* | *96.16%* |

A fine-tuning phase with a reduced learning rate helped push the model beyond the initial 95% accuracy threshold.

---

### 🔍 Why This Project?

AI-generated images have become increasingly realistic. Detecting whether an image is human-captured or AI-constructed is now a relevant challenge in:

- misinformation and fake media detection  
- forensic analysis  
- content authenticity verification  
- academic and industrial AI safety research  

This project serves as a step toward automated detection systems.

---

### 📦 Dataset

This model was trained using the *CIFAKE dataset*, which contains labeled real vs AI-generated images compressed to 32×32 resolution.

📌 Dataset credit:  
*CIFAKE — A Benchmark Dataset for AI-Generated vs Real Image Detection*  
Kaggle link: https://www.kaggle.com/datasets/mohamedhanyyy/cifake

Since CIFAKE images are 32×32, the prediction pipeline includes resolution alignment:

> *Images are first downsampled to 32×32, then upscale-aligned to 128×128 for inference.*

This significantly improves real-world prediction reliability.

---

### 🧠 Model Architecture

The model includes:

- 4 convolution blocks  
- Batch Normalization  
- Max Pooling  
- Global Average Pooling  
- Dropout-regularized fully connected layers  

This design achieves stronger generalization than V1, which peaked at 92.83%.

---

### 🧪 Technologies Used

| Area | Tool |
|------|------|
| Deep Learning Framework | PyTorch |
| Dataset transforms | TorchVision |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | Pillow |
| UX enhancements | Progress feedback + confidence breakdown |

---

### 🚀 Running the Project

Clone the repository:

```sh
git clone <your-repo-link>
cd AI_vs_Real_Classifier
```

Create a virtual environment:

```sh
python -m venv .venv
.venv\Scripts\activate  # Windows
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Run the app:

```sh
python app.py
```

Open the browser:

http://127.0.0.1:5000

---
👥 Authors & Credits

Developed by: Musaddik Peerzade,
AI Assistance: ChatGPT 5.1,
Human intuition + machine intelligence.

---
