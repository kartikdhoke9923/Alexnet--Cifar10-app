# AlexNet CIFAR-10 Image Classifier

A browser-based image classification app built with a custom AlexNet CNN trained on CIFAR-10, converted to TensorFlow.js for **zero-latency, server-free inference** directly in the browser.

---

## Live Demo

> [Try it here](https://kartikdhoke9923.github.io/Alexnet--Cifar10-app/)

---

## What It Does

Upload any image and the model predicts which of the **10 CIFAR-10 classes** it belongs to, along with confidence scores for all top-5 predictions — all running **100% in your browser**, no server required.

| Class | Class | Class | Class | Class |
|---|---|---|---|---|
| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
| 🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚛 Truck |

---

## Project Structure

```
Alexnet--Cifar10-app/
│
├── index.html              ← Full app (HTML + CSS + JS, single file)
├── convert.py              ← Script to convert trained model to TF.js format
│
├── tfjs_model/             ← TensorFlow.js model files
│   ├── model.json          ← Model architecture
│   ├── group1-shard1of8.bin
│   ├── group1-shard2of8.bin
│   ├── ...
│   └── group1-shard8of8.bin
│
└── .gitignore
```

---

## Model Architecture

Custom **AlexNet** adapted for CIFAR-10's 32×32 input size:

```
Input (32×32×3)
│
├── Conv2D(96, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
├── Conv2D(256, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
├── Conv2D(384, 3×3) → ReLU
├── Conv2D(384, 3×3) → ReLU
├── Conv2D(256, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
│
├── Flatten
├── Dense(1024) → ReLU → Dropout(0.5)
└── Dense(10) → Softmax
```

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Input Shape | 32 × 32 × 3 |
| Output Classes | 10 |
| Training Accuracy | ~92% |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Epochs | 10 |

---

## How It Works

```
User uploads image
        ↓
Browser center-crops to square
        ↓
Resizes to 32×32 via Canvas API
        ↓
Normalises pixel values to [0, 1]
        ↓
TensorFlow.js runs inference (WebGL accelerated)
        ↓
Top-5 predictions displayed with confidence bars
```

No Flask. No Python. No server. Everything runs on the client side.

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/kartikdhoke9923/Alexnet--Cifar10-app.git
cd Alexnet--Cifar10-app
```

### 2. Start a local server
```bash
python -m http.server 8000
```

### 3. Open in browser
```
http://localhost:8000
```

> Must use a local server — opening `index.html` directly via `file://` will block model loading due to browser CORS restrictions.

---

## 🔁 Retrain & Reconvert (Optional)

If you want to retrain the model from scratch:

### Step 1 — Set up environment
```bash
conda create -p venv python==3.10 -y
conda activate venv/
pip install tensorflow==2.15.0
pip install tensorflowjs==4.8.0
```

### Step 2 — Train and convert
```bash
python convert.py
```

This will:
- Load CIFAR-10 dataset automatically
- Train AlexNet for 10 epochs
- Save model as `model_v2.h5`
- Convert and export to `tfjs_model/`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model Training | TensorFlow / Keras 2.15 |
| Browser Inference | TensorFlow.js 4.10 |
| Frontend | HTML5 · CSS3 · Vanilla JS |
| Preprocessing | Canvas API |
| Hosting | GitHub Pages |

---

## Training Results

```
Epoch 10/10
782/782 ━━━━━━━━━━━━━━━━━━━━ 297s
loss: 0.2297 · accuracy: 0.9205
val_loss: 0.8357 · val_accuracy: 0.7926
```

---

## Acknowledgements

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) — Alex Krizhevsky, University of Toronto
- [TensorFlow.js](https://www.tensorflow.org/js) — for enabling in-browser ML inference
- Original AlexNet paper — *ImageNet Classification with Deep Convolutional Neural Networks* (Krizhevsky et al., 2012)

---

## Author

**Kartik Dhoke**
- GitHub: [@kartikdhoke9923](https://github.com/kartikdhoke9923)
- Location: Pune, India

---

## License

This project is open source and available under the [MIT License](LICENSE).
