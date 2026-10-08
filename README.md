# 🐾 Animal Image Classification with Deep Learning

A deep learning project that classifies animal images using **Convolutional Neural Networks (CNN)**. Two approaches are compared:

1. **Custom CNN** – a convolutional network built and trained from scratch.
2. **MobileNet (Transfer Learning)** – a lightweight model pre-trained on ImageNet, fine-tuned on the animal dataset.

The goal is to see how a model trained from scratch compares to transfer learning in accuracy, training time and model size.

---

## 📂 Project Structure

<!-- TODO: update the file names below to match the repo exactly -->

```
DeepLearning/
├── dataset/
│   ├── train/
│   │   ├── cat/
│   │   ├── dog/
│   │   └── ...          # one folder per animal class
│   └── test/
│       ├── cat/
│       ├── dog/
│       └── ...
├── animal_classification.ipynb   # main notebook (training + evaluation)
├── requirements.txt
└── README.md
```

> Images are organised **one folder per class**. The folder name is used as the label.

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- NumPy, Matplotlib
- scikit-learn (evaluation metrics)
- Jupyter Notebook / Google Colab

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/nialhaikal/DeepLearning.git
cd DeepLearning
```

### 2. Create a virtual environment (recommended)

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt`, install manually:

```bash
pip install tensorflow numpy matplotlib scikit-learn pillow jupyter
```

---

## 🚀 How to Run

### Option A — Run locally with Jupyter

```bash
jupyter notebook
```

Then open `animal_classification.ipynb` and run the cells from top to bottom (**Kernel → Restart & Run All**).

### Option B — Run on Google Colab (no setup, free GPU)

1. Go to [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** and select the `.ipynb` file.
2. Enable GPU: **Runtime → Change runtime type → GPU**.
3. Upload the dataset to Google Drive and mount it:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update the dataset path in the notebook, e.g.:
   ```python
   DATASET_DIR = '/content/drive/MyDrive/DeepLearning/dataset'
   ```
5. Run all cells.

---

## 🧠 Model Overview

### Custom CNN
- Stacked `Conv2D` + `MaxPooling2D` layers for feature extraction
- `Flatten` → `Dense` layers with `Dropout` to reduce overfitting
- `Softmax` output layer (one neuron per class)

### MobileNet (Transfer Learning)
- `MobileNetV2` base pre-trained on ImageNet (`include_top=False`)
- Base layers frozen, new classification head added (`GlobalAveragePooling2D` → `Dense` → `Softmax`)
- Optional fine-tuning of the top layers with a low learning rate

### Training setup
| Setting | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Augmentation | Rotation, flip, zoom (via `ImageDataGenerator`) |

<!-- TODO: adjust the table to match the actual settings used -->

---

## 📊 Results

<!-- TODO: fill in your actual numbers and add screenshots of the accuracy/loss plots -->

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Custom CNN | xx% | xx% |
| MobileNet | xx% | xx% |

The notebook also outputs:
- Training vs validation accuracy and loss curves
- Confusion matrix
- Classification report (precision, recall, F1-score)

---

## 🔍 Predict on Your Own Image

After training, use the model to classify a new image:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('path/to/your_image.jpg', target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print('Predicted class:', class_names[np.argmax(pred)])
```

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Make sure the virtual environment is activated, then `pip install tensorflow` |
| Training is very slow | Use Google Colab with GPU enabled |
| `FileNotFoundError` for dataset | Check that `DATASET_DIR` points to the correct folder |
| Out of memory | Lower the `batch_size` (e.g. 16 or 8) |

---

## 👤 Author

**Danial Haikal**
Bachelor of IT (Hons.) in Software Engineering — UniKL MIIT

- GitHub: [@nialhaikal](https://github.com/nialhaikal)
- LinkedIn: [Danial Haikal](https://linkedin.com/in/danial-haikal-125208346)
- Portfolio: [nialhaikalportfolio.vercel.app](https://nialhaikalportfolio.vercel.app)
