# DeepFake Image Classifier 🔍🧠

This project aims to classify **DeepFake images** using two distinct approaches:
1. **Support Vector Machine (SVM)** with **Histogram of Oriented Gradients (HOG)** features.
2. A **Convolutional Neural Network (CNN)** based deep learning model using PyTorch.

---

## 📁 Dataset Structure
The dataset is organized into three folders:
- `train/` - Training images and `train.csv` containing image IDs and labels.
- `validation/` - Validation set with `validation.csv`.
- `test/` - Test set with `test.csv`, no labels.

All images are PNG files named using their `image_id` (e.g., `012abc.png`).

---

## 🧠 CNN Architecture Highlights

- **Input Size**: 224×224×3 (standardized to ImageNet size)
- **Preprocessing**:
  ```python
  transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
  ])
  ```
  > This normalization centers pixel values to zero-mean and unit-variance based on ImageNet stats.

- **Architecture**:
  - Multiple convolutional layers with ReLU activations.
  - MaxPooling to downsample.
  - Dropout layers for regularization.
  - Fully connected layers for final classification.
  - Trained using `CrossEntropyLoss` and `Adam` optimizer.

---

## 💡 Classical ML: SVM + HOG

- Extracted HOG features using:
  ```python
  HOG_PARAMS = {
      "orientations": 9,
      "pixels_per_cell": (8, 8),
      "cells_per_block": (2, 2),
      "block_norm": "L2-Hys",
      "transform_sqrt": True
  }
  ```

- Applied `StandardScaler` before training.

- Hyperparameter tuning for `C` parameter was done, with performance plotted across values.

- Comparison between `LinearSVC` and `SVC(kernel='rbf')` was made and plotted.

---

## 📊 Evaluation

- Accuracy and classification reports were generated for multiple `C` values.
- Confusion matrices used for in-depth error analysis.
- Visualization of accuracy trends was done using matplotlib.

---

## 📦 Output

- Final predictions are exported in `submission.csv` with columns:
  ```csv
  image_id,label
  ```

- Trained models and scalers are saved to `models/` directory as `.pkl` files.

---

## 🚀 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- scikit-learn
- matplotlib
- pandas
- numpy
