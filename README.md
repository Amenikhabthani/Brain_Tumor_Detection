# Brain Tumor Detection using CNN

## Project Overview
This project employs a Convolutional Neural Network (CNN) to classify brain MRI scans into two categories:
- **Tumor Detected (yes)**
- **No Tumor Detected (no)**

The model is trained to recognize patterns in MRI images for automated diagnosis.

## Dataset
- **Source**: MRI images from Google Drive (`/content/drive/MyDrive/brain_mri_detection`).
- **Classes**:
  - `yes`: 86 images (tumor present).
  - `no`: 85 images (no tumor).
- **Image Size**: Varied dimensions (max 1275x1427), resized to `200x200` for model input.

## Preprocessing
- **Resizing**: All images standardized to `200x200` pixels.
- **Normalization**: Pixel values scaled to `[0, 1]`.
- **Data Augmentation**: Applied to reduce overfitting (rotation, shifts, flips).
- **Train-Val-Test Split**:
  - `80%` training
  - `10%` validation
  - `10%` testing

## Model Architecture (CNN)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
### Hyperparameters Tested:
- Dropout (`0.3–0.5`), Batch Normalization
- Optimizers: Adam, RMSprop, SGD
- **Best configuration**: Adam optimizer + `0.5` Dropout + Batch Normalization

## Training
- **Epochs**: `30` (early stopping with `3-epoch` patience)
- **Batch Size**: `16`
- **Callbacks**: Model checkpointing and early stopping
- **Augmentation**: Applied to training data for generalization

## Evaluation
### Best CNN Model
- **Test Accuracy**: `92%` (varies by hyperparameters)
- **Confusion Matrix**:
```python
[[8  2]
 [0 12]]  # Example for 20 test samples
```
- **Classification Report**:
```plaintext
            Precision  Recall  F1-Score
Tumor (yes)    1.00      0.89     0.94
No Tumor (no)  0.86      1.00     0.92
```
### SVM Baseline
- **Linear Kernel**: `88%` accuracy
- **RBF Kernel (GridSearch-tuned)**: `90%` accuracy

## Usage
### Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
### Install Dependencies:
```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python
```
### Run the Code:
1. Update dataset paths in the code.
2. Execute cells sequentially for preprocessing, training, and evaluation.

## Key Findings
- CNNs outperformed SVM (`92%` vs `90%` accuracy).
- Data augmentation and dropout significantly reduced overfitting.
- Hyperparameter tuning improved model robustness.

## Conclusion
This CNN-based approach effectively detects brain tumors in MRI scans, demonstrating the potential for automated medical diagnostics. Future work could explore **transfer learning** (e.g., ResNet, VGG) and **larger datasets** for improved generalization.

