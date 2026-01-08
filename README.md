# 🧠 Brain Tumor Detection using Deep Learning (ResNet50 + Grad-CAM)

A deep learning–based brain tumor classification system built using **PyTorch** and **ResNet50**, capable of detecting and visualizing tumor regions in brain MRI scans using **Grad-CAM**.
This project supports **multi-class tumor classification** and provides **visual explainability**, making it suitable for research, academic projects, and portfolio demonstration.

---

## 🔍 Project Overview

This project performs **automatic brain tumor detection** from MRI images and classifies them into the following categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

In addition to prediction, the model uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight **important regions in the MRI scan**, improving transparency and interpretability—an essential requirement in medical AI systems.

---

## ✨ Key Features

* ✅ ResNet50-based CNN architecture
* ✅ Multi-class brain tumor classification
* ✅ Class imbalance handling using weighted loss
* ✅ Mixed precision training support (GPU / Apple MPS)
* ✅ Grad-CAM heatmap visualization
* ✅ Confidence scores for each class
* ✅ Supports CPU, CUDA GPU, and Apple Silicon (MPS)
* ✅ Models saved in multiple formats (`.h5`, `.keras`, `.legacy`)

---

## 🧠 Model Architecture

* **Backbone:** ResNet50 (pretrained on ImageNet)
* **Custom Classification Head:**

  * Dropout
  * Fully Connected Layer (512 units)
  * ReLU Activation
  * Batch Normalization
  * Dropout
  * Final Classification Layer

---

## 📂 Dataset Structure

The dataset follows a folder-based structure compatible with `torchvision.datasets.ImageFolder`:

```
brain_tumor_dataset/
├── Training/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No Tumor/
└── Testing/
    ├── Glioma/
    ├── Meningioma/
    ├── Pituitary/
    └── No Tumor/
```

> ⚠️ Dataset is **not included** in this repository due to licensing restrictions.

---

## 🚀 Training Pipeline

* Data Augmentation:

  * Random resized crops
  * Horizontal flips
  * Color jitter
* Loss Function:

  * **CrossEntropyLoss with class weights**
* Optimizer:

  * **AdamW**
* Learning Rate Scheduler:

  * StepLR
* Early Stopping:

  * Monitors validation loss
* Model Checkpointing:

  * Best model + last checkpoint saved automatically

---

## 💾 Saved Outputs

All outputs are stored in the `outputs/` directory:

```
outputs/
├── best_model.h5
├── best_model.keras
├── best_model.legacy
├── final_model.h5
├── final_model.keras
├── final_model.legacy
├── training_info.json
└── class_info.json
```

---

## 🔬 Grad-CAM Explainability

Grad-CAM highlights **important regions** in the MRI image that influenced the model’s prediction.
This improves:

* Model transparency
* Trust in predictions
* Clinical interpretability (for research use)

---

## 🖥️ Inference & Visualization

### Single Image Analysis

```python
result = analyze_brain_mri("/path/to/mri_image.jpg")
```

### Interactive Heatmap Intensities

```python
result = analyze_with_interactive_heatmap("/path/to/mri_image.jpg")
```

### Batch Analysis

```python
results = analyze_multiple_images("/path/to/image_folder")
```

---

## 📊 Output Details

For each MRI scan, the system provides:

* Predicted tumor class
* Confidence percentage
* Probability scores for all classes
* Grad-CAM heatmap
* Heatmap overlay on original MRI

---

## ⚠️ Medical Disclaimer

> **This project is intended for educational and research purposes only.**
> It is **NOT a medical diagnostic tool** and should **not be used for clinical decision-making**.
> Always consult a certified medical professional for diagnosis and treatment.

---

## 🛠️ Requirements

* Python 3.8+
* PyTorch
* Torchvision
* NumPy
* OpenCV
* Matplotlib
* Pillow

Install dependencies:

```bash
pip install torch torchvision numpy opencv-python matplotlib pillow
```

---

## 🌱 Future Improvements

* Tumor segmentation and localization
* Web or desktop deployment
* Uncertainty estimation
* Multi-modal MRI support
* Model optimization for edge devices

---

## 👨‍💻 Author

**Kshitij Verma**
B.Tech | AI / ML | Medical Imaging | Computer Vision

---

## ⭐ If you find this project useful, consider starring the repository!
