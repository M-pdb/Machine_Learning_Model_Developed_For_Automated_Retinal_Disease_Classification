# Machine_Learning_Model_Developed_For_Automated_Retinal_Disease_Classification

This repository presents a single-notebook project that classifies retinal images into Normal, Cataract, and Glaucoma using a deep-learning pipeline. All code and outputs (figures, metrics) are embedded in the notebook

---

## Directory Structure

```
Scripts/
  7BBG1005_Coursework.ipynb
```

---

## Project Overview

- **Goal**: Assist screening by predicting one of three classes from a retinal image.
- **Approach:** Transfer learning with a CNN backbone (fine-tuned) + a small classification head.
- **Image size:** 224×224 RGB (resized on the fly).
- **Training:** Stratified K-Fold cross-validation, early stopping, learning-rate scheduling.
- **Imbalance handling:** class weights and/or focal loss.
- **Augmentation:** rotation, shear, zoom, brightness shift, horizontal flips.
- **Reporting:** accuracy, precision/recall/F1, confusion matrices, and ROC-AUC per class.
- **Outputs:** All plots and tables are shown inside 7BBG1005_Coursework.ipynb.

---

## Requirements

- **Python 3.10+**
- **Jupyter Notebook** (via Anaconda or pip install jupyter) Any additional libraries used are imported directly in the notebook

---

## Pipeline Workflow

- **Load & Label** images from the three class folders.
- **Split** data with Stratified K-Fold (train/val per fold).
- **Preprocess & Augment** (resize, rescale, geometric/photometric transforms).
- **Model Build:** load pre-trained backbone, add small head, softmax(3).
- **Train** with class weights/focal loss; callbacks: EarlyStopping + LR scheduler.
- **Evaluate** per fold; compute metrics; render confusion matrices & ROC curves.
- **Summarise** cross-fold performance; display aggregated tables in-notebook.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
