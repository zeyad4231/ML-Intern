# Machine Learning Internship — Elevvo Pathways

Projects completed during my Machine Learning internship at **Elevvo Pathways** (Jul – Aug 2025). They cover regression, classification, audio and image tasks, from data cleaning through model comparison.

| # | Task | Problem type | Methods | Best result |
|---|---|---|---|---|
| 1 | [Student Score Prediction](Task1.ipynb) | Regression | Linear vs. polynomial regression, feature analysis | Model comparison with visualizations |
| 4 | [Loan Approval Prediction](Task4.ipynb) | Binary classification | Logistic regression vs. decision tree, SMOTE for class imbalance | **97.1% accuracy**, F1 0.96 (decision tree) |
| 6 | [Music Genre Classification](Task6.ipynb) | 10-class audio classification | MFCC feature extraction (librosa); dense network on tabular MFCCs vs. CNN on MFCC "images" | 49.5% test accuracy (dense network) |
| 8 | [Traffic Sign Classification](Task8.ipynb) | Multi-class image classification (GTSRB) | Custom CNN with data augmentation vs. transfer learning with MobileNetV2 | **97.6% test accuracy** (MobileNetV2) |

## Highlights

- **Traffic signs (GTSRB):** a custom CNN reached 93.7% test accuracy; transfer learning with MobileNetV2 raised it to 97.6%.
- **Loan approval:** handled class imbalance with SMOTE and showed the decision tree outperforming logistic regression on every metric (78.5% → 97.1% accuracy).
- **Music genres (GTZAN):** compared two ways of representing MFCC audio features. The dense network on tabular MFCCs (49.5%) beat a small CNN on MFCC images (42.3%); the notebook discusses why and what to try next.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · TensorFlow / Keras · librosa · OpenCV · Matplotlib · Seaborn · Google Colab (GPU)

## Running the notebooks

Each notebook is self-contained and was developed in Google Colab. Datasets are downloaded from Kaggle inside the notebooks, so a Kaggle API token (`kaggle.json`) is needed for the image and audio tasks.
