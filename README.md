# ROB5-S9-CardioSense_AI

*(English version below)*

## Structure du projet

Le projet est organisé en deux laboratoires distincts, chacun contenant un notebook Jupyter dédié :

```
│
├── Lab1/      → Détection de l’apnée du sommeil avec ECG + HRV + MLP
├── Lab2/      → Estimation de la pression sanguine (SBP/DBP) avec PPG + modèle hybride CNN–LSTM–MLP
└── README.md
```

Chaque dossier contient :

* le notebook principal (`.ipynb`),
* éventuellement les données dérivées ou scripts annexes,
* les graphiques générés lors de l’analyse dans les fichiers .ipynb

---

## Objectif

CardioScene est un projet d'analyse de signaux biomédicaux (ECG et PPG) en utilisant des techniques avancées de traitement du signal et d'apprentissage automatique. Il se compose de deux études complémentaires :

1. **Détection de l'apnée du sommeil à partir du signal ECG**, via l'extraction des caractéristiques HRV et classification MLP.
2. **Estimation de la pression artérielle (SBP/DBP) à partir du signal PPG**, via un modèle profond hybride CNN–LSTM–MLP.

Ces deux laboratoires sont basés sur des données réelles issues de PhysioNet et du MIMIC-III Waveform Database.

---

## Lab 1 - Détection d’apnée (ECG, HRV, MLP)

Ce travail explore l’utilisation du signal ECG pour détecter automatiquement les épisodes d’apnée du sommeil.

Principaux concepts abordés :

* Prétraitement de l’ECG : filtrage passe-bande, effets de filtfilt, théorème de Nyquist
* Détection des pics R et extraction des intervalles RR
* Calcul des caractéristiques HRV (SDNN, RMSSD, LF/HF, etc.)
* Construction d’un **Perceptron Multicouche (MLP)** pour classifier les segments en *Normal* ou *Apnée*
* Analyse du modèle : matrice de confusion, courbes ROC, sur/sous-apprentissage
* Étude des courbes d’apprentissage et implémentation d'early stopping

---

## Lab 2 — Estimation SBP/DBP (PPG, CNN–LSTM–MLP)

Ce laboratoire vise à estimer les valeurs systolique et diastolique de la pression sanguine à partir du signal PPG (Photopléthysmographie) uniquement.

Principaux concepts abordés :

* Extraction des signaux **PLETH (PPG)** et **ABP (pression invasive)** depuis MIMIC-III
* Filtrage, sous-échantillonnage polyphasé, normalisation Min–Max globale
* Normalisation indépendante des valeurs SBP et DBP
* Construction d’un modèle profond hybride :

  * **CNN** pour l’extraction de motifs locaux
  * **LSTM** pour capturer la dynamique temporelle
  * **MLP** pour la prédiction finale SBP/DBP
* Entraînement avec `tf.data`, Adam, MSE/MAE, EarlyStopping, ModelCheckpoint
* Visualisation avancée :

  * Courbes d’entraînement/validation
  * Graphiques Estimé vs Référence
  * Diagrammes de Bland–Altman


---

## Project structure

The project is organized into two separate laboratory projects, each contained in its own folder:

```
│
├── Lab1/      → Sleep apnea detection using ECG + HRV + MLP
├── Lab2/      → Blood pressure estimation (SBP/DBP) using PPG + hybrid CNN–LSTM–MLP
└── README.md
```

Each folder contains:

* the main Jupyter notebook (`.ipynb`),
* optional additional scripts or preprocessed datasets,
* figures generated during the analyses inside the .ipynb files

---

## Goal of the project

CardioScene is a biomedical signal analysis project focused on ECG and PPG processing, combining signal processing techniques and machine learning/deep learning methods.

It includes two complementary components:

1. **Sleep apnea detection from ECG**, through HRV feature extraction and MLP classification.
2. **Blood pressure estimation (SBP/DBP) from PPG**, using a deep hybrid CNN–LSTM–MLP architecture.

Both labs rely on real clinical waveform data from PhysioNet and the MIMIC-III Waveform Databases.

---

## Lab 1 — Apnea Detection using ECG, HRV and an MLP

This lab explores the use of ECG signals to automatically detect sleep apnea episodes.

Topics covered include:

* ECG preprocessing: band-pass filtering, filtfilt, Nyquist considerations
* R-peak detection and RR-interval extraction
* HRV feature computation (SDNN, RMSSD, LF/HF, etc.)
* Construction of a **Multilayer Perceptron (MLP)** classifier
* Model evaluation: confusion matrices, ROC curves, over/underfitting analysis
* Learning curve exploration and early stopping

---

## Lab 2 — SBP/DBP Estimation (PPG, CNN–LSTM–MLP)

This lab focuses on estimating systolic and diastolic blood pressure from raw PPG signals alone.

Topics covered include:

* Extraction of **PLETH (PPG)** and **ABP** signals from MIMIC-III
* Filtering, polyphase resampling, global Min–Max normalization
* Independent normalization of SBP/DBP labels
* Building a hybrid deep model:

  * **CNN** for local pattern extraction
  * **LSTM** for temporal modeling
  * **MLP** for the final regression output
* Model training with `tf.data`, Adam, MSE/MAE, EarlyStopping, ModelCheckpoint
* Advanced evaluation:

  * Training/validation curves
  * Estimated vs Reference plots
  * Bland–Altman diagrams
metrics
Just tell me!
