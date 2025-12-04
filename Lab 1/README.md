# Lab 1 — Détection de l’apnée du sommeil à partir du signal ECG

## Objectif général

Détecter automatiquement des minutes d’apnée à partir du seul signal ECG en :

1. prétraitant le signal,
2. extrayant des caractéristiques HRV,
3. construisant un classifieur (MLP) et en
4. analysant les courbes d’apprentissage et l’early stopping.

---

## Étapes attendues (détaillées — faire dans l’ordre)

### 1 — Préparation et visualisation (Exercice 1)

1. **Charger un enregistrement** depuis la base *Apnea-ECG* (PhysioNet) avec `wfdb.rdsamp` / `wfdb.rdann`.

   * Extraire le canal ECG et la fréquence d’échantillonnage (fs ≈ 100 Hz).
   * Récupérer les annotations minute-par-minute (`annotation.symbol`).

2. **Afficher un segment "normal" d’1 minute** (par ex. 2ᵉ minute) et son annotation (N/A).

   * Tracer à l’aide de `matplotlib` et documenter l’observation.

3. **Trouver et afficher le premier segment labellisé 'A'** (apnée).

   * Comparer visuellement les deux segments (normal vs apnée) et noter les différences observées (variations du rythme, amplitude, forme des battements…).

4. **Filtrer** les deux segments avec un **filtre Butterworth passe-bande** (ordre 4, 0.1–35 Hz) en utilisant `scipy.signal.butter` + `filtfilt`.

   * Montrer avant/après filtrage et expliquer l’effet du filtrage (nettoyage du bruit, modification éventuelle de la forme des pics).
   * Expliquer pourquoi `filtfilt` évite le déphasage et mentionner les effets de bord (padding recommandé).

5. **Fréquence d’échantillonnage minimale** : proposer (et justifier) une fréquence minimale acceptable après filtrage en se référant au théorème de Nyquist.

> **Livrable pour 1** : notebook / section montrant : chargement, figure du segment normal et apnéique, code du filtre, plots avant/après, justification Nyquist.

---

### 2 — Extraction des caractéristiques HRV (Exercice 2)

1. **Détection des pics R** :

   * Utiliser `biosppy.signals.ecg.hamilton_segmenter` (ou équivalent) pour détecter les R-peaks.
   * Corriger les R-peaks avec `correct_rpeaks` si nécessaire.
   * Tracer le signal ECG avec les R-peaks marqués.

2. **Calculer les intervalles R-R** (en ms) et vérifier leur plausibilité (ex. intervalles trop courts/longs).

3. **Extraire les caractéristiques HRV** :

   * Time-domain : SDNN, RMSSD, SDSD, pNN50, etc.
   * Frequency-domain : Total power, VLF (0–0.04 Hz), LF (0.04–0.15 Hz), HF (0.15–0.4 Hz).
   * Utiliser `hrvanalysis` ou coder les métriques si tu préfères.

4. **Tableau comparatif** : afficher un tableau comparant chaque caractéristique pour le segment normal et le segment apnéique, avec brèves interprétations des différences.

> **Livrable pour 2** : notebook / table (pandas DataFrame), figures des R-peaks, code d’extraction HRV, bref commentaire d’interprétation.

---

### 3 — Conception d’un classifieur MLP (Exercice 3)

1. **Chargement du dataset de caractéristiques** (fichier `features_apnea.csv` fourni) :

   * Séparer `X` (caractéristiques) et `y` (label 0=Normal, 1=Apnée).

2. **Split des données** :

   * Train 70%, Validation 15%, Test 15% (utiliser `train_test_split` avec `stratify=y` pour garder la proportion des classes).

3. **Construire et tester des MLP** :

   * Utiliser `sklearn.neural_network.MLPClassifier`.
   * Paramètres à tester manuellement (au moins 6 combinaisons) : `hidden_layer_sizes` (ex. (10,), (20,), (20,10)), `activation` ('relu','tanh'), `alpha` (0.0001,0.001,0.01). `max_iter=500`, `solver='adam'`, `random_state=42`.
   * Pour chaque configuration, enregistrer accuracy sur train/val/test, matrice de confusion.

4. **Sélection du meilleur modèle** selon l’accuracy sur l’ensemble de validation.

5. **Interprétation** :

   * Underfitting vs overfitting (comparer train vs val).
   * Cohérence entre validation et test.
   * Discussion sur l’efficacité des caractéristiques HRV pour la tâche.

> **Livrable pour 3** : code d’entraînement, tableau récapitulatif des configurations et métriques, matrices de confusion, choix final et justification.

---

### 4 — Courbes d’apprentissage (Exercice 4)

1. **Entraînement époque-par-époque** :

   * Utiliser `warm_start=True` et `max_iter=1` dans une boucle pour enregistrer la perte / accuracy à chaque époque.
   * Construire les courbes d’entraînement et de validation (loss/accuracy).

2. **Lissage & interprétation** :

   * Appliquer un lissage (ex. EMA) pour mieux lire les tendances.
   * Identifier l’époque où l’overfitting apparaît et indiquer à quelle époque l’on aurait dû arrêter (early stopping).

3. **Analyse** :

   * Décrire à quelles époques le modèle est sous-appris ou sur-appris.
   * Confronter les observations aux résultats sur le test.

> **Livrable pour 4** : plots (train/val loss, éventuellement accuracy), code de warm_start, court texte d’analyse.

---

### 5 — Early Stopping (Exercice 5)

* Implémenter un mécanisme d’arrêt anticipé basé sur la perte de validation (patience = 150, epochs max = 1500, min_epochs = 20).
* Montrer où et pourquoi l’entraînement s’est arrêté, tracer les courbes lissées et commenter l’intérêt.

---

## Exemple minimal de snippet (à inclure dans le notebook, paraphrasé)

```python
# Charger un signal ECG depuis PhysioNet
import wfdb
record_name = 'a01'
signal, fields = wfdb.rdsamp(record_name, pn_dir='apnea-ecg')
ecg = signal[:,0]
fs = fields['fs']  # ~100 Hz

# Filtre Butterworth passe-bande
from scipy.signal import butter, filtfilt
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut/nyq; high = highcut/nyq
    b,a = butter(order, [low, high], btype='band')
    return b,a
b,a = butter_bandpass(0.1, 35, fs)
ecg_filt = filtfilt(b, a, ecg)
```
