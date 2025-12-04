# Lab 2 — Estimation de la pression artérielle (SBP/DBP) à partir du signal PPG

## Objectif général

Dans ce laboratoire, l’objectif est d’entraîner un modèle profond **hybride CNN–LSTM–MLP** capable d’estimer les valeurs de pression artérielle systolique (SBP) et diastolique (DBP) à partir du signal **PPG (PLETH)** seul.

Le lab couvre :

1. Extraction & prétraitement du signal PPG (filtrage, normalisation, segmentation).
2. Extraction des segments synchronisés avec les valeurs SBP/DBP du signal ABP.
3. Création d’un dataset propre (inputs = segments de PPG, labels = SBP/DBP).
4. Construction d’un modèle hybride CNN + LSTM + MLP.
5. Entraînement avancé avec `tf.data`, EarlyStopping, ModelCheckpoint.
6. Visualisation complète des résultats (courbes, scatter plots, Bland–Altman).
7. Étude d’ablation (influence de la normalisation et du sous-échantillonnage).

---

## Étapes attendues (organisation en parties comme dans le lab original)

### 1 — Chargement et visualisation du signal PPG & ABP

1. Charger un **enregistrement MIMIC-III** contenant :

   * canal **PLETH** (PPG)
   * canal **ABP** (pression artérielle invasive)

2. Extraire :

   * les signaux PPG et ABP
   * la fréquence d’échantillonnage (souvent 125 Hz)

3. Tracer les deux signaux sur une courte fenêtre (ex. 10 secondes) pour visualisation.

4. Expliquer brièvement la relation physiologique entre PPG et ABP.

Livrables :

* Plots PPG/ABP synchronisés.
* Commentaire sur leur relation temporelle.

### 2 — Extraction des valeurs de SBP/DBP et synchronisation

1. À partir du signal ABP :

   * détecter les pressions systoliques (max)
   * détecter les pressions diastoliques (min)
     pour chaque battement.

2. Associer chaque SBP/DBP à un segment de **PPG de durée fixe** (ex. une fenêtre autour du battement ou un segment glissant).

3. Créer des vecteurs `X` (segments PPG) et `y` (paires SBP/DBP).

Livrables :

* Code de détection SBP/DBP.
* Vérification sur un petit échantillon (ex. SBP/DBP vs ABP réel).
* Tableau montrant quelques paires SBP/DBP.

### 3 — Prétraitement : filtrage, sous-échantillonnage, normalisation

1. **Filtrer le PPG** avec un filtre passe-bande (0.5–8 Hz typique).

2. **Sous-échantillonner** le PPG (ex. de 125 Hz → 62.5 Hz ou 31.25 Hz).

3. **Normaliser** :

   * Segment de PPG : soit MinMax global, soit normalisation indépendante par segment.
   * Labels SBP/DBP : soit normalisation indépendante, soit directe sans normalisation.

4. Comparer l’impact des normalisations (sera réutilisé pour l’étude d’ablation).

Livrables :

* Figures avant/après filtrage.
* Justification du downsampling.
* Code de normalisation.

### 4 — Construction du dataset TensorFlow (`tf.data`)

1. Transformer les listes/arrays `X` et `y` en `tf.data.Dataset`.

2. Mélanger (`shuffle`) et batcher (`batch`).

3. Préparer trois ensembles :

   * **Train** (70%)
   * **Validation** (15%)
   * **Test** (15%)

4. (Bonus) mise en cache avec `.cache()`, préchargement avec `.prefetch()`.

Livrables :

* Code de création du dataset.
* Affichage d’un batch de données (forme des tensores).

### 5 — Modèle hybride CNN–LSTM–MLP

#### Architecture attendue (à adapter légèrement si voulu) :

1. **Bloc CNN**

   * Plusieurs petites convolutions 1D (ex. kernel 5 ou 7)
   * Extraction de motifs locaux du PPG
   * Pooling facultatif

2. **Bloc LSTM**

   * 1 ou 2 couches LSTM ou Bidirectional LSTM
   * Capture des dépendances temporelles

3. **Bloc MLP**

   * Dense(64), Dropout(0.2)
   * Dense(32)
   * **Sortie finale : 2 neurones (SBP, DBP)**

#### Paramètres d’entraînement :

* Optimiseur : Adam (lr 1e−3 ou 1e−4)
* Loss : MSE
* Métriques : MSE, MAE
* Callbacks :

  * `EarlyStopping(patience=50)`
  * `ModelCheckpoint(save_best_only=True)`

#### Livrables :

* Code du modèle.
* `model.summary()` dans le notebook.

---

## 6 — Entraînement & courbes d’apprentissage

### Étapes :

1. Entraîner avec early stopping.
2. Conserver les meilleurs poids.
3. Tracer les courbes :

   * train vs validation (MSE/MAE)
   * sur 100–300 époques selon la configuration

#### Livrables :

* Courbes d’entraînement propres.
* Interprétation (under/overfitting ?).

### 7 — Évaluation finale du modèle

#### Étapes :

1. Faire des prédictions sur *validation* et *test*.
2. Tracer :

   * **Predicted SBP vs True SBP**
   * **Predicted DBP vs True DBP**
   * **Bland–Altman plots** pour SBP et DBP
3. Calculer :

   * MAE final (test)
   * MSE final
   * Écart moyen (bias)

#### Livrables :

* Scatter plots.
* Bland–Altman.
* Tableau des métriques.
* Conclusions.

### 8 — Étude d’ablation (obligatoire)

Tester l’impact de certaines variantes :

#### 1) Sans normalisation

Comparer les métriques.

#### 2) Différents taux de sous-échantillonnage

Exemples : 125 Hz, 62.5 Hz, 31.25 Hz.

#### 3) Architecture modifiée

Quelques possibilités :

* retirer le CNN → LSTM seul
* retirer le LSTM → CNN seul
* varier le nombre de couches

#### Livrables :

* Tableau comparatif (MAE SBP/DBP).
* Graphique si souhaité.
* Conclusions expliquant le meilleur compromis.

---

### 9 — Discussion finale

Quelques points à inclure :

* Pertinence du PPG pour estimer la pression sanguine.
* Limites identifiées (données bruitées, synchronisation ABP/PPG, variabilité inter-patients).
* Pistes d’amélioration (transformers, calibration individuelle, plus long contexte temporel).