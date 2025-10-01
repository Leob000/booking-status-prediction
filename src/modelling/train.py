import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import joblib

from data_processing import HotelDataFrameTransformer

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

DATA_DIR = "./src/data"
MODELS_DIR = "./src/models"

# Créer le dossier pour sauvegarder les modèles
os.makedirs(MODELS_DIR, exist_ok=True)

# Traitement des données et séparation train/test (80/20) 
print("Chargement des données...")
df = pd.read_csv(f"{DATA_DIR}/train_data.csv")

print("Transformation des données...")
transformer = HotelDataFrameTransformer(drop_cols=True, encode_categoricals=True, scale_numeric=True)
X = transformer.fit_transform(df)
y = transformer.get_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Sauvegarder le transformeur
joblib.dump(transformer, os.path.join(MODELS_DIR, "hotel_transformer.pkl"))
print(f"Transformeur sauvegardé : {MODELS_DIR}/hotel_transformer.pkl")

# Définir les modèles
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        boosting_type="gbdt", random_state=42, n_jobs=-1
    ),
    "CatBoost": cb.CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6,
        verbose=0, random_state=42
    )
}

# Benchmark des modèles
results = {}
print("\nBenchmark des modèles en cours...\n")

for name, model in tqdm(models.items(), desc="Training models"):
    print(f"\nEntraînement du modèle {name}...")
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    model_path = os.path.join(MODELS_DIR, f"{name.lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    # Évaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Résumé des performances
print("\nRésumé des performances :")
for name, acc in results.items():
    print(f"{name:12s} -> {acc:.4f}")


"""
Résumé des performances :
RandomForest -> 0.8806
XGBoost      -> 0.8699
LightGBM     -> 0.8668
CatBoost     -> 0.8511
"""