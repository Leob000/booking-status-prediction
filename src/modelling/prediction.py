import pandas as pd
import joblib
from data_processing import HotelDataFrameTransformer

DATA_DIR = "./src/data"
MODEL_DIR = "./src/models"

# Charger le dataset de test (sans colonne cible)
df_test = pd.read_csv(f"{DATA_DIR}/test_data.csv")

# Preserve row ids for the submission
row_ids = df_test["row_id"]

# Charger le transformeur et le modèle
transformer = joblib.load(f"{MODEL_DIR}/hotel_transformer.pkl")
model = joblib.load(f"{MODEL_DIR}/randomforest_model.pkl")

# Transformer le dataset
X_test = transformer.transform(df_test)

# Prédire
y_pred = model.predict(X_test)

# Décoder si nécessaire
if hasattr(transformer, "target_encoder_"):
    y_pred = transformer.target_encoder_.inverse_transform(y_pred)

# Sauvegarder les prédictions

predictions= pd.DataFrame({"row_id": row_ids, "reservation_status": y_pred})
predictions.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print(f"Prédictions enregistrées dans {DATA_DIR}/submission.csv")


