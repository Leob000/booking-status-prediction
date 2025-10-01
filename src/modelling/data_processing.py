import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class HotelDataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Préprocesseur pour le dataset hôtelier avant entraînement.

    Paramètres
    ----------
    drop_cols : bool, par défaut=True
        Si True, supprime certaines colonnes inutiles (ex: `row_id`, `reservation_status` dans X).
    encode_categoricals : bool, par défaut=True
        Si True, applique un encodage ordinal sur les variables catégorielles et binaires.
    scale_numeric : bool, par défaut=False
        Si True, normalise les variables numériques avec (x - mean) / std.
    """
    def __init__(self, drop_cols=True, encode_categoricals=True, scale_numeric=False):
        self.drop_cols = drop_cols
        self.encode_categoricals = encode_categoricals
        self.scale_numeric = scale_numeric
        self.encoders_ = {}  # sauvegarde des encodeurs OrdinalEncoder pour test/train consistency

        # Colonnes numériques et catégorielles
        self.numeric_cols = [
            "lead_time",
            "arrival_date_year",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "days_in_waiting_list",
            "adr",
            "required_car_parking_spaces",
            "total_of_special_requests"
        ]
        self.categorical_cols = [
            "hotel",
            "arrival_date_month",
            "meal",
            "country",
            "market_segment",
            "distribution_channel",
            "reserved_room_type",
            "assigned_room_type",
            "deposit_type",
            "customer_type"
        ]
        self.binary_cols = [
            "is_repeated_guest"
        ]

    def fit(self, X, y=None):
        """
        Apprend les encodeurs pour les variables catégorielles.
        """
        if self.encode_categoricals:
            for col in self.categorical_cols + self.binary_cols:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
                encoder.fit(X[[col]].astype(str))
                self.encoders_[col] = encoder
        return self

    def clean_column_names(self, df):
        """
        Supprime les caractères spéciaux des noms de colonnes.
        (pour résoudre le bug avec le modèle LightGBM)
        """
        df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        return df

    def transform(self, X):
        """
        Transforme le DataFrame en appliquant encodage et normalisation.
        """
        df = X.copy()

        # Index
        if "row_id" in df.columns and self.drop_cols:
            df = df.set_index("row_id")

        # Encodage catégoriel
        if self.encode_categoricals:
            for col in self.categorical_cols + self.binary_cols:
                if col in df.columns:
                    df[col] = self.encoders_[col].transform(df[[col]].astype(str))

        # Normalisation optionnelle
        if self.scale_numeric:
            for col in self.numeric_cols:
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Drop colonne cible si présente
        if self.drop_cols and "reservation_status" in df.columns:
            self.y_ = df["reservation_status"]
            df = df.drop(columns=["reservation_status"])

        # Nettoyer noms de colonnes pour compatibilité LightGBM
        df = self.clean_column_names(df)

        return df

    def get_target(self, X):
        """
        Encode et retourne la variable cible `reservation_status`.
        """
        if "reservation_status" in X.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(X["reservation_status"])
            self.target_encoder_ = le
            return y
        else:
            raise ValueError("La colonne 'reservation_status' est absente du dataset.")
