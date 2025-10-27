from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


@dataclass
class ArrivalDateConfig:
    """Configuration driving how the arrival date features are engineered."""

    reference_year: int = 2020
    cycle_length: int = 366  # leap year keeps the encoding smooth


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Feature engineering pipeline tailored to the booking-status dataset."""

    def __init__(
        self,
        *,
        drop_auxiliary_columns: bool = True,
        drop_room_type_columns: bool = True,
        drop_adr_outliers: bool = True,
        adr_upper_bound: float = 1000.0,
        city_hotel_label: str = "City Hotel",
        home_country: str = "PRT",
        arrival_config: ArrivalDateConfig | None = None,
        include_week_number_8to10_flag: bool = False,
    ) -> None:
        self.drop_auxiliary_columns = drop_auxiliary_columns
        self.drop_room_type_columns = drop_room_type_columns
        self.drop_adr_outliers = drop_adr_outliers
        self.adr_upper_bound = adr_upper_bound
        self.city_hotel_label = city_hotel_label
        self.home_country = home_country
        self.arrival_config = arrival_config or ArrivalDateConfig()
        self.include_week_number_8to10_flag = include_week_number_8to10_flag

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: N803
        df = X.copy()
        if "distribution_channel" in df.columns:
            non_null = df.loc[
                df["distribution_channel"].notna()
                & (df["distribution_channel"] != "Undefined"),
                "distribution_channel",
            ]
            if len(non_null) > 0:
                self._distribution_channel_mode = non_null.mode().iloc[0]
            else:
                self._distribution_channel_mode = "Undefined"
        else:
            self._distribution_channel_mode = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        df = X.copy()

        if "row_id" in df.columns:
            df = df.set_index("row_id")

        df = self._encode_hotel(df)
        df = self._encode_country(df)
        df = self._replace_distribution_channel(df)
        df = self._add_room_mismatch_indicator(df)
        df = self._add_parking_flag(df)
        df = self._encode_arrival_date(df)
        df = self._add_week_number_flag(df)
        df = self._coerce_integer_flags(df)

        if self.drop_auxiliary_columns:
            columns_to_drop = [
                "Unnamed: 0",
                "hotel",
                "country",
                "arrival_date_month",
                "arrival_date_week_number",
                "arrival_date_day_of_month",
                "arrival_day_of_year",
                "required_car_parking_spaces",
            ]
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

        if self.drop_room_type_columns:
            columns_to_drop = ["reserved_room_type", "assigned_room_type"]
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

        if self.drop_adr_outliers and "adr" in df.columns:
            df = df[df["adr"] <= self.adr_upper_bound].copy()

        return df

    def set_output(self, *, transform=None):  # type: ignore[override]
        # For compatibility with sklearn Pipeline
        self._output_config = {"transform": transform}
        return self

    def _encode_hotel(self, df: pd.DataFrame) -> pd.DataFrame:
        if "hotel" not in df.columns:
            return df
        df["is_city_hotel"] = (df["hotel"] == self.city_hotel_label).astype(int)
        return df

    def _encode_country(self, df: pd.DataFrame) -> pd.DataFrame:
        if "country" not in df.columns:
            return df
        df["is_home_country"] = (df["country"] == self.home_country).astype(int)
        return df

    def _replace_distribution_channel(self, df: pd.DataFrame) -> pd.DataFrame:
        if "distribution_channel" not in df.columns:
            return df
        replacement = getattr(self, "_distribution_channel_mode", None)
        if replacement is None:
            replacement = (
                df.loc[
                    df["distribution_channel"] != "Undefined", "distribution_channel"
                ]
                .mode()
                .iloc[0]
                if (df["distribution_channel"] != "Undefined").any()
                else "Undefined"
            )
        df["distribution_channel"] = df["distribution_channel"].replace(
            "Undefined", replacement
        )
        return df

    def _add_room_mismatch_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        if (
            "reserved_room_type" not in df.columns
            or "assigned_room_type" not in df.columns
        ):
            return df
        df["room_not_same_as_reserved"] = (
            df["reserved_room_type"] != df["assigned_room_type"]
        ).astype(int)
        return df

    def _add_parking_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        col = "required_car_parking_spaces"
        if col not in df.columns:
            return df
        df["has_required_car_parking_spaces"] = (df[col] > 0).astype(int)
        return df

    def _add_week_number_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.include_week_number_8to10_flag:
            return df
        col = "arrival_date_week_number"
        if col not in df.columns:
            return df
        week_numbers = pd.to_numeric(df[col], errors="coerce")
        df["is_week_number_8to10"] = (
            ((week_numbers >= 8) & (week_numbers <= 10)).astype(int)
        )
        return df

    def _encode_arrival_date(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"arrival_date_month", "arrival_date_day_of_month"}
        if not required_cols.issubset(df.columns):
            return df

        year = self.arrival_config.reference_year
        df["arrival_day_of_year"] = pd.to_datetime(
            df["arrival_date_month"].astype(str)
            + " "
            + df["arrival_date_day_of_month"].astype(str)
            + f" {year}",
            format="%B %d %Y",
            errors="coerce",
        ).dt.dayofyear  # type:ignore

        week_number = df.get("arrival_date_week_number")
        if week_number is not None:
            fallback = (
                week_number.astype(float)
                .mul(7)
                .clip(lower=1, upper=self.arrival_config.cycle_length)
            )
            df["arrival_day_of_year"] = df["arrival_day_of_year"].fillna(fallback)
        else:
            df["arrival_day_of_year"] = df["arrival_day_of_year"].fillna(1)

        radians = (
            2 * np.pi * df["arrival_day_of_year"] / self.arrival_config.cycle_length
        )
        df["arrival_day_sin"] = np.sin(radians)
        df["arrival_day_cos"] = np.cos(radians)
        return df

    def _coerce_integer_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "total_of_special_requests",
        ]
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df


class CustomColumnScaler(BaseEstimator, TransformerMixin):
    """Standardise selected numerical columns while keeping a DataFrame output."""

    def __init__(
        self,
        columns: Optional[Iterable[str]] = None,
        *,
        exclude: Optional[Iterable[str]] = None,
        output_as_pandas: bool = True,
    ) -> None:
        self.columns = list(columns) if columns is not None else None
        self.exclude = list(exclude) if exclude is not None else []
        self.output_as_pandas = output_as_pandas

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # noqa: N803
        if self.columns is None:
            self.cols_to_normalize_ = [
                col
                for col in X.columns
                if pd.api.types.is_numeric_dtype(X[col]) and col not in self.exclude
            ]
        else:
            available = [col for col in self.columns if col in X.columns]
            self.cols_to_normalize_ = available

        if not self.cols_to_normalize_:
            raise ValueError("No columns available for scaling.")

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.cols_to_normalize_])

        self.other_columns_ = [
            col for col in X.columns if col not in self.cols_to_normalize_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X_scaled = X.copy()
        X_scaled[self.cols_to_normalize_] = self.scaler_.transform(
            X_scaled[self.cols_to_normalize_]
        )

        if not self.output_as_pandas:
            # Retain column ordering (scaled columns first, then others)
            ordered_cols = self.cols_to_normalize_ + self.other_columns_
            return X_scaled[ordered_cols].values

        return X_scaled

    def set_output(self, *, transform=None):  # type: ignore[override]
        self._output_config = {"transform": transform}
        return self
