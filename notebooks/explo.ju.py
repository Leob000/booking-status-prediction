# %%
# ruff: noqa: E402
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Util to import functions from project root
p = Path.cwd()
root = next(
    (
        parent
        for parent in [p] + list(p.parents)
        if (parent / "pyproject.toml").exists()
    ),
    None,
)
if root is None:
    root = Path.cwd()
sys.path.insert(0, str(root))

from src.model.data_processing import CustomColumnScaler, DataFrameTransformer

# %%
DATA_PATH = Path("../src/data/train_data.csv")
TARGET_COL = "reservation_status"
INCLUDE_WEEK_NUMBER_8TO10 = True

df_raw = pd.read_csv(DATA_PATH)
df_raw.head()

# %%
df = df_raw.copy()
if "row_id" in df.columns:
    df.set_index("row_id", inplace=True)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
if INCLUDE_WEEK_NUMBER_8TO10 and "arrival_date_week_number" in df.columns:
    week_numbers = pd.to_numeric(df["arrival_date_week_number"], errors="coerce")
    df["is_week_number_8to10"] = ((week_numbers >= 8) & (week_numbers <= 10)).astype(
        int
    )

assert (df.isna().sum() == 0).all()

df.describe()

# %%
df.dtypes

# %%
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns if col != TARGET_COL
]
categorical_cols, numeric_cols


# %%
def plot_target_breakdown(
    data: pd.DataFrame,
    column: str,
    target: str = TARGET_COL,
    *,
    max_classes: int = 80,
) -> None:
    if column not in data.columns or column == target:
        return

    n_classes = data[column].nunique()
    if n_classes == 0 or n_classes > max_classes:
        print(f"{column} skipped ({n_classes} classes).")
        return

    target_pct = (
        data.groupby(column)[target]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .multiply(100)
    )

    class_pct = (
        data[column]
        .value_counts(normalize=True)
        .multiply(100)
        .reindex(target_pct.index)
        .fillna(0)
    )

    ax = target_pct.plot(kind="bar", stacked=True, figsize=(7, 4))
    ax.set_title(f"{target} % by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Target share (%)")
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop={"size": 8},
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.3,
        framealpha=0.9,
    )
    ax.set_ylim(0, 100)

    ax2 = ax.twinx()
    class_pct.plot(kind="line", marker="o", color="k", ax=ax2, linewidth=2)
    ax2.set_ylabel("% of total samples")
    # ax2.set_ylim(0, max(100, class_pct.max() * 1.1))
    ax2.set_ylim(0, 100)

    for x, y in enumerate(class_pct.values):
        ax2.text(
            x,
            y + (ax2.get_ylim()[1] * 0.02),
            f"{y:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.show()


for col in df.columns:
    plot_target_breakdown(df, col)

# %%
# Large class imbalance => Stratified cross-validation needed
target_pct = df[TARGET_COL].value_counts(normalize=True).multiply(100)
print("Overall reservation status %:")
for cls, pct in target_pct.sort_index().items():
    print(f"  {cls}: {pct:.2f}%")


# %%
def plot_boxplot(column: str) -> None:
    if column not in df.columns:
        return
    df.boxplot(column=column, by=TARGET_COL)
    plt.title(f"Boxplot of {column} by {TARGET_COL}")
    plt.suptitle("")
    plt.xlabel(TARGET_COL)
    plt.ylabel(column)
    plt.show()


for col in ["lead_time", "days_in_waiting_list", "adr"]:
    plot_boxplot(col)

# %%
# Outlier with extremely high adr
adr_extremes = df[df["adr"] > 1000][["adr", TARGET_COL]]
adr_extremes

# %%
# Majority of customers are from PRT, we make the hypothesis that this is the home country.
df["country"].value_counts().head(10)

# %%
# We apply multiple transformations to the data:
# - Treat `row_id` as an index so we do not leak unwanted information
# - We create a binary flag `is_city_hotel` instead of `hotel`
# - We create a variable `is_home_country` instead of `country` (here the home country is PRT), to avoid encoding numerous categorical variables, under the hypothesis that the customers from foreign countries behave similarly.
# - There is one `Undefined` value in `distribution_channel`, we replace it with the most common channel.
# - We create a binary flag `room_not_same_as_reserved` to capture potential dissatisfaction when the assigned room differs from the reserved one. Note: we could then drop the original columns `assigned_room_type` and `reserved_room_type`.
# - We create a binary flag `has_required_car_parking_spaces` to indicate whether the customer requested parking. (in the train set, every customer with a positive number for required_car_parking_spaces is of class 1).
# - Optionally, we create a binary flag `is_week_number_8to10` to highlight the 3 weeks with really high "no-show" rates.
# - Instead of month/week/day_of_month number, we encode day_of_year as cyclical features (sine and cosine) to capture seasonality.

df_transformer = DataFrameTransformer(
    include_week_number_8to10_flag=INCLUDE_WEEK_NUMBER_8TO10
)
df_transformed = df_transformer.fit_transform(df_raw)

df_raw.shape, df_transformed.shape

# %%
df_transformed.head()

# %%
df_transformed.columns.tolist()

# %%
engineered_plots = [
    "is_city_hotel",
    "is_home_country",
    "room_not_same_as_reserved",
    "has_required_car_parking_spaces",
    "distribution_channel",
    "is_week_number_8to10",
]
for col in engineered_plots:
    plot_target_breakdown(df_transformed, col)

# %%
df_transformed["adr"].max()

# %%
# Seasonality visualization using cyclical encoding of arrival day of year
# Generated with the help of ChatGPT

angle = np.arctan2(
    df_transformed["arrival_day_sin"].values,
    df_transformed["arrival_day_cos"].values,
)
theta = (angle + 2 * np.pi) % (2 * np.pi)
df_transformed["_theta"] = theta

statuses = df_transformed[TARGET_COL].value_counts().index.tolist()
palette = sns.color_palette("tab10", n_colors=max(3, len(statuses)))
colors = {s: palette[i % len(palette)] for i, s in enumerate(statuses)}

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")  # type:ignore
ax.set_theta_direction(-1)  # type:ignore

for status in statuses:
    subset = df_transformed[df_transformed[TARGET_COL] == status]
    if len(subset) == 0:
        continue
    r = np.random.uniform(0.55, 0.95, size=len(subset))
    ax.scatter(
        subset["_theta"],
        r,
        s=10,
        alpha=0.6,
        color=colors[status],
        label=f"{status} ({len(subset)})",
    )

bins = 36
bar_bottom = 1.05
bar_max_height = 0.4
for status in statuses:
    subset = df_transformed[df_transformed[TARGET_COL] == status]
    counts, bin_edges = np.histogram(subset["_theta"], bins=bins, range=(0, 2 * np.pi))
    if counts.sum() == 0:
        continue
    heights = counts / counts.max() * bar_max_height
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(
        bin_centers,
        heights,
        width=(2 * np.pi / bins) * 0.9,
        bottom=bar_bottom,
        color=colors[status],
        alpha=0.25,
        edgecolor="k",
        linewidth=0.2,
    )

month_ticks = {
    "Jan": 0.0,
    "Feb": 31 / 366 * 2 * np.pi,
    "Mar": (31 + 29) / 366 * 2 * np.pi,
    "Apr": (31 + 29 + 31) / 366 * 2 * np.pi,
    "May": (31 + 29 + 31 + 30) / 366 * 2 * np.pi,
    "Jun": (31 + 29 + 31 + 30 + 31) / 366 * 2 * np.pi,
    "Jul": (31 + 29 + 31 + 30 + 31 + 30) / 366 * 2 * np.pi,
    "Aug": (31 + 29 + 31 + 30 + 31 + 30 + 31) / 366 * 2 * np.pi,
    "Sep": (31 + 29 + 31 + 30 + 31 + 30 + 31 + 31) / 366 * 2 * np.pi,
    "Oct": (31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30) / 366 * 2 * np.pi,
    "Nov": (31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31) / 366 * 2 * np.pi,
    "Dec": (31 + 29 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30) / 366 * 2 * np.pi,
}
for month, ang in month_ticks.items():
    ax.text(ang, 1.55, month, ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.65)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Cyclical arrival day encoding vs reservation status", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))

plt.tight_layout()
plt.show()

df_transformed.drop(columns=["_theta"], inplace=True)

# %%
cols_to_normalize = [
    "lead_time",
    # "arrival_date_year",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    # "meal",
    # "market_segment",
    # "distribution_channel",
    # "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    # "deposit_type",
    "days_in_waiting_list",
    # "customer_type",
    "adr",
    "total_of_special_requests",
    # "reservation_status",
    # "is_city_hotel",
    # "is_home_country",
    # "room_not_same_as_reserved",
    # "has_required_car_parking_spaces",
    # "arrival_day_sin",
    # "arrival_day_cos",
    # "is_week_number_8to10",
    "reserved_room_type",
    "assigned_room_type",
]

scaler = CustomColumnScaler(columns=cols_to_normalize, exclude=[TARGET_COL])
df_scaled = scaler.fit_transform(df_transformed)
df_scaled.describe().head()
# %%
df_scaled.describe().T
