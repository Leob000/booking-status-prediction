# %%
# ruff: noqa: E402
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

# %%
df = pd.read_csv("../src/data/train_data.csv")
df.set_index("row_id", inplace=True)
df.drop(columns=["Unnamed: 0"], inplace=True)
target = "reservation_status"

# Check nans
assert df.isna().sum().unique() == 0

df.head()
# %%
df.describe()
# %%
# Transform "hotel" into is_City_hotel 1 for true, 0 for false
print(df["hotel"].astype("category").value_counts())
df["is_City_hotel"] = (df["hotel"] == "City Hotel").astype(int)
df["is_City_hotel"].value_counts()
df.drop(columns=["hotel"], inplace=True)
# %%
# We differentiate between the home country (PRT) and others
df["country"].value_counts()
df["is_PRT"] = (df["country"] == "PRT").astype(int)
df["is_PRT"].value_counts()
df.drop(columns=["country"], inplace=True)


# %%
def make_percent(col):
    if len(df[col].unique()) >= 100:
        print(f"{col} not plotted too many classes")
        return 0
    if col == target:
        return 0
    meal_target_pct = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .multiply(100)
    )

    # percentage weight of each class in `col` relative to total rows
    class_pct = (
        df[col]
        .value_counts(normalize=True)
        .multiply(100)
        .reindex(meal_target_pct.index)
        .fillna(0)
    )

    ax = meal_target_pct.plot(kind="bar", stacked=True)
    ax.set_title(f"Reservation Status % by {col}")
    ax.set_xlabel(f"{col}")
    ax.set_ylabel("Reservation Status %")
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop={"size": 8},
        handlelength=1,
        handletextpad=0.4,
        borderpad=0.3,
        framealpha=0.9,
    )

    # overlay the class weight as a line on a secondary y-axis
    ax2 = ax.twinx()
    class_pct.plot(kind="line", marker="o", color="k", ax=ax2, linewidth=2)
    ax2.set_ylabel("% of total samples")
    ax2.set_ylim(0, max(100, class_pct.max() * 1.1))

    # annotate class weights
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
    make_percent(col)
# %%
