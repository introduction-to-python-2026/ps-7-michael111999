import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# -----------------------------
# 1. Load and inspect the data
# -----------------------------
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# -----------------------------
# 2. Select several features
# -----------------------------
features = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "flavanoids"
]

selected_df = df[features]

# -----------------------------
# 3. Create histograms
# -----------------------------
selected_df.hist(bins=20, figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Create scatter plots
# -----------------------------
plt.figure(figsize=(6, 4))
plt.scatter(df["alcohol"], df["color_intensity"])
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.title("Alcohol vs Color Intensity")
plt.show()

# -----------------------------
# 5. Correlation scatter plot
# -----------------------------
plt.figure(figsize=(6, 4))
sns.regplot(
    x="alcohol",
    y="flavanoids",
    data=df,
    scatter_kws={"alpha": 0.6}
)
plt.title("Correlation: Alcohol vs Flavanoids")
plt.tight_layout()

# Save figure
plt.savefig("correlation.png")
plt.show()
