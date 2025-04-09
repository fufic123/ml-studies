# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load Dataset
red_df = pd.read_csv("winequality-red.csv", sep=";")
white_df = pd.read_csv("winequality-white.csv", sep=";")

# Add wine type
red_df["type"] = "red"
white_df["type"] = "white"
wine_df = pd.concat([red_df, white_df], ignore_index=True)

# ---------------------------------------------
# 1. Univariate Analysis
# ---------------------------------------------
plt.figure(figsize=(10, 6))
wine_df["alcohol"].hist(bins=30, edgecolor='black')
plt.title("Univariate Analysis: Alcohol Distribution")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.savefig("univariate_alcohol.png")
plt.close()

# ---------------------------------------------
# 2. Discrete, Categorical Attribute (1D)
# ---------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(data=wine_df, x="quality", palette="Set2")
plt.title("Wine Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.savefig("categorical_quality.png")
plt.close()

# ---------------------------------------------
# 3. Multivariate Analysis - Parallel Coordinates
# ---------------------------------------------
subset = wine_df[wine_df["quality"].isin([3, 5, 7])]
parallel_coordinates(subset[["alcohol", "pH", "density", "residual sugar", "quality"]], class_column="quality", colormap="viridis")
plt.title("Parallel Coordinates Plot for Wine Features by Quality")
plt.savefig("parallel_coordinates.png")
plt.close()

# ---------------------------------------------
# 4. Two Continuous Numeric Attributes: Jointplot
# ---------------------------------------------
sns.jointplot(data=wine_df, x="alcohol", y="quality", kind="scatter", hue="type", palette="Set1", height=8)
plt.suptitle("Joint Plot: Alcohol vs Quality", y=1.02)
plt.savefig("jointplot_alcohol_quality.png")
plt.close()

# ---------------------------------------------
# 5. Boxplot & Violin Plot: Alcohol across Quality
# ---------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=wine_df, x="quality", y="alcohol", palette="coolwarm")
plt.title("Boxplot: Alcohol Content Across Quality Levels")
plt.savefig("boxplot_alcohol_quality.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=wine_df, x="quality", y="alcohol", palette="coolwarm")
plt.title("Violin Plot: Alcohol Content Across Quality Levels")
plt.savefig("violinplot_alcohol_quality.png")
plt.close()

# ---------------------------------------------
# 6. 3D Scatter Plot: Alcohol, Density, pH
# ---------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(wine_df["alcohol"], wine_df["density"], wine_df["pH"],
                c=wine_df["quality"], cmap=cm.viridis, alpha=0.7)
ax.set_xlabel("Alcohol")
ax.set_ylabel("Density")
ax.set_zlabel("pH")
plt.title("3D Scatter: Alcohol vs Density vs pH")
fig.colorbar(sc, label="Quality")
plt.savefig("3dscatter_alcohol_density_ph.png")
plt.close()

# ---------------------------------------------
# 7. Alcohol Distribution per Quality Level
# ---------------------------------------------
plt.figure(figsize=(12, 6))
sns.kdeplot(data=wine_df, x="alcohol", hue="quality", fill=True, palette="viridis", alpha=0.6)
plt.title("Alcohol Distribution Across Quality Levels")
plt.savefig("kde_alcohol_quality.png")
plt.close()
