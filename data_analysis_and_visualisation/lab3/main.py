import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids


column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv("adult.data", names=column_names, skipinitialspace=True)

df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})


# Plot distributions
numerical_columns = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "income"]

plt.figure(figsize=(12, 10))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=20, kde=True, edgecolor='black')  
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.suptitle("Distribution of Numerical Features", fontsize=14, y=1.02)
plt.savefig("num_distribution.png")
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="income")
plt.title("Income Distribution (0 = <=50K, 1 = >50K)")
plt.xlabel("Income Category")
plt.ylabel("Count")
plt.savefig("income_distribution.png")
plt.show()


print("✅ Descriptive statistics saved to 'descriptive_statistics.csv'.")
print("✅ Graphs saved as PNG images.")


scaler = MinMaxScaler(feature_range=(-1, 1))

df_scaled = df.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("✅ Normalized data (first 5 rows):")
print(df_scaled[numerical_columns].head())



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_columns = [col for col in df.columns if df[col].dtype == "object" and col != "income"]
numeric_columns = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

preprocessor = ColumnTransformer(transformers=[
    ("onehot", OneHotEncoder(drop="first"), categorical_columns),
    ("scaler", MinMaxScaler(feature_range=(-1, 1)), numeric_columns)
])

X = df.drop(columns="income")
y = df["income"]

X_transformed = preprocessor.fit_transform(X)
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X_transformed, y, test_size=0.3, random_state=42
)

sizes = {
    "80:20": [X_train_80.shape[0], X_test_20.shape[0]],
    "70:30": [X_train_70.shape[0], X_test_30.shape[0]]
}

split_df = pd.DataFrame(sizes, index=["Train", "Test"])

split_df.plot(kind='bar', figsize=(8, 5), rot=0)
plt.title("Train/Test Split Comparison")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("split_comparison.png")
plt.show()


smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_80, y_train_80)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_smote)
plt.title("Class Distribution After SMOTE Oversampling")
plt.xlabel("Income Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("smote_distribution.png")
plt.close()

cc = ClusterCentroids(random_state=42)
X_cc, y_cc = cc.fit_resample(X_train_80, y_train_80)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_cc)
plt.title("Class Distribution After ClusterCentroids Undersampling")
plt.xlabel("Income Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("cc_distribution.png")
plt.close()


param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}

lr = LogisticRegression(max_iter=500)
grid = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
grid.fit(X_smote, y_smote)
best_lr = grid.best_estimator_

y_pred = best_lr.predict(X_test_20)

print(classification_report(y_test_20, y_pred))

cm = confusion_matrix(y_test_20, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Logistic Regression (Best Model) on SMOTE Sampled Data")
plt.tight_layout()
plt.savefig("logreg_best_confusion_matrix.png")
plt.close()


lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_smote.toarray(), y_smote)

lda_df = pd.DataFrame({
    "LDA Component 1": X_lda[:, 0],
    "income": y_smote.values
})

plt.figure(figsize=(8, 4))
sns.histplot(data=lda_df, x="LDA Component 1", hue="income", bins=30, kde=True, palette="Set1")
plt.title("LDA Projection (1D) of SMOTE Sampled Data")
plt.xlabel("LDA Component 1")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("lda_projection.png")
plt.close()
