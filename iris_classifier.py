import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda i: iris.target_names[i])

print("First 5 rows:")
print(df.head())

print("\nClass labels (target names):")
print(iris.target_names)


print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

sns.pairplot(df, hue='species')
plt.suptitle("Iris Feature Pairplot", y=1.02)

plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")

X = df.iloc[:, :4].values   
y = df['target'].values     

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    pred_class = model.predict(sample_scaled)[0]
    pred_name = iris.target_names[pred_class]
    return pred_name

example_pred = predict_iris(5.1, 3.5, 1.4, 0.2)
print(f"\nExample Prediction for [5.1, 3.5, 1.4, 0.2]: {example_pred}")


def cli_predict():
    print("\n--- Iris Flower Prediction ---")
    sepal_length = float(input("Sepal length (cm): "))
    sepal_width = float(input("Sepal width (cm): "))
    petal_length = float(input("Petal length (cm): "))
    petal_width = float(input("Petal width (cm): "))

    species = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    print(f"\nPredicted Iris species: {species}")

if __name__ == "__main__":
    cli_predict()
