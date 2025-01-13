    # Step1: Data Prep
    # import needed packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv('/Users/macbook/Downloads/divorce2.csv')
print(data.head(170))
# Get the number of rows and columns in the dataset
rows, columns = data.shape
rows, columns

    # Step2: Group the Attributes
# Define the groups and their corresponding variables
groups = {
    "Enhancing Positive Interactions": ["Atr5", "Atr8", "Atr9", "Atr10", "Atr11", "Atr12", "Atr13", "Atr14", "Atr15", "Atr16", "Atr17"],
    "Four Horsemen Behavior Patterns": ["Atr31", "Atr32", "Atr33", "Atr34", "Atr35", "Atr36", "Atr37", "Atr38", "Atr39", "Atr40", "Atr41"],
    "Love Maps": ["Atr18", "Atr21", "Atr22", "Atr23", "Atr24", "Atr25", "Atr26", "Atr27", "Atr28", "Atr29", "Atr30"],
    "Positive Perspective": ["Atr1", "Atr2", "Atr3", "Atr4", "Atr11"],  # Atr11 appears in two groups
    "Conflict Resolution Style": ["Atr42", "Atr43", "Atr44", "Atr45", "Atr46", "Atr47", "Atr48", "Atr49", "Atr50", "Atr51", "Atr52", "Atr53", "Atr54"]
}
# Create composite scores for each group
for group, variables in groups.items():
    data[group] = data[variables].mean(axis=1)

# Step3: Machine Learning Analysis Method [Logistic Regression]
    # Features and target
features = data[list(groups.keys())]
target = data["Class"]
    # Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    # Train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
    # Predict on test set
y_pred_logistic = logistic_model.predict(X_test)
    # Evaluate logistic regression model
print("Logistic Regression Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.2f}")
print(classification_report(y_test, y_pred_logistic))
    # Feature importance (odds ratios)
import numpy as np
coefficients = logistic_model.coef_[0]
odds_ratios = np.exp(coefficients)
print("\nFeature Importance (Logistic Regression):")
for feature, coef, odds in zip(features.columns, coefficients, odds_ratios):
    print(f"{feature}: Coefficient = {coef:.2f}, Odds Ratio = {odds:.2f}")

# Step4: Visualization
    # Data for visualization
features = [
    "Enhancing Positive Interactions",
    "Four Horsemen Behavior Patterns",
    "Love Maps",
    "Positive Perspective",
    "Conflict Resolution Style"
]
odds_ratios = [2.19, 4.03, 2.44, 2.47, 2.02]
    # Create the bar plot
plt.figure(figsize=(10, 6))
plt.barh(features, odds_ratios, color='skyblue', edgecolor='black')
    # Add labels and title
plt.xlabel("Odds Ratio", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Feature Importance (Logistic Regression - Odds Ratios)", fontsize=14)
for i, v in enumerate(odds_ratios):
    plt.text(v + 0.1, i, f"{v:.2f}", color='black', va='center', fontsize=10)
    # Show the plot
plt.tight_layout()
plt.show()