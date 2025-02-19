import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_excel("divorce2.xlsx")

# Define behavioral groups based on Gottman’s framework
groups = {
    "Enhancing_Positive_Interactions": ["Atr5", "Atr8", "Atr9", "Atr10", "Atr11", "Atr12", "Atr13", "Atr14", "Atr15", "Atr16", "Atr17"],
    "Four_Horsemen_Behavior_Patterns": ["Atr31", "Atr32", "Atr33", "Atr34", "Atr35", "Atr36", "Atr37", "Atr38", "Atr39", "Atr40", "Atr41"],
    "Love_Maps": ["Atr18", "Atr21", "Atr22", "Atr23", "Atr24", "Atr25", "Atr26", "Atr27", "Atr28", "Atr29", "Atr30"],
    "Positive_Perspective": ["Atr1", "Atr2", "Atr3", "Atr4", "Atr11"],
    "Conflict_Resolution_Style": ["Atr42", "Atr43", "Atr44", "Atr45", "Atr46", "Atr47", "Atr48", "Atr49", "Atr50", "Atr51", "Atr52", "Atr53", "Atr54"]
}

# Compute group averages
for group, variables in groups.items():
    data[group] = data[variables].mean(axis=1)

# Define features and target variable
X = data[list(groups.keys())]
y = data["Class"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression Model
print("\nLogistic Regression Model Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.2f}")
print(classification_report(y_test, y_pred_logistic))

# Train Decision Tree Model
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

# Evaluate Decision Tree Model
print("\nDecision Tree Model Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(classification_report(y_test, y_pred_tree))

# Plot Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(decision_tree, feature_names=X.columns, class_names=["Not Divorced", "Divorced"], filled=True)
plt.title("Decision Tree")
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": decision_tree.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Decision Tree):")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color='lightcoral', edgecolor='black')
plt.xlabel("Importance Score")
plt.ylabel("Behavior Group")
plt.title("Feature Importance (Decision Tree)")
plt.gca().invert_yaxis()
plt.show()
