    # Step1: Data Prep
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

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


# Step3: Machine Learning Analysis Method [Decision Tree]
    # Features and target
features = data[list(groups.keys())]
target = data["Class"]
    # Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    # Train Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=5)  # Adjust max_depth if needed
decision_tree.fit(X_train, y_train)
    # Predict on test set
y_pred_tree = decision_tree.predict(X_test)
    # Evaluate Decision Tree model
print("\nDecision Tree Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(classification_report(y_test, y_pred_tree))

# Step 4: Visualization
    # Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(decision_tree, feature_names=X_train.columns, class_names=["Not Divorced", "Divorced"], filled=True)
plt.title("Decision Tree")
plt.show()
    # Feature Importance
print("\nFeature Importance (Decision Tree):")
for feature, importance in zip(X_train.columns, decision_tree.feature_importances_):
    print(f"{feature}: Importance = {importance:.2f}")
    # Assuming X_train.columns and decision_tree.feature_importances_ are available
    # Example feature names and importance values for visualization
feature_names = X_train.columns
feature_importances = decision_tree.feature_importances_
    # Create a DataFrame for easier visualization
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
    # Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (Decision Tree)")
plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance at the top
plt.show()
