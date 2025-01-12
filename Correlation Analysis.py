    # Step1: Data Prep
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

    # Step3: Machine Learning Analysis Method [Correlation Analysis]
# Calculate correlations between composite scores and the target variable
correlations = {group: data[group].corr(data["Class"]) for group in groups}
# Convert correlations to a DataFrame for easy interpretation
correlation_df = pd.DataFrame(list(correlations.items()), columns=["Group", "Correlation"]).sort_values(by="Correlation", ascending=False)
# Display the correlation results
print("Correlation Between Group Scores and Divorce (Class):")
print(correlation_df)