import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading dataset
data = pd.read_csv("data/titanic.csv")

print("First 5 rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nBasic statistics:")
print(data.describe())

# Handle missing values
data = data.fillna({"Age": data["Age"].median(), "Embarked": "S"})

# Visualizations
sns.countplot(x="Survived", hue="Sex", data=data)
plt.title("Survival Count by Gender")
plt.savefig("images/survival_by_gender.png")
plt.show()

plt.hist(data["Age"], bins=20, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("images/age_distribution.png")
plt.show()