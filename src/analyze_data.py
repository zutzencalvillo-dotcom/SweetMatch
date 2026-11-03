
import pandas as pd

# Load the purchase dataset
data = pd.read_csv("data/purchases.csv")

# Show the first rows
print("PURCHASE DATASET")
print(data.head())

# Count the most popular cakes
print("\nMOST POPULAR CAKES")
print(data["cake"].value_counts())

# Count the most popular beverages
print("\nMOST POPULAR BEVERAGES")
print(data["beverage"].value_counts())

# Find the most common combinations
print("\nMOST COMMON COMBINATIONS")
print(
    data.groupby(["cake", "beverage"])
    .size()
    .sort_values(ascending=False)
)
