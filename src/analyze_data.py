
import pandas as pd

# datos de compra
data = pd.read_csv("data/purchases.csv")


print("PURCHASE DATASET")
print(data.head())

# contar los pasteles mas populares
print("\nMOST POPULAR CAKES")
print(data["cake"].value_counts())

# bebidas mas populares 
print("\nMOST POPULAR BEVERAGES")
print(data["beverage"].value_counts())

# Combinaciones mas comunes
print("\nMOST COMMON COMBINATIONS")
print(
    data.groupby(["cake", "beverage"])
    .size()
    .sort_values(ascending=False)
)
