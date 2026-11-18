
import pandas as pd

# Leer los datos de Kaggle
data = pd.read_csv("data/Bakery Sales.csv")

# Mostrar las primeras 5 filas
print("KAGGLE DATASET")
print(data.head())

# Mostrar cantidad de filas y columnas
print("\nDATASET SIZE")
print(data.shape)

# Mostrar nombres de las columnas
print("\nCOLUMNS")
print(data.columns)

# Revisar valores faltantes
print("\nMISSING VALUES")
print(data.isnull().sum())



# Revisar el tamaño real del dataset
print("\nTOTAL ROWS AND COLUMNS")
print(data.shape)

# Revisar las últimas 5 filas
print("\nLAST 5 ROWS")
print(data.tail())


print("\nEMPTY ROWS")
print(data.isnull().all(axis=1).sum())

print("\nROWS WITH DATE AND TOTAL")
print(data.dropna(subset=["datetime", "total"]).shape[0])

print("\nFIRST PURCHASES")
print(data.head(10).to_string())


# Eliminar filas completamente vacías
clean_data = data.dropna(how="all").copy()

# Identificar columnas de productos
products = clean_data.columns[4:]

# Reemplazar valores vacíos de productos por 0
clean_data[products] = clean_data[products].fillna(0)

# Mostrar resultados de la limpieza
print("\nCLEAN DATASET SIZE")
print(clean_data.shape)

print("\nFIRST CLEAN ROWS")
print(clean_data.head())

# Revisar valores faltantes después de limpiar
print("\nMISSING VALUES AFTER CLEANING")
print(clean_data.isnull().sum())


# Contar las unidades vendidas de cada producto
sales = clean_data[products].sum()

# Ordenar de mayor a menor
sales = sales.sort_values(ascending=False)

# Mostrar los 10 productos más vendidos
print("\nTOP 10 BEST SELLING PRODUCTS")
print(sales.head(10))



# Pasteles que utilizaremos en SweetMatch
cakes = [
    "gateau chocolat",
    "cheese cake",
    "tiramisu"
]

# Bebidas que utilizaremos
beverages = [
    "americano",
    "caffe latte",
    "milk tea",
    "vanila latte",
    "lemon ade",
    "berry ade"
]

# Contar unidades vendidas de pasteles
print("\nCAKE SALES")
print(clean_data[cakes].sum().sort_values(ascending=False))

# Contar unidades vendidas de bebidas
print("\nBEVERAGE SALES")
print(clean_data[beverages].sum().sort_values(ascending=False))



# Buscar compras donde aparecen juntos un pastel y una bebida
print("\nCAKE AND BEVERAGE COMBINATIONS")

for cake in cakes:
    for beverage in beverages:
        count = (
            (clean_data[cake] > 0) &
            (clean_data[beverage] > 0)
        ).sum()

        print(cake, "+", beverage, "=", count)



# Recomendar la bebida más comprada junto con un pastel
def recommend_beverage(cake):
    results = {}

    for beverage in beverages:
        count = (
            (clean_data[cake] > 0) &
            (clean_data[beverage] > 0)
        ).sum()

        results[beverage] = count

    best_beverage = max(results, key=results.get)

    return best_beverage


# Probar las recomendaciones
print("\nSWEETMATCH RECOMMENDATIONS")

print("Gateau chocolat:", recommend_beverage("gateau chocolat"))
print("Cheese cake:", recommend_beverage("cheese cake"))
print("Tiramisu:", recommend_beverage("tiramisu"))



# Calcular la confianza de cada combinación
print("\nRECOMMENDATION CONFIDENCE")

for cake in cakes:
    total_cake = (clean_data[cake] > 0).sum()

    print("\nPastel:", cake)

    for beverage in beverages:
        together = (
            (clean_data[cake] > 0) &
            (clean_data[beverage] > 0)
        ).sum()

        confidence = (together / total_cake) * 100

        print(beverage, ":", round(confidence, 2), "%")


# Calcular lift para cada combinación
print("\nRECOMMENDATION LIFT")

total_orders = len(clean_data)

for cake in cakes:
    print("\nPastel:", cake)

    cake_orders = (clean_data[cake] > 0).sum()

    for beverage in beverages:
        beverage_orders = (clean_data[beverage] > 0).sum()

        together = (
            (clean_data[cake] > 0) &
            (clean_data[beverage] > 0)
        ).sum()

        confidence = together / cake_orders
        beverage_probability = beverage_orders / total_orders

        lift = confidence / beverage_probability

        print(beverage, ":", round(lift, 2))



# Guardar el dataset limpio
clean_data.to_csv("data/bakery_clean.csv", index=False)

print("\nConjunto de datos limpio guardado correctamente!")
