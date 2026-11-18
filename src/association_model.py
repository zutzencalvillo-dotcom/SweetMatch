
import pandas as pd

# Cargar datos limpios
data = pd.read_csv("data/bakery_clean.csv")

# Pasteles y bebidas
cakes = ["gateau chocolat", "cheese cake", "tiramisu"]

beverages = [
    "americano",
    "caffe latte",
    "milk tea",
    "vanila latte",
    "lemon ade",
    "berry ade"
]

# Mostrar que los datos se cargaron
print("ASSOCIATION MODEL")
print("Total records:", len(data))
print("Cakes:", cakes)
print("Beverages:", beverages)



# Calcular reglas de asociación
print("\nASSOCIATION RULES")

total_orders = len(data)

for cake in cakes:
    cake_orders = (data[cake] > 0).sum()

    print("\nCake:", cake)

    for beverage in beverages:
        beverage_orders = (data[beverage] > 0).sum()

        together = (
            (data[cake] > 0) &
            (data[beverage] > 0)
        ).sum()

        confidence = together / cake_orders
        beverage_probability = beverage_orders / total_orders
        lift = confidence / beverage_probability

        print(
            beverage,
            "| Purchases:", together,
            "| Confidence:", round(confidence, 2),
            "| Lift:", round(lift, 2)
        )



# Recomendar bebida usando reglas de asociación
def recommend_association(cake):
    best_beverage = None
    best_lift = 0

    total_orders = len(data)
    cake_orders = (data[cake] > 0).sum()

    for beverage in beverages:
        beverage_orders = (data[beverage] > 0).sum()

        together = (
            (data[cake] > 0) &
            (data[beverage] > 0)
        ).sum()

        confidence = together / cake_orders
        probability = beverage_orders / total_orders
        lift = confidence / probability

        if together >= 5 and lift > 1 and lift > best_lift:
            best_lift = lift
            best_beverage = beverage

    return best_beverage


# Probar el segundo recomendador
print("\nASSOCIATION RECOMMENDATIONS")

for cake in cakes:
    print(cake, ":", recommend_association(cake))
