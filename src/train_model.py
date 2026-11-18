
import pandas as pd

# Cargar dataset limpio
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

print("MACHINE LEARNING TRAINING")
print("Total records:", len(data))
print("Dataset loaded successfully!")



# Seleccionar compras con exactamente un tipo de pastel
cake_count = (data[cakes] > 0).sum(axis=1)

# Seleccionar compras con exactamente un tipo de bebida
beverage_count = (data[beverages] > 0).sum(axis=1)

# Filtrar compras válidas para el entrenamiento
training_data = data[
    (cake_count == 1) &
    (beverage_count == 1)
].copy()

# Identificar el pastel y la bebida de cada compra
training_data["cake"] = training_data[cakes].idxmax(axis=1)
training_data["beverage"] = training_data[beverages].idxmax(axis=1)

print("\nTRAINING DATA")
print("Valid purchases:", len(training_data))

print("\nFIRST TRAINING RECORDS")
print(training_data[["cake", "beverage"]].head(10))

print("\nBEVERAGE DISTRIBUTION")
print(training_data["beverage"].value_counts())



# Revisar cantidad de ejemplos por pastel
print("\nCAKE DISTRIBUTION")
print(training_data["cake"].value_counts())

# Revisar combinaciones disponibles
print("\nTRAINING COMBINATIONS")
print(pd.crosstab(
    training_data["cake"],
    training_data["beverage"]
))


from sklearn.model_selection import train_test_split

# Excluir tiramisu porque solamente tiene 2 ejemplos
model_data = training_data[
    training_data["cake"] != "tiramisu"
].copy()

# Convertir el nombre del pastel en columnas numéricas
X = pd.get_dummies(model_data["cake"]).astype(int)

# Bebida que queremos predecir
y = model_data["beverage"]

# Dividir los datos para entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nMODEL DATA PREPARED")
print("Total examples:", len(model_data))
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Input columns:", X.columns.tolist())



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Crear el modelo
model_tree = DecisionTreeClassifier(random_state=42)

# Entrenar con los datos
model_tree.fit(X_train, y_train)

# Realizar predicciones
predictions_tree = model_tree.predict(X_test)

# Evaluar el modelo
accuracy_tree = accuracy_score(y_test, predictions_tree)

print("\nDECISION TREE RESULTS")
print("Accuracy:", round(accuracy_tree * 100, 2), "%")

print("\nPREDICTIONS")
for real, predicted in zip(y_test, predictions_tree):
    print("Real:", real, "| Predicted:", predicted)



from sklearn.ensemble import RandomForestClassifier

# Crear el segundo modelo
model_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entrenar el modelo
model_forest.fit(X_train, y_train)

# Realizar predicciones
predictions_forest = model_forest.predict(X_test)

# Evaluar el modelo
accuracy_forest = accuracy_score(y_test, predictions_forest)

# Comparar con recomendar siempre americano
baseline_predictions = ["americano"] * len(y_test)
accuracy_baseline = accuracy_score(
    y_test, baseline_predictions
)

print("\nMODEL COMPARISON")
print("Baseline:", round(accuracy_baseline * 100, 2), "%")
print("Decision Tree:", round(accuracy_tree * 100, 2), "%")
print("Random Forest:", round(accuracy_forest * 100, 2), "%")

print("\nRANDOM FOREST PREDICTIONS")
for real, predicted in zip(y_test, predictions_forest):
    print("Real:", real, "| Predicted:", predicted)



import os
import joblib

# Crear carpeta para guardar modelos
os.makedirs("models", exist_ok=True)

# Guardar modelos entrenados
joblib.dump(model_tree, "models/decision_tree.pkl")
joblib.dump(model_forest, "models/random_forest.pkl")

# Guardar nombres de columnas utilizados
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("\nMODELS SAVED SUCCESSFULLY!")
