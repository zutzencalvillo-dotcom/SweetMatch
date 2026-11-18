
import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(
    page_title="SweetMatch",
    page_icon="🍰"
)

st.title("🍰 SweetMatch")
st.write("Find the perfect drink for your favorite cake!")

# Cargar modelo y columnas
model = joblib.load("models/random_forest.pkl")
features = joblib.load("models/features.pkl")

# Seleccionar pastel
cake = st.selectbox(
    "Choose your cake:",
    ["gateau chocolat", "cheese cake"]
)

# Botón para recomendar
if st.button("Recommend a drink"):
    # Preparar datos para el modelo
    input_data = pd.DataFrame(
        [[int(column == cake) for column in features]],
        columns=features
    )

    # Obtener recomendación
    beverage = model.predict(input_data)[0]

    st.success(f"☕ Recommended drink: {beverage}")
