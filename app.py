
import logging
import os
import time

import joblib
import pandas as pd
import streamlit as st

# Configuracion del monitoreo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SweetMatch")

# Configuracion de la pagina
st.set_page_config(
    page_title="SweetMatch",
    page_icon="🍰",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FFF5EC, #FCE4EC);
}

h1 {
    color: #8B4557 !important;
    text-align: center;
    font-size: 65px !important;
}

h2, h3 {
    color: #8B4557 !important;
}

p, label {
    color: #654438 !important;
}

.stButton > button {
    background-color: #C6758D;
    color: white;
    border: none;
    border-radius: 15px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #A94F6C;
    color: white;
}

[data-testid="stImage"] img {
    border-radius: 20px;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #EAC5CD;
}

[data-testid="stMetricValue"] {
    color: #8B4557;
}

[data-testid="stAlert"] {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Cargar datos y modelo
data = pd.read_csv("data/bakery_clean.csv")
model = joblib.load("models/decision_tree.pkl")
features = joblib.load("models/features.pkl")

cakes = {
    "Chocolate Cake": "gateau chocolat",
    "Cheesecake": "cheese cake",
    "Tiramisu": "tiramisu"
}

beverages = [
    "americano",
    "caffe latte",
    "milk tea",
    "vanila latte",
    "lemon ade",
    "berry ade"
]

cake_images = {
    "Chocolate Cake": "assets/chocolate.png",
    "Cheesecake": "assets/cheesecake.png",
    "Tiramisu": "assets/tiramisu.png"
}

drink_images = {
    "americano": "assets/americano.png",
    "caffe latte": "assets/caffe_latte.png",
    "milk tea": "assets/milk_tea.png",
    "vanila latte": "assets/vanilla_latte.png",
    "lemon ade": "assets/lemon_ade.png",
    "berry ade": "assets/berry_ade.png"
}

drink_names = {
    "americano": "Americano",
    "caffe latte": "Caffe Latte",
    "milk tea": "Milk Tea",
    "vanila latte": "Vanilla Latte",
    "lemon ade": "Lemon Ade",
    "berry ade": "Berry Ade"
}

# Modelo de reglas de asociacion
def recommend_association(cake):

    best_beverage = None
    best_lift = 1

    cake_orders = (data[cake] > 0).sum()

    if cake_orders == 0:
        return None

    for beverage in beverages:

        together = (
            (data[cake] > 0) &
            (data[beverage] > 0)
        ).sum()

        probability = (data[beverage] > 0).mean()

        if probability == 0:
            continue

        confidence = together / cake_orders
        lift = confidence / probability

        if together >= 5 and lift > best_lift:
            best_lift = lift
            best_beverage = beverage

    return best_beverage


# Encabezado
st.title("🍰 SweetMatch")

st.markdown("""
<p style="text-align:center; font-size:22px;">
Every cake deserves its perfect match ♡
</p>
""", unsafe_allow_html=True)

st.divider()

# Menu de pasteles
st.subheader("🧁 Discover our cakes")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(cake_images["Chocolate Cake"],
             use_container_width=True)
    st.subheader("Chocolate Cake")
    st.write("Rich, sweet and delicious.")

with col2:
    st.image(cake_images["Cheesecake"],
             use_container_width=True)
    st.subheader("Cheesecake")
    st.write("Creamy, soft and irresistible.")

with col3:
    st.image(cake_images["Tiramisu"],
             use_container_width=True)
    st.subheader("Tiramisu")
    st.write("A classic Italian dessert.")

st.divider()

# Seleccion del pastel
st.subheader("✨ Find your perfect match")

selected_cake = st.selectbox(
    "Choose your favorite cake:",
    list(cakes.keys())
)

cake = cakes[selected_cake]

# Sistema de recomendacion
if st.button("💗 Find my perfect match"):

    start_time = time.perf_counter()

    beverage = recommend_association(cake)

    if beverage is not None:
        method = "Association Rules"

    elif cake in features:

        input_data = pd.DataFrame(
            [[int(column == cake) for column in features]],
            columns=features
        )

        beverage = model.predict(input_data)[0]
        method = "Decision Tree"

    else:
        beverage = (data[beverages] > 0).sum().idxmax()
        method = "Popularity"

        # Monitoreo de la prediccion
    response_time = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"PREDICTION | "
        f"Cake: {selected_cake} | "
        f"Beverage: {drink_names[beverage]} | "
        f"Method: {method} | "
        f"Response time: {response_time:.2f} ms"
    )

    # Mostrar resultado
    st.balloons()

    st.success(
        f"Your perfect match is: "
        f"{selected_cake} + {drink_names[beverage]}"
    )

    result1, result2 = st.columns(2)

    with result1:
        st.image(
            cake_images[selected_cake],
            use_container_width=True
        )
        st.subheader(selected_cake)

    with result2:

        image_path = drink_images.get(beverage)

        if image_path and os.path.exists(image_path):
            st.image(
                image_path,
                use_container_width=True
            )
        else:
            st.markdown("""
            <div style="text-align:center;
                        font-size:100px;">
                🥤
            </div>
            """, unsafe_allow_html=True)

        st.subheader(drink_names[beverage])

    st.info(f"Recommendation method: {method}")

st.divider()

# Estadisticas del proyecto
st.subheader("📊 SweetMatch in numbers")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Real purchase records", len(data))

with m2:
    st.metric("Cake options", len(cakes))

with m3:
    st.metric("Drink options", len(beverages))

st.divider()

# Informacion del proyecto
st.markdown("""
<div style="text-align:center;">
    <h3>Made with Machine Learning ♡</h3>
    <p>Real data. Sweet moments. Perfect combinations.</p>
    <p>SweetMatch | 2026</p>
</div>
""", unsafe_allow_html=True)
