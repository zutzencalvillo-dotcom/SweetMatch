
import os

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest


# Prueba 1: Verificar el dataset
def test_dataset():
    data = pd.read_csv("data/bakery_clean.csv")
    assert len(data) > 0


# Prueba 2: Verificar las columnas
def test_columns():
    data = pd.read_csv("data/bakery_clean.csv")

    assert "gateau chocolat" in data.columns
    assert "cheese cake" in data.columns
    assert "americano" in data.columns


# Prueba 3: Verificar los modelos guardados
def test_models():
    assert os.path.exists("models/decision_tree.pkl")
    assert os.path.exists("models/random_forest.pkl")
    assert os.path.exists("models/features.pkl")


# Prueba 4: Verificar Decision Tree
def test_decision_tree():
    model = joblib.load("models/decision_tree.pkl")
    features = joblib.load("models/features.pkl")

    input_data = pd.DataFrame(
        [[int(column == "cheese cake") for column in features]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    assert prediction in [
        "americano", "caffe latte", "milk tea",
        "vanila latte", "lemon ade", "berry ade"
    ]


# Prueba 5: Verificar las imagenes
def test_images():
    images = [
        "chocolate.png",
        "cheesecake.png",
        "tiramisu.png",
        "americano.png",
        "caffe_latte.png",
        "milk_tea.png",
        "vanilla_latte.png",
        "lemon_ade.png",
        "berry_ade.png"
    ]

    for image in images:
        assert os.path.exists("assets/" + image)


# Prueba 6: Verificar la aplicacion web
def test_app():
    app = AppTest.from_file("../app.py", default_timeout=30).run()

    assert len(app.exception) == 0


# Prueba 7: Recomendacion de chocolate
def test_chocolate_recommendation():
    app = AppTest.from_file("../app.py", default_timeout=30).run()

    app.selectbox[0].select("Chocolate Cake")
    app.button[0].click().run()

    assert len(app.exception) == 0
    assert "Berry Ade" in app.success[0].value


# Prueba 8: Recomendacion de cheesecake
def test_cheesecake_recommendation():
    app = AppTest.from_file("../app.py", default_timeout=30).run()

    app.selectbox[0].select("Cheesecake")
    app.button[0].click().run()

    assert len(app.exception) == 0
    assert "Vanilla Latte" in app.success[0].value


# Prueba 9: Recomendacion con pocos datos
def test_tiramisu_recommendation():
    app = AppTest.from_file("../app.py", default_timeout=30).run()

    app.selectbox[0].select("Tiramisu")
    app.button[0].click().run()

    assert len(app.exception) == 0
    assert len(app.success) == 1
    assert "Popularity" in app.info[0].value

