# SweetMatch

Every cake deserves its perfect match.

## 1. Descripción del proyecto

SweetMatch es una aplicación web que utiliza Machine Learning para recomendar bebidas según el pastel seleccionado por el usuario. Su objetivo es mejorar la experiencia del cliente y facilitar la elección de combinaciones de productos.

## 2. Dataset

Se utilizó un dataset real de ventas de una panadería obtenido de Kaggle. Después del procesamiento y limpieza de los datos, se obtuvieron 2,421 registros de compras.

## 3. Modelos de Machine Learning

Se entrenaron y evaluaron dos modelos:

| Modelo | Accuracy |
|---|---:|
| Decision Tree | 53.85% |
| Random Forest | 53.85% |
| Baseline de popularidad | 53.85% |

Se seleccionó Decision Tree por su menor complejidad. También se implementaron reglas de asociación y un mecanismo de recomendación por popularidad para productos con información insuficiente.

## 4. Tecnologías utilizadas

- Python 3.10
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Pytest
- Ruff
- Git y GitHub Actions

## 5. Servicios de AWS

- Amazon S3: almacenamiento del dataset y modelos entrenados.
- Amazon EC2: alojamiento y ejecución de la aplicación web.
- Amazon CloudWatch: monitoreo de predicciones y tiempos de procesamiento.
- AWS IAM: administración de permisos entre servicios.

## 6. Estructura del proyecto

```text
SweetMatch/
├── .github/workflows/
├── .streamlit/
├── assets/
├── data/
├── models/
├── src/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## 7. Instalación y ejecución

Clonar el repositorio:

```bash
git clone https://github.com/zutzencalvillo-dotcom/SweetMatch.git
cd SweetMatch
```

Crear un entorno virtual con Python 3.10:

```bash
python -m venv venv
```

Activar el entorno virtual en Windows:

```bash
venv\Scripts\activate
```

En Linux:

```bash
source venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python -m streamlit run app.py
```

## 8. Testing y análisis de código

```bash
python -m pytest tests/ -v
python -m ruff check app.py src/ tests/
```

El proyecto cuenta con nueve pruebas automatizadas aprobadas y análisis estático de código mediante Ruff.

## 9. CI/CD

GitHub Actions ejecuta automáticamente las pruebas y el análisis de código cuando se realizan cambios en el repositorio.

Amazon EC2 cuenta con un script y un temporizador de systemd configurados para consultar GitHub cada cinco minutos y desplegar actualizaciones que hayan aprobado la integración continua.

## 10. Monitoreo

Amazon CloudWatch permite consultar los registros de las recomendaciones, incluyendo el pastel seleccionado, la bebida recomendada, el método utilizado y el tiempo de procesamiento.

En tres solicitudes de prueba se obtuvo un tiempo promedio de procesamiento de 7.05 ms.

## 11. Análisis ético y limitaciones

El sistema no requiere información personal de los usuarios. Sin embargo, las recomendaciones pueden presentar sesgos hacia los productos más populares o aquellos con mayor cantidad de registros históricos.

Se recomienda incorporar nuevos datos, ampliar la variedad de productos y evaluar periódicamente el desempeño del sistema. Actualmente no se mide la precisión de las recomendaciones en producción mediante retroalimentación de los clientes.

## 12. Demostración

Aplicación desplegada en Amazon EC2:

http://34.229.158.206:8501

La aplicación está disponible mientras la instancia se encuentre en ejecución. La dirección IP pública puede cambiar al reiniciar la instancia mediante una detención y un nuevo inicio.

---

SweetMatch | Proyecto académico de Machine Learning | 2026