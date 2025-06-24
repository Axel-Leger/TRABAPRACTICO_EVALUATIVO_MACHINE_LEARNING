import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("alquileres_modificado.csv")
df["precio_alquiler"] = pd.to_numeric(df["precio_alquiler"], errors="coerce")
df = df.dropna()

def es_entero(val):
    try:
        int(val)
        return True
    except:
        return False

df = df[df["habitaciones"].apply(es_entero)]
df = df[df["baños"].apply(es_entero)]

df["habitaciones"] = df["habitaciones"].astype(int)
df["baños"] = df["baños"].astype(int)

df = df[df["precio_alquiler"].between(1, 999999)]

X = df[["habitaciones", "baños", "tipo_pago", "tipo", "localidad"]]
y = df["precio_alquiler"]

preprocesador = ColumnTransformer([
    ("tipo_pago", OneHotEncoder(handle_unknown='ignore'), ["tipo_pago"]),
    ("tipo", OneHotEncoder(handle_unknown='ignore'), ["tipo"]),
    ("localidad", OneHotEncoder(handle_unknown='ignore'), ["localidad"])
], remainder='passthrough')


modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("regresor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo.fit(X_train, y_train)

joblib.dump(modelo, "modelo_alquiler.pkl")
print(f"Precisión del modelo: {modelo.score(X_test, y_test):.2f}")
