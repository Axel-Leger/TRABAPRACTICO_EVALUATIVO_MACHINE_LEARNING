import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Cargar datos
df = pd.read_csv("alquileres.csv")

# Convertir precio_alquiler a numérico
df["precio_alquiler"] = pd.to_numeric(df["precio_alquiler"], errors="coerce")

# Limpiar datos básicos (descartamos errores evidentes)
df = df.dropna()
df = df[df["precio_alquiler"] > 0]
df = df[df["habitaciones"].apply(lambda x: str(x).isdigit())]
df = df[df["baños"].apply(lambda x: str(x).isdigit())]
df = df[df["superficie_m2"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

df["habitaciones"] = df["habitaciones"].astype(int)
df["baños"] = df["baños"].astype(int)
df["superficie_m2"] = df["superficie_m2"].astype(float)

# Variables
X = df[["habitaciones", "baños", "superficie_m2"]]
y = df["precio_alquiler"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Guardar modelo
joblib.dump(modelo, "modelo_alquiler.pkl")

# Mostrar precisión
score = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {score:.2f}")
