from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
modelo = joblib.load("modelo_alquiler.pkl")

@app.route("/")
def inicio():
    return "API de predicción de alquileres funcionando"

@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json()
    try:
        habitaciones = int(datos["habitaciones"])
        baños = int(datos["baños"])
        superficie = float(datos["superficie_m2"])
        entrada = np.array([[habitaciones, baños, superficie]])
        prediccion = modelo.predict(entrada)[0]
        return jsonify({"prediccion": round(prediccion, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/entrenar", methods=["POST"])
def entrenar():
    try:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        # Cargar y limpiar datos
        df = pd.read_csv("alquileres.csv")
        df["precio_alquiler"] = pd.to_numeric(df["precio_alquiler"], errors="coerce")
        df = df.dropna()
        df = df[df["precio_alquiler"] < 1000000]
        df = df[df["precio_alquiler"] > 0]
        df = df[df["habitaciones"].apply(lambda x: str(x).isdigit())]
        df = df[df["baños"].apply(lambda x: str(x).isdigit())]
        df = df[df["superficie_m2"].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        df["habitaciones"] = df["habitaciones"].astype(int)
        df["baños"] = df["baños"].astype(int)
        df["superficie_m2"] = df["superficie_m2"].astype(float)

        # Modelo
        X = df[["habitaciones", "baños", "superficie_m2"]]
        y = df["precio_alquiler"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Guardar modelo
        joblib.dump(modelo, "modelo_alquiler.pkl")

        score = modelo.score(X_test, y_test)
        return jsonify({"mensaje": "Modelo entrenado con éxito", "precision": round(score, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
