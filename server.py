from flask import Flask, request, jsonify
import joblib
import pandas as pd
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
        entrada = pd.DataFrame([{
            "habitaciones": int(datos["habitaciones"]),
            "baños": int(datos["baños"]),
            "tipo_pago": datos["tipo_pago"],
            "tipo": datos["tipo"],
            "localidad": datos["localidad"]
        }])

        # Hacer la predicción
        prediccion = modelo.predict(entrada)[0]


        # Leer el dataset para buscar ejemplos similares
        df = pd.read_csv("alquileres_modificado.csv")
        df = df[df["localidad"] == datos["localidad"]]
        df = df[df["tipo"] == datos["tipo"]]
        df["habitaciones"] = pd.to_numeric(df["habitaciones"], errors="coerce")
        df = df[df["habitaciones"].notna()]
        df["habitaciones"] = df["habitaciones"].astype(int)

        # Elegimos hasta 3 filas que tengan misma cantidad de habitaciones
        ejemplos = df[df["habitaciones"] == int(datos["habitaciones"])].head(3)

        ejemplos_json = ejemplos[["tipo", "habitaciones", "baños", "tipo_pago", "precio_alquiler"]].to_dict(orient="records")

        return jsonify({
            "prediccion": round(prediccion, 2),
            "ejemplos": ejemplos_json
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/entrenar", methods=["POST"])
def entrenar():
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

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

        # Quitamos superficie_m2 del dataset y modelo
        X = df[["habitaciones", "baños", "tipo_pago", "tipo", "localidad"]]
        y = df["precio_alquiler"]

        preprocesador = ColumnTransformer([
            ("tipo_pago", OneHotEncoder(handle_unknown='ignore'), ["tipo_pago"]),
            ("tipo", OneHotEncoder(handle_unknown='ignore'), ["tipo"]),
            ("localidad", OneHotEncoder(handle_unknown='ignore'), ["localidad"])
        ], remainder='passthrough')


        modelo_local = Pipeline([
            ("preprocesamiento", preprocesador),
            ("regresor", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        modelo_local.fit(X_train, y_train)

        joblib.dump(modelo_local, "modelo_alquiler.pkl")

        return jsonify({
            "mensaje": "Modelo entrenado con éxito",
            "precision": round(modelo_local.score(X_test, y_test), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
