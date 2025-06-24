import pandas as pd

# Cargar el archivo CSV original
df = pd.read_csv("alquileres_modificado.csv")

# Convertir superficie a numérico por si viene como texto
df["superficie_m2"] = pd.to_numeric(df["superficie_m2"], errors="coerce")

# Función para calcular precio base según tipo de inmueble
def calcular_precio_base(row):
    if row["tipo"] == "Casa":
        return row["superficie_m2"] * 1800
    elif row["tipo"] == "PH":
        return row["superficie_m2"] * 1600
    else:  # Departamento
        return row["superficie_m2"] * 1500

# Ajuste por localidad
def ajustar_por_localidad(precio, localidad):
    if localidad == "Palermo":
        return precio * 1.25
    elif localidad == "Villa Crespo":
        return precio * 1.15
    elif localidad == "San Telmo":
        return precio * 0.95
    else:
        return precio

# Ajuste por tipo de pago
def ajustar_por_pago(precio, tipo_pago):
    if tipo_pago == "diario":
        return precio / 30
    elif tipo_pago == "anual":
        return precio * 12
    else:  # mensual
        return precio

# Aplicar cálculos
df["precio_base"] = df.apply(calcular_precio_base, axis=1)
df["ajustado_localidad"] = df.apply(lambda row: ajustar_por_localidad(row["precio_base"], row["localidad"]), axis=1)
df["nuevo_precio_alquiler"] = df.apply(lambda row: ajustar_por_pago(row["ajustado_localidad"], row["tipo_pago"]), axis=1)

# Reemplazar precios y redondear
df["precio_alquiler"] = df["nuevo_precio_alquiler"].round(2)

# Eliminar columnas auxiliares si no las necesitás
df = df.drop(columns=["precio_base", "ajustado_localidad", "nuevo_precio_alquiler"])

# Guardar nuevo archivo
df.to_csv("alquileres_ajustados.csv", index=False)

print("Archivo guardado como alquileres_ajustados.csv")
