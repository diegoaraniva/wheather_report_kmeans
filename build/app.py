import requests
import pandas as pd
import json
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar variables de entorno
load_dotenv()

# Configuración de la API desde variables de entorno
API_KEY = os.getenv("API_KEY")
API_HOST = os.getenv("API_HOST")
BASE_URL = "https://meteostat.p.rapidapi.com/stations/daily"

# Parámetros de la consulta desde variables de entorno
params = {
    "station": os.getenv("STATION", "10637"),
    "start": os.getenv("START_DATE", "2015-03-01"),
    "end": os.getenv("END_DATE", "2025-01-01"),
    "tz": os.getenv("TIMEZONE", "America/El_Salvador")
}

# Headers requeridos por RapidAPI
headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# Realizar la petición
try:
    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()

    # Obtener los datos
    data = response.json()

    print("✓ Datos obtenidos exitosamente")
    print(f"Status Code: {response.status_code}")

    # Convertir a DataFrame de pandas para análisis
    if 'data' in data:
        df = pd.DataFrame(data['data'])

        # Convertir la columna de fecha a datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        print(f"\nTotal de registros: {len(df)}")
        print("\nPrimeras filas:")
        print(df.head())

    else:
        print("Respuesta completa:", json.dumps(data, indent=2))

except requests.exceptions.HTTPError as e:
    print(f"✗ Error HTTP: {e}")
    print(f"Respuesta: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"✗ Error en la petición: {e}")
except json.JSONDecodeError:
    print(f"✗ Error al decodificar JSON. Respuesta: {response.text}")

# Mostrar información del DataFrame
print("\nColumnas disponibles:")
print(df.columns.tolist())

print("\nInformación del DataFrame:")
print(df.info())

# Limpieza y preparación de datos para ML

# Renombrar columnas al español
column_mapping = {
    'date': 'fecha',
    'tavg': 'temp_promedio',
    'tmin': 'temp_minima',
    'tmax': 'temp_maxima',
    'prcp': 'precipitacion',
    'snow': 'nieve',
    'wdir': 'direccion_viento',
    'wspd': 'velocidad_viento',
    'wpgt': 'rafaga_viento',
    'pres': 'presion_atmosferica',
    'tsun': 'horas_sol'
}

df = df.rename(columns=column_mapping)

print("Columnas renombradas:")
print(df.columns.tolist())

# Crear una copia para no modificar los datos originales
df_clean = df.copy()

# 1. Manejo de valores nulos
print("Valores nulos por columna:")
print(df_clean.isnull().sum())

# 2. Eliminar columnas con demasiados valores nulos (>50%)
threshold = len(df_clean) * 0.5
df_clean = df_clean.dropna(thresh=threshold, axis=1)

# 3. Rellenar valores nulos en columnas numéricas con la mediana
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# 4. Eliminar filas con valores nulos restantes
df_clean = df_clean.dropna()

# 5. Eliminar duplicados basados en la fecha
df_clean = df_clean.drop_duplicates(subset=['fecha'], keep='first')

# 6. Ordenar por fecha
df_clean = df_clean.sort_values('fecha').reset_index(drop=True)

# 7. Verificar outliers usando IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers)

print("\nOutliers detectados por columna:")
for col in numeric_columns:
    if col in df_clean.columns:
        n_outliers = detect_outliers_iqr(df_clean, col)
        print(f"{col}: {n_outliers} outliers")

# 8. Mostrar columnas disponibles después de la limpieza
print("\nColumnas disponibles después de limpieza:")
print(df_clean.columns.tolist())

# 9. Información del DataFrame limpio
print("\nInformación del DataFrame limpio:")
print(df_clean.info())

# 10. Resumen final
print(f"\nDatos originales: {len(df)} filas")
print(f"Datos limpios: {len(df_clean)} filas")
print(f"Columnas finales: {df_clean.shape[1]}")
print("\nPrimeras filas del dataset limpio:")
print(df_clean.head())

# Preparar datos para PCA (excluir la columna de fecha)
df_pca = df_clean.select_dtypes(include=[np.number]).copy()

# 1. Estandarizar los datos (importante para PCA)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

# 2. Aplicar PCA
pca = PCA()
pca_features = pca.fit_transform(df_scaled)

# 3. Analizar la varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

print("Varianza explicada por cada componente:")
for i, var in enumerate(varianza_explicada, 1):
    print(f"PC{i}: {var*100:.2f}%")

print(f"\nVarianza acumulada:")
for i, var in enumerate(varianza_acumulada, 1):
    print(f"PC{i}: {var*100:.2f}%")

# 4. Visualizar la varianza explicada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de varianza por componente
ax1.bar(range(1, len(varianza_explicada) + 1), varianza_explicada)
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Varianza Explicada')
ax1.set_title('Varianza Explicada por Componente')
ax1.set_xticks(range(1, len(varianza_explicada) + 1))

# Gráfico de varianza acumulada
ax2.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, 'bo-')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
ax2.set_xlabel('Número de Componentes')
ax2.set_ylabel('Varianza Acumulada')
ax2.set_title('Varianza Acumulada')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 5. Determinar número óptimo de componentes (95% varianza)
n_components_95 = np.argmax(varianza_acumulada >= 0.95) + 1
print(f"\nNúmero de componentes para explicar 95% de varianza: {n_components_95}")

# 6. Crear DataFrame con componentes principales reducidos
pca_reduced = PCA(n_components=n_components_95)
df_pca_reduced = pca_reduced.fit_transform(df_scaled)

# Crear DataFrame con los componentes principales
columnas_pca = [f'PC{i+1}' for i in range(n_components_95)]
df_final = pd.DataFrame(df_pca_reduced, columns=columnas_pca)
df_final['fecha'] = df_clean['fecha'].values

print(f"\nDimensiones originales: {df_pca.shape[1]} features")
print(f"Dimensiones reducidas: {n_components_95} componentes principales")
print(f"\nPrimeras filas del dataset reducido:")
print(df_final.head())

# 1. Determinar el número óptimo de clusters usando el método del codo
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_pca_reduced)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_pca_reduced, kmeans.labels_))

# Visualizar métricas para elegir k óptimo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Método del codo
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inercia')
ax1.set_title('Método del Codo')
ax1.grid(True)

# Silhouette Score
ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score por Cluster')
ax2.grid(True)

plt.tight_layout()
plt.show()

# 2. Aplicar K-Means con el número óptimo de clusters
k_optimo = K_range[np.argmax(silhouette_scores)]
print(f"Número óptimo de clusters sugerido: {k_optimo}")

kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(df_pca_reduced)

# 3. Agregar etiquetas de cluster al DataFrame
df_final['cluster'] = clusters

# 4. Evaluar calidad de clustering
silhouette = silhouette_score(df_pca_reduced, clusters)
davies_bouldin = davies_bouldin_score(df_pca_reduced, clusters)
calinski_harabasz = calinski_harabasz_score(df_pca_reduced, clusters)

print("\nMétricas de calidad del clustering:")
print(f"Silhouette Score: {silhouette:.3f} (más cercano a 1 es mejor)")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (más cercano a 0 es mejor)")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f} (mayor es mejor)")

# 5. Distribución de clusters
print("\nDistribución de observaciones por cluster:")
print(df_final['cluster'].value_counts().sort_index())

# 6. Visualizar clusters en 2D (primeros 2 componentes principales)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_final['PC1'], df_final['PC2'],
                     c=df_final['cluster'], cmap='viridis',
                     alpha=0.6, edgecolors='black', linewidth=0.5)
plt.scatter(kmeans_final.cluster_centers_[:, 0],
           kmeans_final.cluster_centers_[:, 1],
           c='red', marker='X', s=200, edgecolors='black',
           linewidth=2, label='Centroides')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Clusters K-Means (k={k_optimo})')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Características de cada cluster
print("\nCaracterísticas promedio por cluster (datos originales):")
df_clusters = df_clean.copy()
df_clusters['cluster'] = clusters

for cluster_id in range(k_optimo):
    print(f"\n--- Cluster {cluster_id} ---")
    cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
    print(f"Número de días: {len(cluster_data)}")
    print(cluster_data.select_dtypes(include=[np.number]).mean().round(2))

print("\nPrimeras filas con clusters asignados:")
print(df_final.head(10))

# 1. Preparar datos para reglas de asociación (discretizar variables continuas)
df_asociacion = df_clean.copy()
df_asociacion['cluster'] = clusters

# Discretizar variables numéricas en categorías
def discretizar_temp(x):
    if x < 15: return 'temp_baja'
    elif x < 25: return 'temp_media'
    else: return 'temp_alta'

def discretizar_precipitacion(x):
    if x == 0: return 'sin_lluvia'
    elif x < 5: return 'lluvia_ligera'
    else: return 'lluvia_fuerte'

def discretizar_viento(x):
    if pd.isna(x): return 'viento_na'
    elif x < 10: return 'viento_calmo'
    elif x < 20: return 'viento_moderado'
    else: return 'viento_fuerte'

# Aplicar discretización
df_asociacion['cat_temp'] = df_asociacion['temp_promedio'].apply(discretizar_temp)
df_asociacion['cat_precipitacion'] = df_asociacion['precipitacion'].apply(discretizar_precipitacion)
df_asociacion['cat_viento'] = df_asociacion['velocidad_viento'].apply(discretizar_viento)
df_asociacion['cat_cluster'] = df_asociacion['cluster'].apply(lambda x: f'cluster_{x}')

# Agregar estación del año
df_asociacion['mes'] = pd.to_datetime(df_asociacion['fecha']).dt.month
def obtener_estacion(mes):
    if mes in [12, 1, 2]: return 'verano'  # El Salvador
    elif mes in [3, 4, 5]: return 'transicion_seca'
    elif mes in [6, 7, 8]: return 'invierno'
    else: return 'transicion_lluviosa'

df_asociacion['estacion'] = df_asociacion['mes'].apply(obtener_estacion)

# 2. Crear transacciones (cada fila es una transacción con items)
transacciones = []
for _, row in df_asociacion.iterrows():
    transaccion = [
        row['cat_temp'],
        row['cat_precipitacion'],
        row['cat_viento'],
        row['cat_cluster'],
        row['estacion']
    ]
    transacciones.append(transaccion)

# 3. Codificar transacciones
te = TransactionEncoder()
te_array = te.fit(transacciones).transform(transacciones)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("Transacciones creadas:")
print(f"Total de transacciones: {len(transacciones)}")
print(f"Items únicos: {len(te.columns_)}")
print("\nItems disponibles:")
print(list(te.columns_))

# 4. Encontrar itemsets frecuentes con Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
print(f"\nItemsets frecuentes encontrados: {len(frequent_itemsets)}")
print("\nTop 10 itemsets más frecuentes:")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# 5. Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False)

print(f"\nReglas de asociación generadas: {len(rules)}")

# 6. Filtrar y mostrar las mejores reglas
reglas_significativas = rules[
    (rules['lift'] > 1.2) &
    (rules['confidence'] > 0.6) &
    (rules['support'] > 0.05)
].copy()

print(f"\nReglas significativas (lift>1.2, conf>0.6, sup>0.05): {len(reglas_significativas)}")

# Formatear para mejor visualización
reglas_significativas['antecedents'] = reglas_significativas['antecedents'].apply(lambda x: ', '.join(list(x)))
reglas_significativas['consequents'] = reglas_significativas['consequents'].apply(lambda x: ', '.join(list(x)))

# Seleccionar columnas relevantes
columnas_mostrar = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
print("\nTop 20 reglas con mayor lift:")
print(reglas_significativas[columnas_mostrar].head(20).to_string(index=False))

# 7. Análisis de reglas por cluster
print("\n=== Reglas específicas por cluster ===")
for cluster_id in range(k_optimo):
    cluster_name = f'cluster_{cluster_id}'
    reglas_cluster = reglas_significativas[
        reglas_significativas['consequents'].str.contains(cluster_name)
    ]
    if len(reglas_cluster) > 0:
        print(f"\n--- Reglas que predicen {cluster_name} ---")
        print(reglas_cluster[columnas_mostrar].head(5).to_string(index=False))

# 8. Visualizar las mejores reglas
top_rules = reglas_significativas.head(15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico Soporte vs Confianza coloreado por Lift
scatter = ax1.scatter(top_rules['support'], top_rules['confidence'],
                     s=top_rules['lift']*50, c=top_rules['lift'],
                     cmap='YlOrRd', alpha=0.6, edgecolors='black')
ax1.set_xlabel('Soporte')
ax1.set_ylabel('Confianza')
ax1.set_title('Reglas de Asociación: Soporte vs Confianza\n(Tamaño y color por Lift)')
plt.colorbar(scatter, ax=ax1, label='Lift')
ax1.grid(True, alpha=0.3)

# Gráfico de barras con las mejores reglas por Lift
reglas_labels = [f"{ant[:30]}... → {cons[:30]}..."
                for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])]
ax2.barh(range(len(top_rules)), top_rules['lift'].values)
ax2.set_yticks(range(len(top_rules)))
ax2.set_yticklabels(reglas_labels, fontsize=8)
ax2.set_xlabel('Lift')
ax2.set_title('Top 15 Reglas por Lift')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# 9. Resumen de insights
print("\n=== RESUMEN DE INSIGHTS ===")
print(f"Total de reglas significativas: {len(reglas_significativas)}")
print(f"\nLift promedio: {reglas_significativas['lift'].mean():.2f}")
print(f"Confianza promedio: {reglas_significativas['confidence'].mean():.2f}")
print(f"Soporte promedio: {reglas_significativas['support'].mean():.2f}")

print("\nInterpretación de métricas:")
print("- Soporte: % de transacciones que contienen el itemset")
print("- Confianza: Probabilidad de que ocurra el consecuente dado el antecedente")
print("- Lift > 1: Indica asociación positiva (items relacionados)")
print("- Lift = 1: Independencia (no hay relación)")
print("- Lift < 1: Asociación negativa (items mutuamente exclusivos)")
