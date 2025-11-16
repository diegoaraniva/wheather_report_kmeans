import streamlit as st
import requests
import pandas as pd
import json
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import umap

# Configuración de Streamlit
st.set_page_config(
    page_title="Análisis Climático - K-Means & DBSCAN",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Análisis Climático: Comparación de Modelos")
st.markdown("### Clustering con PCA+K-Means vs UMAP+DBSCAN")

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
with st.spinner('Obteniendo datos de la API...'):
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        st.success("✓ Datos obtenidos exitosamente")
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            st.info(f"Total de registros obtenidos: {len(df)}")
        else:
            st.error("No se encontraron datos en la respuesta")
            st.stop()
    
    except requests.exceptions.HTTPError as e:
        st.error(f"✗ Error HTTP: {e}")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"✗ Error en la petición: {e}")
        st.stop()
    except json.JSONDecodeError:
        st.error("✗ Error al decodificar JSON")
        st.stop()

# Información inicial
with st.expander("📊 Ver información inicial del dataset"):
    st.write("**Columnas disponibles:**")
    st.write(df.columns.tolist())
    st.dataframe(df.head())

# Limpieza y preparación de datos para ML
st.header("🧹 Limpieza y Preparación de Datos")

with st.spinner('Limpiando datos...'):
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

    # Crear una copia para no modificar los datos originales
    df_clean = df.copy()

    # 1. Manejo de valores nulos
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

    st.success(f"✓ Datos limpios: {len(df_clean)} filas, {df_clean.shape[1]} columnas")
    
    with st.expander("📊 Ver resumen de limpieza"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros originales", len(df))
            st.metric("Registros limpios", len(df_clean))
        with col2:
            st.metric("Columnas finales", df_clean.shape[1])
        st.dataframe(df_clean.head())

# Tabs para los dos modelos
tab1, tab2 = st.tabs(["🔵 Modelo 1: PCA + K-Means", "🟢 Modelo 2: UMAP + DBSCAN"])

# ==================== MODELO 1: PCA + K-MEANS ====================
with tab1:
    st.header("🔵 Modelo 1: PCA + K-Means")    # Preparar datos para PCA (excluir la columna de fecha)
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

    st.subheader("📉 Análisis de Componentes Principales (PCA)")
    
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
    st.pyplot(fig)

    # 5. Determinar número óptimo de componentes (95% varianza)
    n_components_95 = np.argmax(varianza_acumulada >= 0.95) + 1
    st.info(f"Número de componentes para explicar 95% de varianza: **{n_components_95}**")

    # 6. Crear DataFrame con componentes principales reducidos
    pca_reduced = PCA(n_components=n_components_95)
    df_pca_reduced = pca_reduced.fit_transform(df_scaled)

    # Crear DataFrame con los componentes principales
    columnas_pca = [f'PC{i+1}' for i in range(n_components_95)]
    df_final = pd.DataFrame(df_pca_reduced, columns=columnas_pca)
    df_final['fecha'] = df_clean['fecha'].values

    st.subheader("🎯 Clustering K-Means")
    
    # 1. Determinar el número óptimo de clusters usando el método del codo
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)

    with st.spinner('Calculando número óptimo de clusters...'):
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
    st.pyplot(fig)

    # 2. Aplicar K-Means con el número óptimo de clusters
    k_optimo = K_range[np.argmax(silhouette_scores)]
    st.success(f"Número óptimo de clusters: **{k_optimo}**")

    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(df_pca_reduced)

    # 3. Agregar etiquetas de cluster al DataFrame
    df_final['cluster'] = clusters

    # 4. Evaluar calidad de clustering
    silhouette = silhouette_score(df_pca_reduced, clusters)
    davies_bouldin = davies_bouldin_score(df_pca_reduced, clusters)
    calinski_harabasz = calinski_harabasz_score(df_pca_reduced, clusters)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{silhouette:.3f}", help="Más cercano a 1 es mejor")
    with col2:
        st.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", help="Más cercano a 0 es mejor")
    with col3:
        st.metric("Calinski-Harabasz", f"{calinski_harabasz:.1f}", help="Mayor es mejor")

    # 5. Distribución de clusters
    st.write("**Distribución de observaciones por cluster:**")
    dist_df = df_final['cluster'].value_counts().sort_index().to_frame()
    st.dataframe(dist_df)

    # 6. Visualizar clusters en 2D (primeros 2 componentes principales)
    fig = plt.figure(figsize=(10, 7))
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
    st.pyplot(fig)

    # 7. Características de cada cluster
    st.subheader("📊 Características promedio por cluster")
    df_clusters = df_clean.copy()
    df_clusters['cluster'] = clusters

    for cluster_id in range(k_optimo):
        with st.expander(f"Cluster {cluster_id}"):
            cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
            st.write(f"**Número de días:** {len(cluster_data)}")
            st.dataframe(cluster_data.select_dtypes(include=[np.number]).mean().round(2))
    
    # 8. Reglas de Asociación para K-Means
    st.subheader("🔗 Reglas de Asociación")
    
    with st.spinner('Generando reglas de asociación...'):
        # Preparar datos para reglas de asociación
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
            if mes in [12, 1, 2]: return 'verano'
            elif mes in [3, 4, 5]: return 'transicion_seca'
            elif mes in [6, 7, 8]: return 'invierno'
            else: return 'transicion_lluviosa'

        df_asociacion['estacion'] = df_asociacion['mes'].apply(obtener_estacion)

        # Crear transacciones
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

        # Codificar transacciones
        te = TransactionEncoder()
        te_array = te.fit(transacciones).transform(transacciones)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Encontrar itemsets frecuentes
        frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

        # Generar reglas de asociación
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values('lift', ascending=False)

        # Filtrar reglas significativas
        reglas_significativas = rules[
            (rules['lift'] > 1.2) &
            (rules['confidence'] > 0.6) &
            (rules['support'] > 0.05)
        ].copy()

        # Formatear para visualización
        reglas_significativas['antecedents'] = reglas_significativas['antecedents'].apply(lambda x: ', '.join(list(x)))
        reglas_significativas['consequents'] = reglas_significativas['consequents'].apply(lambda x: ', '.join(list(x)))

        st.success(f"✓ Reglas significativas encontradas: {len(reglas_significativas)}")
        
        # Mostrar top reglas
        columnas_mostrar = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        st.dataframe(reglas_significativas[columnas_mostrar].head(20))

        # Visualizar reglas
        if len(reglas_significativas) > 0:
            top_rules = reglas_significativas.head(15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Gráfico Soporte vs Confianza
            scatter = ax1.scatter(top_rules['support'], top_rules['confidence'],
                                 s=top_rules['lift']*50, c=top_rules['lift'],
                                 cmap='YlOrRd', alpha=0.6, edgecolors='black')
            ax1.set_xlabel('Soporte')
            ax1.set_ylabel('Confianza')
            ax1.set_title('Reglas: Soporte vs Confianza (color=Lift)')
            plt.colorbar(scatter, ax=ax1, label='Lift')
            ax1.grid(True, alpha=0.3)

            # Gráfico de barras
            reglas_labels = [f"{ant[:30]}→{cons[:30]}"
                            for ant, cons in zip(top_rules['antecedents'], top_rules['consequents'])]
            ax2.barh(range(len(top_rules)), top_rules['lift'].values)
            ax2.set_yticks(range(len(top_rules)))
            ax2.set_yticklabels(reglas_labels, fontsize=8)
            ax2.set_xlabel('Lift')
            ax2.set_title('Top 15 Reglas por Lift')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            st.pyplot(fig)

# ==================== MODELO 2: UMAP + DBSCAN ====================
with tab2:
    st.header("🟢 Modelo 2: UMAP + DBSCAN")
    
    # Preparar datos para UMAP
    st.subheader("🗺️ Reducción con UMAP")
    
    with st.spinner('Aplicando UMAP...'):
        df_umap_input = df_clean.select_dtypes(include=[np.number]).copy()
        
        # Estandarizar
        scaler_umap = StandardScaler()
        df_scaled_umap = scaler_umap.fit_transform(df_umap_input)
        
        # Aplicar UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        df_umap_reduced = reducer.fit_transform(df_scaled_umap)
        
        # Crear DataFrame
        df_umap_final = pd.DataFrame(df_umap_reduced, columns=['UMAP1', 'UMAP2'])
        df_umap_final['fecha'] = df_clean['fecha'].values
        
        st.success("✓ Reducción UMAP completada")
        
        # Visualizar UMAP
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(df_umap_final['UMAP1'], df_umap_final['UMAP2'],
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=30, c='steelblue')
        plt.xlabel('UMAP Componente 1')
        plt.ylabel('UMAP Componente 2')
        plt.title('Reducción de Dimensionalidad con UMAP')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Clustering DBSCAN
    st.subheader("🎯 Clustering DBSCAN")
    
    with st.spinner('Aplicando DBSCAN...'):
        X_umap = df_umap_reduced
        
        # Calcular eps óptimo
        min_samples = 5
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_umap)
        distances, _ = nbrs.kneighbors(X_umap)
        k_distances = np.sort(distances[:, -1])
        
        # Visualizar k-dist
        fig = plt.figure(figsize=(10, 5))
        plt.plot(k_distances)
        plt.ylabel(f'{min_samples}-NN distancia')
        plt.xlabel('Puntos ordenados')
        plt.title('k-dist plot - Buscar el "codo" para elegir eps')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=np.percentile(k_distances, 90), color='r', linestyle='--',
                    label=f'Percentil 90: {np.percentile(k_distances, 90):.3f}')
        plt.legend()
        st.pyplot(fig)
        
        # Elegir eps
        eps_optimo = float(np.percentile(k_distances, 90))
        st.info(f"eps sugerido (percentil 90): **{eps_optimo:.4f}**")
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=eps_optimo, min_samples=min_samples, metric='euclidean')
        labels_dbscan = dbscan.fit_predict(X_umap)
        
        # Agregar clusters
        df_umap_final['dbscan_cluster'] = labels_dbscan
        
        # Resumen
        n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        n_noise = int((labels_dbscan == -1).sum())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clusters encontrados", n_clusters)
        with col2:
            st.metric("Puntos de ruido", f"{n_noise} ({n_noise/len(labels_dbscan)*100:.1f}%)")
        
        # Métricas de calidad
        valid_mask = labels_dbscan != -1
        if n_clusters >= 2 and valid_mask.sum() > 0:
            silhouette_db = silhouette_score(X_umap[valid_mask], labels_dbscan[valid_mask])
            davies_bouldin_db = davies_bouldin_score(X_umap[valid_mask], labels_dbscan[valid_mask])
            calinski_db = calinski_harabasz_score(X_umap[valid_mask], labels_dbscan[valid_mask])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{silhouette_db:.3f}")
            with col2:
                st.metric("Davies-Bouldin", f"{davies_bouldin_db:.3f}")
            with col3:
                st.metric("Calinski-Harabasz", f"{calinski_db:.1f}")
        
        # Visualizar clusters DBSCAN
        fig = plt.figure(figsize=(12, 8))
        
        # Graficar ruido
        noise_mask = labels_dbscan == -1
        if noise_mask.sum() > 0:
            plt.scatter(df_umap_final.loc[noise_mask, 'UMAP1'],
                        df_umap_final.loc[noise_mask, 'UMAP2'],
                        c='lightgray', s=20, alpha=0.4, label='Ruido',
                        edgecolors='black', linewidth=0.2)
        
        # Graficar clusters
        for cluster_id in sorted(set(labels_dbscan)):
            if cluster_id == -1:
                continue
            cluster_mask = labels_dbscan == cluster_id
            plt.scatter(df_umap_final.loc[cluster_mask, 'UMAP1'],
                        df_umap_final.loc[cluster_mask, 'UMAP2'],
                        s=50, alpha=0.7, label=f'Cluster {cluster_id}',
                        edgecolors='black', linewidth=0.3)
        
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title(f'DBSCAN sobre UMAP (eps={eps_optimo:.3f}, min_samples={min_samples})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Características por cluster
        st.subheader("📊 Características promedio por cluster")
        df_dbscan_clusters = df_clean.copy()
        df_dbscan_clusters['dbscan_cluster'] = labels_dbscan
        
        for cluster_id in sorted(set(labels_dbscan)):
            label = "Ruido" if cluster_id == -1 else f"Cluster {cluster_id}"
            with st.expander(label):
                cluster_data = df_dbscan_clusters[df_dbscan_clusters['dbscan_cluster'] == cluster_id]
                st.write(f"**Número de días:** {len(cluster_data)}")
                st.dataframe(cluster_data.select_dtypes(include=[np.number]).mean().round(2))
    
    # Reglas de Asociación para DBSCAN
    st.subheader("🔗 Reglas de Asociación (DBSCAN)")
    
    with st.spinner('Generando reglas de asociación para DBSCAN...'):
        df_asociacion_umap = df_clean.copy()
        df_asociacion_umap['dbscan_cluster'] = labels_dbscan
        
        # Aplicar discretización
        df_asociacion_umap['cat_temp'] = df_asociacion_umap['temp_promedio'].apply(discretizar_temp)
        df_asociacion_umap['cat_precipitacion'] = df_asociacion_umap['precipitacion'].apply(discretizar_precipitacion)
        df_asociacion_umap['cat_viento'] = df_asociacion_umap['velocidad_viento'].apply(discretizar_viento)
        df_asociacion_umap['cat_cluster'] = df_asociacion_umap['dbscan_cluster'].apply(
            lambda x: 'ruido' if x == -1 else f'cluster_{x}'
        )
        
        # Estación
        df_asociacion_umap['mes'] = pd.to_datetime(df_asociacion_umap['fecha']).dt.month
        df_asociacion_umap['estacion'] = df_asociacion_umap['mes'].apply(obtener_estacion)
        
        # Crear transacciones
        transacciones_umap = []
        for _, row in df_asociacion_umap.iterrows():
            transaccion = [
                row['cat_temp'],
                row['cat_precipitacion'],
                row['cat_viento'],
                row['cat_cluster'],
                row['estacion']
            ]
            transacciones_umap.append(transaccion)
        
        # Codificar
        te_umap = TransactionEncoder()
        te_array_umap = te_umap.fit(transacciones_umap).transform(transacciones_umap)
        df_encoded_umap = pd.DataFrame(te_array_umap, columns=te_umap.columns_)
        
        # Apriori
        frequent_itemsets_umap = apriori(df_encoded_umap, min_support=0.05, use_colnames=True)
        
        # Reglas
        rules_umap = association_rules(
            frequent_itemsets_umap,
            metric="confidence",
            min_threshold=0.5
        )
        rules_umap = rules_umap.sort_values('confidence', ascending=False)
        
        # Filtrar
        reglas_significativas_umap = rules_umap[
            (rules_umap['confidence'] > 0.6) &
            (rules_umap['lift'] > 1.1) &
            (rules_umap['support'] > 0.03)
        ].copy()
        
        # Formatear
        reglas_significativas_umap['antecedents'] = reglas_significativas_umap['antecedents'].apply(
            lambda x: ', '.join(list(x))
        )
        reglas_significativas_umap['consequents'] = reglas_significativas_umap['consequents'].apply(
            lambda x: ', '.join(list(x))
        )
        
        st.success(f"✓ Reglas significativas encontradas: {len(reglas_significativas_umap)}")
        
        # Mostrar reglas
        columnas_mostrar = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        st.dataframe(reglas_significativas_umap[columnas_mostrar].head(25))
        
        # Visualizaciones
        if len(reglas_significativas_umap) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            top_20 = reglas_significativas_umap.head(20)
            
            # Support vs Confidence
            axes[0, 0].scatter(top_20['support'], top_20['confidence'],
                               s=top_20['lift']*100, c=top_20['lift'],
                               cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=0.5)
            axes[0, 0].set_xlabel('Support')
            axes[0, 0].set_ylabel('Confidence')
            axes[0, 0].set_title('Support vs Confidence (tamaño y color = Lift)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Top 15 por Confidence
            top_15 = reglas_significativas_umap.head(15)
            reglas_labels = [f"{ant[:25]}→{cons[:25]}"
                            for ant, cons in zip(top_15['antecedents'], top_15['consequents'])]
            axes[0, 1].barh(range(len(top_15)), top_15['confidence'].values, color='steelblue', edgecolor='black')
            axes[0, 1].set_yticks(range(len(top_15)))
            axes[0, 1].set_yticklabels(reglas_labels, fontsize=8)
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_title('Top 15 Reglas por Confidence')
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Lift vs Confidence
            axes[1, 0].scatter(top_20['confidence'], top_20['lift'],
                               s=top_20['support']*1000, c='coral',
                               alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Lift')
            axes[1, 0].set_title('Confidence vs Lift (tamaño = Support)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Distribución Confidence
            axes[1, 1].hist(reglas_significativas_umap['confidence'], bins=20,
                            edgecolor='black', alpha=0.7, color='mediumseagreen')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].set_title('Distribución de Confidence')
            axes[1, 1].axvline(reglas_significativas_umap['confidence'].mean(),
                               color='red', linestyle='--', linewidth=2,
                               label=f'Media: {reglas_significativas_umap["confidence"].mean():.2f}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)

# Comparación de modelos
st.header("📊 Comparación de Modelos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 PCA + K-Means")
    st.metric("Número de Clusters", k_optimo)
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.metric("Reglas Significativas", len(reglas_significativas))

with col2:
    st.subheader("🟢 UMAP + DBSCAN")
    st.metric("Número de Clusters", n_clusters)
    st.metric("Puntos de Ruido", n_noise)
    if n_clusters >= 2 and valid_mask.sum() > 0:
        st.metric("Silhouette Score", f"{silhouette_db:.3f}")
    st.metric("Reglas Significativas", len(reglas_significativas_umap))

st.markdown("---")
st.markdown("### 💡 Interpretación")
st.markdown("""
- **PCA + K-Means**: Método clásico que asume clusters esféricos y requiere definir k de antemano.
- **UMAP + DBSCAN**: Captura estructuras no lineales y encuentra clusters automáticamente, identificando ruido.
- **Reglas de Asociación**: Revelan patrones entre condiciones climáticas y grupos encontrados.
""")
