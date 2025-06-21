# -*- coding: utf-8 -*-
"""
Analyse de clustering des clients d'un centre commercial
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# 1. Chargement des données
def load_data():
    file_path = "C:/Users/zorat/Desktop/Tp1 ML/Mall_Customers.csv"  # À modifier !
    df = pd.read_csv(file_path)
    print("Dimensions du dataset :", df.shape)
    return df

# 2. Prétraitement
def preprocess_data(df):
    # Suppression de la colonne ID
    df = df.drop('CustomerID', axis=1)
    
    # Encodage du genre
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Vérification des NaN
    print("\nValeurs manquantes :")
    print(df.isnull().sum())
    
    return df

# 3. Analyse exploratoire (EDA)
def exploratory_analysis(df):
    # Histogrammes
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.show()

# 4. Clustering K-means
def apply_clustering(df):
    # Sélection des features
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Choix du nombre de clusters
    inertias = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), inertias, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode Elbow')
    plt.show()
    
    # Clustering final (k=5)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualisation silhouette
    plt.figure(figsize=(10, 6))
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X_scaled)
    visualizer.show()
    
    return df

# 5. Visualisation des résultats
def visualize_results(df):
    # Scatter plot 2D
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                    hue='Cluster', palette='viridis', s=100)
    plt.title('Segmentation des clients')
    plt.grid(True)
    plt.show()
    
    # Profils des clusters
    cluster_profile = df.groupby('Cluster').mean()
    print("\nProfils des clusters :")
    print(cluster_profile)

# Exécution principale
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    exploratory_analysis(df)
    df = apply_clustering(df)
    visualize_results(df)
