#!/usr/bin/env python
# coding: utf-8

# # Data uploading

# In[1]:


import kagglehub
import pandas as pd


# Download latest version
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/Mall_Customers.csv")
df.head()



# # Module importation

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.datasets
import pandas as pd


# # Importation des données

# In[3]:


ds = sklearn.datasets.fetch_california_housing()
df = pd.DataFrame(ds.data, columns=ds.feature_names)
df['MedHouseVal'] = ds.target
df.head()


# # Nettoyage et exploration des données

# In[4]:


print(df.isnull().sum())
print(df.dtypes)
print("Nombre de data dupliquer: ", df.duplicated().sum())


# # EDA

# ## Statistiques descriptives

# In[5]:


### Statistiques descriptives
print(df.describe())

# Histogrammes
df.hist(bins=30, figsize=(12, 10))
plt.tight_layout()
plt.show()


# ## Boxplot de la target

# In[6]:


# Boxplot de la target
sns.boxplot(x=df['MedHouseVal'])
plt.title("Distribution des prix médians des maisons")
plt.show()


# ## Scatter plots clés

# In[7]:


# Scatter plots clés
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.3)
plt.title("Prix médian vs Revenu médian")
plt.show()


# ## Matrice de corrélation

# In[8]:


# Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()


# ## Séparation des features (X) et de la target (y)

# In[9]:


# Séparation des features (X) et de la target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# # Normalisation (StandardScaler)

# In[10]:


# Normalisation (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# # Baseline et modélisation

# ## Modèle de base

# In[11]:


# Modèle de base
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


# ## Prédictions sur le validation set

# In[12]:


# Prédictions sur le validation set
y_val_pred = lr.predict(X_val_scaled)


# ## Metriques et Régression Linéaires, MSE, MAE, R²

# In[13]:


# Métriques
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Régression Linéaire - Validation Set:")
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


# # Ridge Regression (L2)

# In[14]:


# Ridge Regression (L2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_val_ridge = ridge.predict(X_val_scaled)
print(f"Ridge R²: {r2_score(y_val, y_val_ridge):.4f}")


# # Lasso Regression (L1)

# In[15]:


# Lasso Regression (L1)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
y_val_lasso = lasso.predict(X_val_scaled)
print(f"Lasso R²: {r2_score(y_val, y_val_lasso):.4f}")


# ## Comparaison des coefficients

# In[16]:


# Comparaison des coefficients
coeffs = pd.DataFrame({
    'Feature': ds.feature_names,
    'Linear': lr.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})
print(coeffs)


# # Evaluation finale

# In[18]:


# Supposons que Ridge soit le meilleur (à vérifier via R²)
y_test_pred = ridge.predict(X_test_scaled)

print("\nPerformance finale sur le test set:")
print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R²: {r2_score(y_test, y_test_pred):.4f}")

# Intervalles de confiance (simplifié)
residuals = y_test - y_test_pred
std_err = np.std(residuals)
conf_int = 1.96 * std_err  # 95% CI
print(f"Intervalle de confiance des erreurs: ±{conf_int:.4f}")



