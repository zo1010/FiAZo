# CONCLUSION

## Interprétation des résultats 

1) Variables importantes (via Lasso) :

    - MedInc (revenu médian) a le poids le plus élevé → impact positif sur le prix.

    - AveOccup (occupation moyenne) a un poids négatif → plus d'occupants → prix plus bas.


2) Performance :
- R² ~0.6 → 60% de la variance expliquée. Améliorable avec des features supplémentaires (ex: distance aux emplois).


3) Pistes d'amélioration
    1) Feature Engineering :
- Créer des interactions (ex: MedInc * HouseAge).

- Ajouter des données externes (ex: proximité écoles, notes étoilés, quartiers évaluation).

    2) Modèles avancés :

- Essayer Random Forest ou XGBoost pour capturer des non-linéarités.

    3) Optimisation :

- GridSearch pour tuner les hyperparamètres de Ridge/Lasso.