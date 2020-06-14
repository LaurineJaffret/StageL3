
import os

import joblib
import pandas as pnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

pnd.set_option('display.max_columns',None)

#chargement des données dans un dataframe
combats = pnd.read_csv("datas/combats.csv")
nosPokemons = pnd.read_csv("datas/pokedex.csv")

#transformation de la colone LEGENDAIRE de booléen à entier (0 = faux et 1 = vrai)
nosPokemons['LEGENDAIRE'] = (nosPokemons['LEGENDAIRE']=='VRAI').astype(int)

#   combien de fois le pokemon a combattu à la premiere et deuxieme position
nbFoisPremierePosition = combats.groupby('First_pokemon').count()
nbFoisSecondePosition = combats.groupby('Second_pokemon').count()
#   combien de fois le pokemon a combatu en tout
nombreTotalDeCombats = nbFoisPremierePosition + nbFoisSecondePosition
#   nombre de victoire par pokemon
nombreDeVictoires = combats.groupby('Winner').count()

#______Agregation des données______

listeAAgreger = combats.groupby('Winner').count()
listeAAgreger.sort_index()
listeAAgreger['NBR_COMBATS'] = nbFoisPremierePosition.Winner + nbFoisSecondePosition.Winner
listeAAgreger['NBR_VICTOIRES'] = nombreDeVictoires.First_pokemon
listeAAgreger['POURCENTAGE_DE_VICTOIRES']= nombreDeVictoires.First_pokemon/(nbFoisPremierePosition.Winner + nbFoisSecondePosition.Winner)
#création d'un nouveau pokedex contenant les nouvelles colones ci-dessus
nouveauPokedex = nosPokemons.merge(listeAAgreger,left_on='NUMERO', right_index = True, how ='left')


dataset = nouveauPokedex
dataset.to_csv("datas/dataset.csv", sep='\t')
dataset = pnd.read_csv("datas/dataset.csv",delimiter='\t')
dataset = dataset.dropna(axis=0, how='any')
X = dataset.iloc[:, 5:12].values
y = dataset.iloc[:, 17].values
from sklearn.model_selection import train_test_split
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION =train_test_split(X, y, test_size = 0.2, random_state = 0)

# apprentissage de l'IA grace à une random forest
algorithme = RandomForestRegressor()

algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

predictions = algorithme.predict(X_VALIDATION)

precision = r2_score(Y_VALIDATION, predictions)

# sauvegarde de l'intelligece artificielle
from sklearn.externals import joblib
fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)


