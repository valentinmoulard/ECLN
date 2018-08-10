import sys
import pickle
import os
import csv
import json
import time
import re
import pandas as pd
import numpy as np
from random import randint
from collections import defaultdict

start_time = time.time()

# chargement des données
url = "C:/Users/vmoulard/Desktop/ECLN/niveau_communeV3.csv"
url2 = "C:/Users/vmoulard/Desktop/ECLN/niveau_epciV3.csv"
data_commune = pd.read_csv(url, low_memory = False)
data_epci = pd.read_csv(url2, low_memory = False)

##############################################################################
## remodélisation du jeu de données pour faciliter la manipulation ###########

data_commune = data_commune[[i for i in list(data_commune.columns) if i != 'EPCI 2014' and i != 'Unnamed: 152' and i != 'somme']]
data_epci = data_epci[[i for i in list(data_epci.columns) if i != 'Commentaires Invest- Fiche SNI' and i != 'Unnamed: 170']]

# reçoit x et le nettoie. fonction utilisé pour renommer les colonnes
def clean_title (x):
    expression1 = r"[0-9]{1}$"
    expression2 = r"[0-9]{2}$"
    expression3 = r"^[0-9]{4}"
    if re.search(expression2, x):
        return (x[:-2] + str(int(x[-2:]) + 2005))
    elif re.search(expression1, x):
    	return (x[:-1] + str(int(x[-1:]) + 2005))
    elif re.search(expression3, x):
    	tmp = x[:4]
    	return (x[5:] + '.' + tmp)
    else:
    	return (x + '.' + str(2005))

# fonction qui renomme les colonnes
def clean_columns_names():
	i = 7
	while i < len(data_commune.columns):
		data_commune.columns.values[i] = clean_title(data_commune.columns[i])
		i+=1
	i = 7
	while i < len(data_epci.columns):
		data_epci.columns.values[i] = clean_title(data_epci.columns[i])
		i+=1

# renommage des colonnes
clean_columns_names()

# le fichier etant complet, on peut mettre des '0' dans les cellules vides
data_epci = data_epci.fillna('0')

# on selectionne uniquement les données à traiter (dont la colonne 'Commentaire est égale à 0') dans 'data_epci'
# on stock les données qu'on ne va pas traiter car on les renvoie avec le resultat final apres les traitements
# data_valide contient le set avec les lignes à traiter, data_reste contient les lignes à ne pas traiter
# 2 façons de faire la meme chose
data_epci_valide = data_epci[data_epci.Commentaires == '0']

# on recupere la liste des epci à traiter pour savoir quelles sont les données à traiter dans data_commune
liste_epci = data_epci_valide.SIREN

# on selectionne les lignes à traiter et on les stock dans 'data_commune_valide' et on les stock dans 'data_commune_valide' 
# on stock le reste de la meme maniere
data_commune_valide = data_commune[data_commune['siren2017'].isin(liste_epci)]
data_commune_reste = data_commune[~data_commune['siren2017'].isin(liste_epci)]

# on remplace tous les '//' présent dans le jeu de données et on les remplace par 0
data_commune_valide = data_commune_valide.replace('//',0)



###########################################################################
## TRAITEMENT #############################################################

data_commune_valide = data_commune_valide.replace('nd',np.NaN)
data_epci_valide = data_epci_valide.replace('nd',np.NaN)
# on convertit toutes les données en float
# je selectionne les données à partir de la 8ème colonne
# 
data_commune_valide.iloc[:,7:].astype('float')
# 'data_commune_zero' contient les lignes dont la somme fait 0, i.e. aucune activité sur la commune => on peut mettre des 0 sur ces lignes
data_commune_zero = data_commune_valide[data_commune_valide.sum(axis=1) == 0]
data_commune_zero = data_commune_zero.fillna('0')

# on met dans data_commune_valide, les lignes qu'il faut traiter.
data_commune_valide = data_commune_valide[data_commune_valide.sum(axis=1) != 0]

# on crée un index numérique pour data_commune_valide
# les 2 lignes suivantes font la meme chose (code mémoire)
data_commune_valide.index = pd.RangeIndex(len(data_commune_valide.index))
data_commune_valide.index = range(len(data_commune_valide.index))


#  fonction qui renvoie False si une ligne possede une valeur nulle (NaN)
def ligne_complete(x):
    # s'il y a une valeur vide ou plus dans la ligne ou si la valeur est 'nd', renvoie False
    for i in range(len(x.columns)):
        if x.iat[0,i] == 'nd':
            return False
        elif x.isna().any().any() == True:
            return False
        elif x.iat[0,i] == np.NaN:
            return False
    return True


# fonction qui estime l'encours à l'année (n-1)
def estim_encours_n_moins_1(row, col):
    ligne = data_commune_valide.iloc[[row] , [col+8, col+9, col+10, col+11, col+12]]
    # print(ligne)
    encours_n = int(ligne.iat[0,4])
    mise_en_vente_n = int(ligne.iat[0,0])
    reservation_n = int(ligne.iat[0,1])
    annulation_n = int((ligne.iat[0,2]))
    changement_dest_n = int(ligne.iat[0,3])
    estim = encours_n - mise_en_vente_n + reservation_n - annulation_n - changement_dest_n
    return estim


# dans un dataframe, si pour une commune donnée, il ne manque qu'une seule ligne apres avoir fait un groupement par commune, il est possible d'estimer cette ligne
# fonction qui cherche dans un dataframe regroupant les communes, l'indice d'une ligne a estimer si cette ligne est la seule du dataframe à etre incomplete
def estim_derniere_ligne(gb):
    tmp = True
    compte = 0
    for row in gb.index:
        if ligne_complete(gb.loc[[row]]) == False:
            ligne = row
            tmp = False
            compte += 1
    if tmp == False and compte == 1:
        # num_siren = groupe.at[ligne,'siren2017']
        return ligne
    else:
        return -1


# on récupère la ligne qu niveau epci pour pouvoir déterminer la ligne commune dans niveau_commune si c'est la seule à estimer
def ligne_epci(compteur, name):
    compteur_epci = compteur
    for col_epci in data_epci_valide:
        if "MEV" in str(col_epci):
            compteur_epci -= 1
            if compteur_epci == 0:
                # total_siren est la somme déja fournie par code_siren dans le fichier 'niveau_epci'
                total_siren = data_epci_valide.loc[data_epci_valide['SIREN'] == name]
                loc_epci = data_epci_valide.columns.get_loc(str(col_epci))
                # print(loc_epci)
                total_siren = total_siren.iloc[:,loc_epci:loc_epci+5]
                # print(total_siren)
                return (total_siren)


# (a vérifier) ( 2-3 ) fonction estimation regle de trois
def regle_de_trois(compteur, name, gb):
    gb.fillna(0, inplace = True)
    gb = gb.replace('nd', 0)
    gb = gb.astype('int')

    # on recupere la ligne au niveau epci
    total_siren = ligne_epci(compteur, name)
    total_siren = total_siren.astype('int')
    print("= total_siren =")
    print(total_siren)
    # on calcul la somme des MEV pour ce sous groupe
    somme_MEV = gb.iloc[:,0].sum()

    # si la somme des MEV est différente de 0 alors on crée une colonne 'proportion'
    if somme_MEV != 0:
        # le compteur sert à itérer sur le dataframe 'total_siren'
        compteur = 0
        tab_proportion = gb.iloc[:,0].apply(lambda x : x/somme_MEV)
        # pour chacune des colonnes du sous groupe, on multiplie chaque valeurs par les valeur de la colonne 'proportion'
        # sauf pour les colonnes MEV et Encours
        for col in gb:
            if not "MEV" in str(col):
                if not "cours" in str(col):
                    # multiplication/regle de trois
                    gb[str(col)] = tab_proportion * total_siren.iloc[0][compteur]
                    # on arrondi les resultats à l'entier le plus proche
                    gb[str(col)] = gb[str(col)].round()
            compteur +=1
        time.sleep(10)
    return (gb)


# ( 0 ) dans les colonnes "MEV" (mise en vente, non soumis au secret statistique), s'il y a un vide, on met des 0
for col in data_commune_valide:
    if "MEV" in str(col):
        loc = data_commune_valide.columns.get_loc(str(col))
        data_commune_valide[col].fillna(value = 0, inplace = True)

####### partie calcul de l'encours (n-1)
# ( 1 ) parcours de toute les cellules pour verifier si la valeur encours est complétée ou non
for row in data_commune_valide.index:
    for col in data_commune_valide:
        # si on est dans une colonne 'encours'
        if "Encours" in str(col):
            # si la cellule est vide il faut l'estimer
            if pd.isnull(data_commune_valide.at[row, col]) == True :
                # on récupere l'indice de la colonne qui nous servira dans les boucles suivantes
                loc = data_commune_valide.columns.get_loc(str(col))
                # condition pour ne pas dépasser la taille du tableau
                if loc < (len(data_commune_valide.columns)-12):
                    # vérifie si la ligne contenant les valeurs qui permettent de calculer l'encours (n-1) sont completes
                    if ligne_complete(data_commune_valide.iloc[[row] , [loc+8, loc+9, loc+10, loc+11, loc+12]]) == True :
                        # ESTIMATION
                        data_commune_valide.at[row, col] = estim_encours_n_moins_1(row, loc)
                        # print(data_commune_valide.at[row, col])


gb = data_commune_valide.groupby(['siren2017'])
liste = data_commune_valide.siren2017.unique()

# ( 2 ) Pour un code siren donné, si au niveau EPCI il y a un 0 dans une colonne, on reporte ce 0 au niveau commune pour le meme code siren dans la colonne correspondante (qu'il y ai qu'une ou plisieurs lignes au niveau commune)
for name in liste:
    groupe = gb.get_group(name)
    compteur = 0
    for col in data_commune_valide:
        if "MEV" in str(col):
            compteur += 1
            total_siren = ligne_epci(compteur, name)
            total_siren = total_siren.replace('nd', np.NaN)
            loc = data_commune_valide.columns.get_loc(str(col))
            sous_groupe = groupe.iloc[:,loc:loc+5]
            for iterator in total_siren:
                if float(total_siren.iloc[0][str(iterator)]) == 0:
                    loc2 = total_siren.columns.get_loc(str(iterator))
                    sous_groupe.iloc[:,loc2] = 0
                    # METTRE A JOUR #
                    # data_commune_valide.update(sous_groupe)



############## partie calcul de la derniere ligne
# pour chaque code siren unique présente dans niveau_commune
for name in liste:
    # on constitue des groupes par code siren
    groupe = gb.get_group(name)
    # compteur qui servira à chercher la ligne qu'il nous faut dans niveau_epci
    compteur = 0
    # on va constituer des sous groupe par année et par forme urbaine (individuel ou collectif)
    for col in data_commune_valide:
        if "MEV" in str(col):
            compteur += 1
            # on récupere l'indice de la colonne pour la fonction 'iloc' qui va suivre
            loc = data_commune_valide.columns.get_loc(str(col))
            # on constitue le sous groupe
            sous_groupe = groupe.iloc[:,loc:loc+5]

            # si le sous groupe contient une seule ligne avec des données à estimer, on récupère son indice en ligne et le code siren
            ligne = estim_derniere_ligne(sous_groupe)
            num_siren = name
            if ligne > -1 :
                # somme_siren fait la somme en colonne de toutes les valeurs sans prendre en compte la ligne à estimer
                somme_siren = sous_groupe.drop([int(ligne)]).sum()
                # total_siren est la somme de toutes les communes
                total_siren = ligne_epci(compteur, name)
                total_siren = total_siren.astype('float')

                # on fait la soustraction entre total_siren et somme_siren pour obtenir la ligne à estimer
                try:
                    sous_groupe.loc[[ligne]] = total_siren.sub (somme_siren)
                except:
                    pass
                sous_groupe.loc[[ligne]] = sous_groupe.loc[[ligne]].fillna(0)

                # on actualise l'opération qu'on a réaliser dans le groupe dans le dataframe original
                # groupe.update(sous_groupe)

print("==================OK==================")
print("--- %s seconds ---" % (time.time() - start_time))
time.sleep(1000)



############# REGLE DE TROIS
for name in liste:
    # on constitue des groupes par code siren
    groupe = gb.get_group(name)
    # compteur qui servira à chercher la ligne qu'il nous faut dans niveau_epci
    compteur = 0
    # on va constituer des sous groupe par année et par forme urbaine (individuel ou collectif)
    for col in data_commune_valide:
        if "MEV" in str(col):
            compteur += 1
            # on ne traite pas les données de l'année 2005
            if compteur > 2:
                # on récupere l'indice de la colonne pour la fonction 'iloc' qui va suivre
                loc = data_commune_valide.columns.get_loc(str(col))
                # on constitue le sous groupe
                sous_groupe = groupe.iloc[:,loc:loc+5]
                # print(name)
                print("\n \n=== AVANT ===")
                print(name)
                print(sous_groupe)
                sous_groupe = regle_de_trois(compteur, name, sous_groupe)
                print("\n \n=== APRES ===")
                print(sous_groupe)
                # METTRE A JOUR #
                data_commune_valide.update(sous_groupe)

print("ok")