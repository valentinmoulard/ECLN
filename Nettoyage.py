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

# sert à calculer le temps d'execution
start_time = time.time()

# chargement des données
url = "C:/Users/vmoulard/Desktop/ECLN/niveau_communeV3.csv"
url2 = "C:/Users/vmoulard/Desktop/ECLN/niveau_epciV3.csv"
data_commune = pd.read_csv(url, low_memory = False)
data_epci = pd.read_csv(url2, low_memory = False)


# Remodélisation du jeu de données pour faciliter la manipulation
data_commune = data_commune[[i for i in list(data_commune.columns) if i != 'EPCI 2014' and i != 'Unnamed: 152' and i != 'somme']]
data_epci = data_epci[[i for i in list(data_epci.columns) if i != 'Commentaires Invest- Fiche SNI' and i != 'Unnamed: 170']]



## FONCTIONS DE NETTOYAGE

# reçoit x et met en forme le label de la colonne
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

# le fichier au niveau des epci étant complet, on peut mettre des '0' dans les cellules vides
data_epci = data_epci.fillna('0')

# on selectionne uniquement les données à traiter (on selectionne les données où la colonne 'Commentaire' est égale à 0) dans 'data_epci'
data_epci_valide = data_epci[data_epci.Commentaires == '0']

# on recupere la liste des epci à traiter pour savoir quelles sont les données à traiter dans data_commune
liste_epci = data_epci_valide.SIREN

# on selectionne les lignes à traiter et on les stock dans 'data_commune_valide' 
# on stock le reste de la meme maniere dans data_commune_reste
# apres les estimations, on concatenera les 2 dataframes
data_commune_valide = data_commune[data_commune['siren2017'].isin(liste_epci)]
data_commune_reste = data_commune[~data_commune['siren2017'].isin(liste_epci)]

# on remplace tous les '//' présent dans le jeu de données et on les remplace par 0
data_commune_valide = data_commune_valide.replace('//',0)




## TRAITEMENT

# on remplacer les 'nd' par des cellules vides, ce sont des données à estimer
data_commune_valide = data_commune_valide.replace('nd',np.NaN)
data_epci_valide = data_epci_valide.replace('nd',np.NaN)
# on convertit les données en float pour qu'elles soient adaptées aux fonctions utilisées plus bas
data_commune_valide.iloc[:,7:].astype('float')


# 'data_commune_zero' contient les lignes dont la somme fait 0, i.e. aucune activité sur la commune => on peut mettre des 0 sur ces lignes
# apres les estimations, on concatenera les 2 dataframes
data_commune_zero = data_commune_valide[data_commune_valide.sum(axis=1) == 0]
data_commune_zero = data_commune_zero.fillna('0')
# on met dans data_commune_valide, les lignes qu'il faut traiter.
data_commune_valide = data_commune_valide[data_commune_valide.sum(axis=1) != 0]


# on crée un index numérique pour data_commune_valide
# les 2 lignes suivantes font la meme chose (code mémoire)
data_commune_valide.index = pd.RangeIndex(len(data_commune_valide.index))
data_commune_valide.index = range(len(data_commune_valide.index))



## FONCTION DE TRAITEMENT DES DONNEES

# fonction qui renvoie False si une ligne possede une valeur nulle (NaN)
def ligne_complete(x):
    # retourne True si la ligne est complete
    # s'il y a une valeur vide ou plus dans la ligne ou si la valeur est 'nd' ou si la valeur est NaN, on renvoie False
    for i in range(len(x.columns)):
        if x.iat[0,i] == 'nd':
            return False
        elif x.isna().any().any() == True:
            return False
        elif x.iat[0,i] == np.NaN:
            return False
    return True


# fonction qui renvoie l'encours calculé à l'année (n-1)
def estim_encours_n_moins_1(row, col):
    # on récupere les indice numérique en ligne et en colonne de la cellule 'encours' à calculer
    # ligne contient les données nécéssaire qu calcul de l'encours à l'année n-1
    ligne = data_commune_valide.iloc[[row] , [col+8, col+9, col+10, col+11, col+12]]
    encours_n = int(ligne.iat[0,4])
    mise_en_vente_n = int(ligne.iat[0,0])
    reservation_n = int(ligne.iat[0,1])
    annulation_n = int((ligne.iat[0,2]))
    changement_dest_n = int(ligne.iat[0,3])
    # calcul de l'encours n-1 avec la formule suivante : encours(n-1) = encours(n) - mise en vente(n) + réservations(n) - annulation(n) - changement de destiantion(n)
    estim = encours_n - mise_en_vente_n + reservation_n - annulation_n - changement_dest_n
    return estim


# fonction qui renvoie l'encours calculé à l'année (n)
def estim_encours_n(row, col):
    # on récupere les indice numérique en ligne et en colonne de la cellule 'encours' à calculer
    # ligne contient les données nécéssaire qu calcul de l'encours à l'année n
    ligne = data_commune_valide.iloc[[row] , [col-12, col-4, col-3, col-2, col-1]]
    encours_n_moins_1 = int(ligne.iat[0,0])
    mise_en_vente_n = int(ligne.iat[0,1])
    reservation_n = int(ligne.iat[0,2])
    annulation_n = int((ligne.iat[0,3]))
    changement_dest_n = int(ligne.iat[0,4])
    estim = encours_n_moins_1 + mise_en_vente_n - reservation_n + annulation_n + changement_dest_n
    return estim


# si pour une commune donnée, dans un dataframe groupé par code siren, il n'y a qu'une seule ligne qui a des données manquantes, alors on peut la calculer
# dans un dataframe regroupant les communes par code siren, cette fonction cherche l'indice d'une ligne à calculer.
# si cette ligne est la seule du dataframe à etre incomplete
def estim_derniere_ligne(gb):
    tmp = True
    # compte sert à compter le nombre de ligne dont les valeurs sont incompletes
    compte = 0
    # on itère sur toute les lignes du dataframe 'gb'
    for row in range(len(gb.index)):
        if ligne_complete(gb.iloc[[row]]) == False:
            ligne = row
            tmp = False
            compte += 1
    # si tmp = False (dataframe avec des données manquantes) et si compte = 1 (le nombre de ligne avec des données manquante est égale à 1)
    # alors on renvoie l'indice de cette ligne pour la calculer
    if tmp == False and compte == 1:
        return ligne
    else:
        return -1


# fonction qui renvoie la ligne epci correspondant à un groupe de commune par code siren
def ligne_epci(compteur, name):
    # name est le code siren des communes dont on cherche la ligne epci correspondante
    # compteur et compteur_epci servent à sélectionner la bonne année
    compteur_epci = compteur
    for col_epci in data_epci_valide:
        if "MEV" in str(col_epci):
            compteur_epci -= 1
            if compteur_epci == 0:
                # on sélectionne la ligne epci avec le code siren correspondant
                total_siren = data_epci_valide.loc[data_epci_valide['SIREN'] == name]
                loc_epci = data_epci_valide.columns.get_loc(str(col_epci))
                total_siren = total_siren.iloc[:,loc_epci:loc_epci+5]
                return (total_siren)


# fonction estimation regle de trois
def regle_de_trois(compteur, name, gb):
    # gb est un sous groupe de commune ayant le meme code siren
    gb = gb.replace('nd', np.NaN)
    gb = gb.astype('float')

    # on recupere la ligne au niveau epci
    total_siren = ligne_epci(compteur, name)
    total_siren = total_siren.astype('float')

    # on calcul la somme des MEV pour ce sous groupe
    somme_MEV = gb.iloc[:,0].sum()

    # si la somme des MEV est différente de 0 alors on crée une colonne 'proportion'
    if somme_MEV != 0:
        # le compteur sert à itérer sur le dataframe 'total_siren'
        compteur = 0
        tab_proportion = gb.iloc[:,0].apply(lambda x : x/somme_MEV)
        # pour chacune des colonnes du sous groupe, on multiplie chaque valeurs par les valeur de la colonne 'proportion'
        # sauf pour les colonnes MEV et Encours (les MEV ne sont pas soumises au secret statistique et l'Ecnours est à calculer à partir de la formule)
        #  on itère sur les colonnes de gb
        for col in gb:
            #  on réalise les estimations pour toutes les colonnes sauf les MEV et Encours
            if not "MEV" in str(col):
                if not "cours" in str(col):
                    # multiplication/regle de trois
                    gb[str(col)] = tab_proportion * total_siren.iloc[0][compteur]
                    # on arrondi les resultats à l'entier le plus proche
                    gb[str(col)] = gb[str(col)].round()
            compteur +=1
    return (gb)



## DEBUT DES TRAITEMENTS

# Dans les colonnes "MEV" (mise en vente, non soumis au secret statistique), s'il y a un vide, on met des 0
print('0 si vide dans colonne MEV')
for col in data_commune_valide:
    if "MEV" in str(col):
        loc = data_commune_valide.columns.get_loc(str(col))
        data_commune_valide[col].fillna(value = 0, inplace = True)
print("--- %s seconds ---" % (time.time() - start_time))


# Calcul de l'encours à l'année n-1 si les données de l'année n ne sont pas manquantes
print('\ncalcul encours n-1')
# on itère sur toutes les lignes et les colonnes du dataframe
for row in data_commune_valide.index:
    for col in data_commune_valide:
        # si on est dans une colonne 'encours'
        if "Encours" in str(col):
            # si la cellule est vide, il faut la calculer
            if pd.isnull(data_commune_valide.at[row, col]) == True :
                # on récupere l'indice de la colonne qui nous servira dans les boucles suivantes
                loc = data_commune_valide.columns.get_loc(str(col))
                # condition pour ne pas dépasser la taille du dataframe en colonne
                if loc < (len(data_commune_valide.columns)-12):
                    # vérifie si la ligne contenant les valeurs qui permettent de calculer l'encours (n-1) sont completes
                    if ligne_complete(data_commune_valide.iloc[[row] , [loc+8, loc+9, loc+10, loc+11, loc+12]]) == True :
                        # calcul de l'encours n-1
                        data_commune_valide.at[row, col] = estim_encours_n_moins_1(row, loc)
print("--- %s seconds ---" % (time.time() - start_time))


# group_by permet de grouper les commune par code siren
gb = data_commune_valide.groupby(['siren2017'])
liste = data_commune_valide.siren2017.unique()


print('\n0 en colonne des commune si 0 au niveau epci')
# Pour un code siren donné, si au niveau EPCI il y a un 0 dans une colonne, on reporte ce 0 au niveau commune dans toute la colonne correspondante
for name in liste:
    # on constitue un groupe qui est un sub dataframe de 'data_commune_valide' qui ne contient que les commune ayant un meme code siren pour les année allant de 2005 à 2017
    groupe = gb.get_group(name)
    # compteur servira à récuperer la ligne epci à l'année correspondante
    compteur = 0
    for col in data_commune_valide:
        if "MEV" in str(col):
            compteur += 1
            total_siren = ligne_epci(compteur, name)
            total_siren = total_siren.replace('nd', np.NaN)
            # 'loc' est l'index numérique en colonne de 'col' pour pouvoir constituer le sous groupe
            loc = data_commune_valide.columns.get_loc(str(col))
            # 'sous groupe' est un sous dataframe de 'groupe' concernant une année et une forme urbaine (collectif ou individuel)
            sous_groupe = groupe.iloc[:,loc:loc+5]
            for iterator in total_siren:
                if float(total_siren.iloc[0][str(iterator)]) == 0:
                    sous_groupe.iloc[:,sous_groupe.columns.get_loc(iterator)] = 0
                    data_commune_valide.update(sous_groupe)
print("--- %s seconds ---" % (time.time() - start_time))



print('\ncalcul de la derniere ligne')
# Calcul de la derniere ligne si dans un regroupement de communes au niveau par code siren, une seule ligne est manquante
# pour chaque code siren unique présente dans niveau_commune
for name in liste:
    # on constitue des groupes par code siren
    groupe = gb.get_group(name)
    # compteur qui servira à chercher la ligne qu'il nous faut dans niveau_epci
    compteur = 0
    for col in data_commune_valide:
        if "MEV" in str(col):
            compteur += 1
            # 'loc' est l'index numérique en colonne de 'col' pour pouvoir constituer le sous groupe
            loc = data_commune_valide.columns.get_loc(str(col))

            # on constitue le sous groupe par année et par forme urbaine (individuel ou collectif)
            sous_groupe = groupe.iloc[:,loc:loc+5]
            sous_groupe = sous_groupe.astype('float')

            # si le sous groupe contient une seule ligne avec des données à estimer, on récupère son indice en ligne et le code siren
            ligne = estim_derniere_ligne(sous_groupe)
            if ligne > -1 :
                # somme_siren fait la somme en colonne de toutes les valeurs sans prendre en compte la ligne à estimer
                somme_siren = sous_groupe.sum()
                # total_siren est la somme de toutes les communes (tout simplement la ligne correspondante au niveau epci)
                total_siren = ligne_epci(compteur, name)
                total_siren = total_siren.astype('float')

                # on fait la soustraction entre total_siren et somme_siren pour obtenir la ligne à estimer
                result = total_siren.sub(somme_siren)

                # pour mettre le résultat dans le sous groupe car la fonction
                # sous_groupe.iloc[[ligne]] = total_siren.sub(somme_siren) ne fonctionne pas
                for col_name in total_siren:
                    loc3 = total_siren.columns.get_loc(str(col_name))
                    sous_groupe.iat[ligne,loc3] = result.iat[0,loc3]

                # on actualise l'opération qu'on a réaliser dans le groupe dans le dataframe original
                data_commune_valide.update(sous_groupe)
print("--- %s seconds ---" % (time.time() - start_time))



print('\nregle de trois')
# estimation par regle de trois
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
                sous_groupe = regle_de_trois(compteur, name, sous_groupe)
                data_commune_valide.update(sous_groupe)
print("--- %s seconds ---" % (time.time() - start_time))




#### derniere fonction en développement : calcul de l'encours apres les estimations de la regle de trois


# fonction qui retourne un compteur qui permettra de trouver la colonne Encours à l'année correspondante au niveau commune
# retourne -1 s'il n'y a pas de colonne Encours avec du volume au niveau epci
# prend comme parametre l'index en ligne et le bom du code siren
def cherche_encours_epci(row, name):
    compteur = 0
    for col in data_epci_valide:
        if "Encours" in str(col):
            compteur += 1
            loc = data_epci_valide.columns.get_loc(str(col))
            if data_epci_valide.iloc[[row],[loc]] != 'nd' and data_epci_valide.iloc[[row],[loc]] != np.NaN:
                return compteur
    return -1


# fonction qui va vérifier s'il y a deja un encours sur la ligne/commune et si oui renvoie l'indice en colonne de l'encours
# si la fonction ne trouve pas d'encours ayant un volume alors elle renvoie -1
def cherche_encours_commune(row, df):
    for col in df:
        if "Encours" in str(col):
            loc = df.columns.get_loc(str(col))
            if df.iloc[[row],[loc]] != np.NaN and df.iloc[[row],[col]] != 'nd':
                return col
    return -1


# fonction qui calcul les encours des années précédentes et suivantes en fonction de l'indice en colonne de l'encours trouvé
def calcul_encours_commune(row, col):
    for col in data_commune_valide:
        if "Encours" in str(col):
            loc = df.columns.get_loc(str(col))
            if loc < col:
                # calcul de l'encours année précédente

            elif loc > col:
                # calcul de l'encours année suivante
                


print("ok")
