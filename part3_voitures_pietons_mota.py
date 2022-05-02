# Arthur Kaboré (8530422) et Kashalala David Tshiswaka (8719666)
# CSI4533
# Partie 3 du projet

import os
import random
import cv2
import gc
import functions as f
from nn_detector import NN_detector as detector
import warnings
warnings.filterwarnings("ignore")

print("Arthur Kaboré (8530422) et Kashalala David Tshiswaka (8719666) \nCSI4533 \nPartie 3 du projet \n")

########## Extraction des données ##########################################################################

ground_truth = [] #initialisation de la variable qui contientdra toutes les vrais boites des objets demandés

with open("C:/Users/Clement/Desktop/part2_voitures/Ground_truth_with_tracking_cleaned.txt",'r') as file: # Lecture du fichier gt.txt
    for line in file: # Lecture de chaque lignes
        b = [] # tableau qui representera une boite
        for word in line.split(","): # leture de chaque mot
            b.append(float(word))
        # Conservation des boites des voitures
        if (b[7] == 3 or b[7] == 1):
            ground_truth.append(b)

########## Programme principal (implémentation des instructions recus) ###################################################################################################################

# fixer un seuil pour le score
score_threshold = 0.4

# instancier la classe avec la valeur souhaitée pour "GPU_detect"
det = detector(GPU_detect = False)# fixer un seuil pour le score

t = 1 # indiquera a quelle image nous somme
prev_colors = [] # contiendra les couleurs precedentes
path = "C:/Users/Clement/Desktop/part2_voitures/img1"
img1 = os.scandir(path) #accede au dossier
images = [] # tableau de chemin pour chaque image dans path
FNs = [] # nombre d'objets non-détectés (IoU < 0.4)
FPs = [] # nombre de détection ne correspondant pas a un obje
IDS = [] # nombre de IDs différents associés a un meme objet
GTs = [] # nombre d'objets dans le fichier GT a l'image t
for image in img1: # met le chemin de chaque image dans le tableau images
    images.append(path + '/' + image.name)
# nombre d'objet dans chaque image
y=1
while y <= len(images):
    total = 0
    for u in ground_truth:
        if u[0] == y:
            total+=1
        #print(total)
    GTs.append(total)
    y+=1

ft = [] # contient les boites a l'image t
ft1 = [] # contient les boites a l'image t+1
ft_verf = [] # pour verifier si l'objet detecte existe dans ground_truth
# Debut du programme
file = open("mygt.txt", "w")
while t <= len(images):
    ious = [] # contient les IoU
    ids = 0 # compte le nombre de IDs différents
    if t == 1: # Dessin des boites de la premiere Image
        detections = det.detect(images[t-1]) # exécuter l'inférence sur l'image
        image = cv2.imread(images[t-1])
        window_name = 'Image'
        z=0
        # garder les objets de l'image 1 de ground_truth
        for x in ground_truth:
            if x[0] == t:
                ft_verf.append(x)
        # détection des objets
        for i in range(0, len(detections["boxes"])):
            box = detections["boxes"][i]
            score = detections["scores"][i]
            #id = int(detections["labels"][i])
            (x1, y1, x2, y2) = box.astype("int")
            # (x1-x2) est la largeur et (y1-y2) est la longueur
            if (score > score_threshold) and ((abs(x1-x2) >= abs(y1-y2)) or ((-10 < abs(x1-x2) - abs(y1-y2) < 0))):
                line = [t,0,x1,y1,abs(x1-x2),abs(y1-y2),1]
                ft.append(line)
            if (score > score_threshold) and not(((x1-x2) >= (y1-y2)) or ((-10 < (x1-x2) - (y1-y2) < 0))):
                line = [t,0,x1,y1,abs(x1-x2),abs(y1-y2),3]
                ft.append(line)
        for u in ft: # pour chaque boite dans L'image t
            # point de depart
            # represente le coin en haut a gauche du rectangle
            start_point = (int(u[2]), int(u[3]))
            # point d'arrive
            # represente le coin en bas a droite du rectangle
            end_point = (int(u[2] + u[4]), int(u[3] + u[5]))
            # couleur du rectangle
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            # Enregistre les couleurs comme couleurs precedentes.
            # l'ordre dans lequel chaque couleur est enregistrée représente son index dans ft.
            prev_colors.append([r,g,b])
            color = (r,g,b)
            #donne la valeur du rgb comme idea
            u[1] = r+g+b
            # Dessin du rectangle
            image = cv2.rectangle(image, start_point, end_point, color, 2)

        f.verification(ft, ft_verf, FNs, FPs, t)
        #pour compter les mauvais ids
        for x in ft_verf:
            for y in ft:
                if (x[1]!=-1 and y[1]!=-1) and (x[0]==t and x[1] == y[1]):
                    ids+=1
            if ids>1:
                ids-=1
        IDS.append(ids)
        # écrire les objets dans mygt.txt
        for u in ft:
            if u[1]!=-1:
                if z!=0:
                    file.write('\n')
                z+=1
                #file.write('\n')
                file.write(str(u[0]))
                file.write(',')
                file.write(str(u[1]))
                file.write(',')
                file.write(str(u[2]))
                file.write(',')
                file.write(str(u[3]))
                file.write(',')
                file.write(str(u[4]))
                file.write(',')
                file.write(str(u[5]))
                file.write(',')
                file.write(str(u[6]))
        t+=1

    else:
        ft = ft1
        ft1 = []
    ft_verf = []
    # garder les objets de l'image t de ground_truth
    for x in ground_truth:
        if x[0] == t:
            ft_verf.append(x)
    # assigner les boites de l'image t+1 à ft1
    detections = det.detect(images[t-1]) # exécuter l'inférence sur l'image
    for i in range(0, len(detections["boxes"])):
        score = detections["scores"][i]
        #id = int(detections["labels"][i])
        box = detections["boxes"][i]
        (x1, y1, x2, y2) = box.astype("int")
        # (x1-x2) est la largeur et (y1-y2) est la longueur
        if (score > score_threshold) and (((x1-x2) <= (y1-y2)) or ((0 < (x1-x2) - (y1-y2) < 10))):
            line = [t,0,x1,y1,abs(x1-x2),abs(y1-y2),1]
            ft1.append(line)
        if (score > score_threshold) and not((abs(x1-x2) >= abs(y1-y2)) or ((-10 < abs(x1-x2) - abs(y1-y2) < 0))):
            line = [t,0,x1,y1,abs(x1-x2),abs(y1-y2),3]
            ft1.append(line)

    f.verification(ft1, ft_verf, FNs, FPs, t)
    #f.equilibre(ft,ft1,t) # equilibre les deux matrices
    # Pour chaque entrée de la matrice, calcule le IoU entre la boite i et la boite j
    for u in ft:
        if u: #verifie si vide
            b = [] # représente les lignes de la matrice ious
            for v in ft1:
                if v: #verifie si vide
                    i = f.iou(u,v)
                    if i < 0.4: # Met à 0 toutes les valeurs IoU inférieur à un seuil de 0.4
                        b.append(0)
                    else:
                        b.append(i)

            ious.append(b)

    # Recherche l'entrée i,j >0 ayant la valeur maximum pour chaque i,j
    w = 0
    j = 0
    if ious: #verifie si vide
        while j < len(ft1): # tant que j < au nombre de colonne de ious
            i = 0
            w = ious[i][j] # initialisation de la variable qui representera le maximum
            while i < len(ft): # tant que i < au nombre de ligne de ious
                if ious[i] and (ious[i][j] > 0) and (ious[i][j] > w): # si ious[i] non vide et...
                    w = ious[i][j]
                i+=1

            # Remplace la valeur maximum i,j par -1
            # Toutes les autres valeurs de la colonne sont mises à 0.
            if w > 0:
                i=0
                d=0 # popur assigner un id une fois
                while i < len(ft):
                    if ious[i] and ious[i][j] == w and ft1[j][1]!=-1:
                        ious[i][j] = -1
                        ft1[j][1] = ft[i][1]
                    else:
                        ious[i][j] = 0
                    i+=1
            j+=1

    #print(ft)
    #print(ft1)
    #print(ious)
    #print(FPs)
    #print(FNs)
    #print(GTs)
    # Lecture de l'image t
    image = cv2.imread(images[t-1])
    # nom de la fenetre de l'Image
    window_name = 'Image'
    j=0
    col = [] # enregistre les couleurs pour la mise a jour des couleurs precedentes (prev_colors)
    while j < len(ft1):
        i=0
        color = (0,0,0)
        start_point = (0,0)
        end_point = (0,0)
        z = -1 # nous aidera a ne pas assigner plusieurs couleurs a certaines boitesS
        while i < len(ft):
            # Chaque objet F(t+1) associé à un objet se voit attribuer la couleur de cet objet precedent
            if ious[i][j] == -1 and ft[i][1]!=-1 and ft1[j][1]!=-1 :
                z+=1
                # point de depart
                # represente le coin en haut a gauche du rectangle
                start_point = (int(ft1[j][2]), int(ft1[j][3]))
                # point d'arrive
                # represente le coin en bas a droite du rectangle
                end_point = (int(ft1[j][2] + ft1[j][4]), int(ft1[j][3] + ft1[j][5]))
                # couleur du rectangle
                # Etant donné que les couleurs de chaque boites dans ft ont été enregistrées dans prev_colors avec les meme indexs que dans ft,
                # on peut utiliser i pour designer la couleur précédente correspondant a la boite avec laquelle la boite ft+1[j] a un iou = -1
                r = prev_colors[i][0]
                g = prev_colors[i][1]
                b = prev_colors[i][2]
                color = (r,g,b)
                col.append([r,g,b])
                if ft1[j][1] == 0:
                    ft1[j][1] = r+g+b
                file.write('\n')
                file.write(str(ft1[j][0]))
                file.write(',')
                file.write(str(ft1[j][1]))
                file.write(',')
                file.write(str(ft1[j][2]))
                file.write(',')
                file.write(str(ft1[j][3]))
                file.write(',')
                file.write(str(ft1[j][4]))
                file.write(',')
                file.write(str(ft1[j][5]))
                file.write(',')
                file.write(str(ft1[j][6]))
                # Dessin du rectangle
                image = cv2.rectangle(image, start_point, end_point, color, 2)
                break

            # Chaque objet créant une nouvelle trajectoire se voit associer une nouvelle couleur rgb aleatoire
            else:
                # si la boite (ft1[j]) a un iou == -1 cela signifie qu'elle a la couleur de la boite précedente (ft[i]), donc on break
                if z != -1:
                    break
                # point de depart
                # represente le coin en haut a gauche du rectangle
                start_point = (int(ft1[j][2]), int(ft1[j][3]))
                # point d'arrive
                # represente le coin en bas a droite du rectangle
                end_point = (int(ft1[j][2] + ft1[j][4]), int(ft1[j][3] + ft1[j][5]))
                # couleur du rectangle
                r = random.randint(0,255)
                g = random.randint(0,255)
                b = random.randint(0,255)
                #pour avoir des id uniques
                for c in prev_colors:
                    if c[0] == r and c[1] == g and c[2] == b :
                        r = random.randint(0,255)
                        g = random.randint(0,255)
                        b = random.randint(0,255)
                prev_colors.append([r,g,b])
                color = (r,g,b)
                #writing the object in the file
                if  i == len(ft)-1:
                    #giving and unique id to those that dont have one
                    if ft1[j][1] == 0:
                        ft1[j][1] = r+g+b
                    file.write('\n')
                    file.write(str(ft1[j][0]))
                    file.write(',')
                    file.write(str(ft1[j][1]))
                    file.write(',')
                    file.write(str(ft1[j][2]))
                    file.write(',')
                    file.write(str(ft1[j][3]))
                    file.write(',')
                    file.write(str(ft1[j][4]))
                    file.write(',')
                    file.write(str(ft1[j][5]))
                    file.write(',')
                    file.write(str(ft1[j][6]))
            i+=1
        if z == -1: # On choisi la derniere couleur assigner pour chaque objet créant une nouvelle trajectoire
            col.append([color[0],color[1],color[2]])
            # Dessin du rectangle
            image = cv2.rectangle(image, start_point, end_point, color, 2)
        j+=1

    #compter les mauvais ids
    for x in ft_verf:
        for y in ft1:
            if (x[1]!=-1 and y[1]!=-1) and (x[0]==t and x[1] == y[1]):
                ids+=1
        if ids>1:
            ids-=1
    IDS.append(ids)
    #print(IDS)
    # mise a jour des couleurs precedentes
    prev_colors = col
    t+=1
    # affichage de l'image
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

    # Libere la memoire
    del ious
    gc.collect()

file.close()
mota = 0
x = 0
err = 0
gt = 0
while x < t-1: #on fait t-1 a cause du dernier increment qui arrete la boucle au dessus
    #print('datas: ', FNs[x], ', ', FPs[x], ', ', IDS[x])
    err += (FNs[x] + FPs[x] + IDS[x])
    gt += GTs[x]
    x+=1
mota = 1 - err/gt
print("Resultas: \n     MOTA = ", mota)

#del boxes
del images
del ground_truth
del FNs
del FPs
del IDS
gc.collect()
