# Arthur Kaboré (8530422) et Kashalala David Tshiswaka (8719666)
# CSI4533
# Part 2 du projet

########## Définition des fontions que nous utiliserons ###########################################################################

# equilibrer deux matrices
def equilibre(fst, snd, t):
    if len(fst) > len(snd): # equilibre les deux matrices au cas ou len(fst) > len(snd)
        while len(fst) > len(snd):
            snd.append([t+1,-1,0,0,0,0,0,0,0])
    elif len(fst) < len(snd): # equilibre les deux matrices au cas ou len(fst) < len(snd)
        while len(fst) < len(snd):
            fst.append([t,-1,0,0,0,0,0,0,0])

# Assigne les boites de l'image t a fst et les boites de l'image t+1 a snd
def assign_next(fst,snd,t,boxes):
    for box in boxes:
        if box[0] == t:
            fst.append(box)
        if box[0] == t+1:
            snd.append(box)
    equilibre(fst, snd)

# Calcul du IoU
def iou(u, v):
    X = 0 # x de l'intersection
    Y = 0 # y de l'intersection
    #boite 1
    x1 = u[2]
    x1l = u[2] + u[4]
    y1 = u[3]
    y1h = u[3] + u[5]
    #boite 2
    x2 = v[2]
    x2l = v[2] + v[4]
    y2 = v[3]
    y2h = v[3] + v[5]
    #recherche de la largeur de l'intersection
    if x1 == x2:
        if x1l == x2l:
            X = abs(x1-x1l)
        elif x1l < x2l:
            X = abs(x1-x1l)
        elif x1l > x2l:
            X = abs(x1-x2l)
        else:
            X = 0
    elif x1 < x2:
        if x1l == x2l:
            X = abs(x2-x1l)
        elif x1l < x2l:
            if x2 < x1l:
                X = abs(x2-x1l)
            else:
                X = 0
        elif x1l > x2l:
            X = abs(x2-x2l)
        else:
            X = 0
    elif x1 > x2:
        if x1l == x2l:
            X = abs(x1-x1l)
        elif x1l < x2l:
            X = abs(x1-x1l)
        elif x1l > x2l:
            if x1 < x2l:
                X = abs(x1-x2l)
            else:
                X = 0
        else:
            X = 0
    #recherche de la hauteur de l'intersection
    if y1 == y2:
        if y1h == y2h:
            Y = abs(y1-y1h)
        elif y1h < y2h:
            Y = abs(y1-y1h)
        elif y1h > y2h:
            Y = abs(y1-y2h)
        else:
            Y = 0
    elif y1 < y2:
        if y1h == y2h:
            Y = abs(y2-y1h)
        elif y1h < y2h:
            if y2 < y1h:
                Y = abs(y2-y1h)
            else:
                Y = 0
        elif y1h > y2h:
            Y = abs(y2-y2h)
        else:
            Y = 0
    elif y1 > y2:
        if y1h == y2h:
            Y = abs(y1-y1h)
        elif y1h < y2h:
            Y = abs(y1-y1h)
        elif y1h > y2h:
            if y1 < y2h:
                Y = abs(y1-y2h)
            else:
                Y = 0
        else:
            Y = 0
    #zone d'intersection
    intersection = X*Y
    #zone d'union
    union = (abs(x1-x1l))*(abs(y1-y1h)) + (abs(x2-x2l))*(abs(y2-y2h)) - (X*Y)

    if union == 0:
        return 0
    else:
        return intersection/union

# pour trouver les fn, fp
def verification(fst, snd, Fn, Fp, t):
    fn = 0
    fp = 0
    IOUs = [] # pour la verification
    #equilibre(fst, snd, t)
    ii = 0
    for u in fst:
        if u: #verifie si vide
            b = [] # représente les lignes de la matrice ious
            for v in snd:
                if v: #verifie si vide
                    ii = iou(u,v)
                    if ii < 0.4: # Met à 0 toutes les valeurs IoU inférieur à un seuil de 0.4
                        b.append(0)
                    else:
                        b.append(ii)
            IOUs.append(b)

    # vérifier si un objet dans Ground_truth_with_tracking_cleaned a un objet correspondant dans le fichier de détection (IoU > 0.4)
    # (i->lignes, j->colonnes)
    j = 0
    if IOUs: #verifie si vide
        while j < len(snd): # tant que j < au nombre de colonne de ious
            w = 0
            i = 0
            while i < len(fst): # tant que i < au nombre de ligne de ious
                if IOUs[i] and (IOUs[i][j] > 0.4) and snd[j][1]!=-1: # si IOUs[i] non vide et...
                    w+=1
                i+=1
            if w==0: #nous permet de compter une fois pour chaque objet du Ground_truth_with_tracking_cleaned
                fn+=1
            j+=1
    # Recherche l'entrée j,i >0 ayant la valeur maximum pour chaque j,i
    #(j->lignes, i->colonnes)
    w = 0
    ww = 0
    IOUS_copy = IOUs
    j = 0
    if IOUs: #verifie si vide
        while j < len(fst): # tant que j < au nombre de colonne de ious
            i = 0
            w = IOUs[j][i] # initialisation de la variable qui representera le maximum
            while i < len(snd): # tant que i < au nombre de ligne de ious
                if IOUs[j] and (IOUs[j][i] > w): # si IOUs[i] non vide et...
                    w = IOUs[j][i]
                i+=1

            if w > 0:
                i=0
                while i < len(snd):
                    if IOUs[j] and (IOUs[j][i] == w) and (snd[i][1]!=-1) and (fst[j][1]!=-1):
                        IOUs[j][i] = -1
                    i+=1
            else:
                fp+=1
            j+=1

        j=0
        while j < len(snd):
            #(i->lignes, j->colonnes)
            i = 0
            ww = IOUS_copy[i][j] # initialisation de la variable qui representera le maximum
            while i < len(fst): # tant que i < au nombre de ligne de ious
                if IOUS_copy[i] and (IOUS_copy[i][j] > 0) and (IOUS_copy[i][j] > ww): # si IOUs[i] non vide et...
                    ww = IOUS_copy[i][j]
                i+=1

            if ww > 0:
                i=0
                while i < len(fst):
                    if IOUS_copy[i] and (IOUS_copy[i][j] == ww) and (IOUs[i][j] == -1) and (fst[i][6]==snd[j][7]) and snd[j][1]!=-1 and fst[i][1]!=-1:
                        fst[i][1] = snd[j][1]
                    i+=1
            j+=1

    Fn.append(fn)
    Fp.append(fp)
    #Ids.append(ids)
