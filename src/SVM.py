import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import joblib

cale_svm = os.path.join("models/", "svm_hog_linear.pkl")
cale_scaler = os.path.join("models/", "scaler_hog.pkl")

# dimensiunea la care vor fi redimensionate toate imaginile
# inainte de a ne apuca sa calculam HOG
# HOG lucreaza pe grile fixe - daca am avea dimensiuni diferite am complica antrenarea SVMului
IMG_SIZE = (224, 224)

# dictionar care mapeaza parametrii functiei HOG
HOG_PARAMS = {
    "orientations": 9,           # nr de bins in histograma pt gradient
    "pixels_per_cell": (8, 8),   # dimensiunea fiecarei celule in pixeli
    "cells_per_block": (2, 2),   # blocurile de normalizare se formeaza din 2x2 celule adiacente
    "block_norm": "L2-Hys",      # varianta de normalizare L2 cu hidrostabilizare - se taie valorile prea mari si se renormalizeaza
    "transform_sqrt": True       # transformare care reduce efectul diferentelormari de lumina, avand pe final un contrast uniform
}

def extragere_featureuri_hog(cale_imagine, dimensiune_imagine=IMG_SIZE, parametri_hog=None):
    

    # Functie care incarca o imagine din memorie folosind cale_imagine si care o preproceseaza
    # si returneaza un vector de caracteristici HOG - sir de numere care descriu
    #structurile de margini si contraste din acea imagine.

    # in cazul in care parametrul e None
    # ii dam parametri setati global in HOG_PARAMS
    if parametri_hog is None:
        parametri_hog = HOG_PARAMS      

    # citim imaginea din memoria in formatul OpenCV - Blue-Green-Red
    imagine_BGR = cv2.imread(cale_imagine)  

    # modificam dimensiunea imaginii
    # la valorile pe care le-am dat anterior
    # metoda de interpolare este buna la micsorarea
    # imaginilor (reduce aliasing)
    imagine_redimensionata = cv2.resize(imagine_BGR, dimensiune_imagine, interpolation=cv2.INTER_AREA) 

    # HOG e bazat pe img in
    # gri, nu pe BGR cum e in OpenCV
    # deci convertim imaginea la o singura
    # componenta anume intensitatea
    culoare_gri = cv2.cvtColor(imagine_redimensionata, cv2.COLOR_BGR2GRAY) 

    # functia hog importata din skimage.feature
    # primeste ca parametru imaginea gri
    # si restul parametrilor care ii spun cum sa calculeze histogramele de gradient
    vector_hog = hog(
        
        # imaginea este oarecum impartita in celule de 8x8 (cum e dat ca parametru in variabila globala in cazul nostru)
        # Pentru fiecare celula se calculeaza histrograma directiilor de gradient - 9 bins
        # mai apoi grupurile de 4 celule sunt normalizate prin L2-Hys
        # se aplica apoi radacina patrata pentru a face imaginile mai uniforme
        # si in final obtinem un vector 1D
        culoare_gri,
        orientations=parametri_hog["orientations"],
        pixels_per_cell=parametri_hog["pixels_per_cell"],
        cells_per_block=parametri_hog["cells_per_block"],
        block_norm=parametri_hog["block_norm"],
        transform_sqrt=parametri_hog["transform_sqrt"],
        visualize=False,
        feature_vector=True
    )
    return vector_hog


def construire_matrice_featureuri(cale_fisier_csv, director_imagini, include_etichete=True):

    # Functia are rolul de a parcurge un fisier CSV dat ca parametru prin parametrul de cale
    # si de a extrage pentru fiecare imagine descriptorul HOG 
    # iar in final intoarce 2 matrici
    # - una cu featureuri = vectorii HOG
    # - alta cu etichetele

    df = pd.read_csv(cale_fisier_csv)   # incarcam continul fisierului CSV in DataFrame
    lista_featureuri = []               # lista in care vom adauga vectorii HOG
    lista_etichete = []                 # lista in care adaugam etichetele, daca include_etichete e false la final returnam None
    iduri_imagini = []                  # lista in care salvam id-ul fiecarei imagini, preluat din CSV

    for idx, linie in df.iterrows():                                           # intoarce cate un obiect din fisier
        id_imagine = linie["image_id"]                                         # luam id-ul imaginii din fisier
        cale_imagine = os.path.join(director_imagini, f"{id_imagine}.png")     # ii gasim calea din fisier si adaugam la final extensia potrivita
        vector_hog = extragere_featureuri_hog(cale_imagine)                    # pentru fiecare imagine pe care o gasim in fisier
                                                                               # ii calculam matricea de HOG cu functia descrisa anterior
        lista_featureuri.append(vector_hog)                                    # adaugam vectorul calculat anterior in lista
                                                                               # la final vom avea un vector pentru fiecare imagine din CSV
        if include_etichete:              # in cazul in care includem si etichetele includem si coloana labels din CSV
            lista_etichete.append(int(linie["label"]))

        iduri_imagini.append(id_imagine) # pastram ordinea imaginilor

    # convertim lista de featureuri intr un array din numpy bidimensional
    # de tipul (numar de imagini, numar componente din HOG)
    X = np.array(lista_featureuri, dtype=np.float32)

    # daca include_etichete nu e none
    # convertim lista de etichete intr-un array numpy bidimensional
    # de tipul (numar de imagini,)
    y = np.array(lista_etichete, dtype=np.int64) if include_etichete else None
    return X, y, iduri_imagini


def main():
    # ne asiguram ca exista folderul in care am salvat modelul
    # pentru a evita reantrenarea 
    os.makedirs("models/", exist_ok=True)

    # verific daca exista deja modele antrenate in folderul de models
    exista_model = os.path.isfile(cale_svm) and os.path.isfile(cale_scaler)

    # in cazul in care am gasit un model
    # nu mai trec prin datele de antrenare si cele de validare
    # si programul se muta direct la maparea imaginilor din
    # folderul de test
    if exista_model:
        print("Se incarca modele deja existente de pe disc.\n")
        scaler = joblib.load(cale_scaler)
        svm = joblib.load(cale_svm)
    # in cazul in care nu avem niciun model deja salvat
    # iteram prin datele de antrenare si validare si ne antrenam
    # modelul
    else:

        print("Nu am gasit niciun model, asa ca vom antrena unul de la 0.\n")
        
        # nu am nevoie aici de image_ids, de aceea am folosit un wildcard
        print("Incep sa contruiesc featureuri pe datele de antrenare\n")
        x_antrenare, y_antrenare, _ = construire_matrice_featureuri("../data/train.csv", "../data/train/", include_etichete=True)
        print("x_antrenare", x_antrenare.shape, "y_antrenare", y_antrenare.shape)
    
        # nu am nevoie de image_ids, de aceea am folosit un wildcard
        print("Incep sa contruiesc featureuri pe datele de validare\n")
        x_validare, y_validare, _ = construire_matrice_featureuri("../data/validation.csv", "../data/validation/", include_etichete=True)
        print("x_validare", x_validare.shape, "y_validare", y_validare.shape)

        # aici are loc normalizarea imaginilor folosind StandardScaler
        # care le aduce pe toate la o medie 0 si o deviatie standard de 1
        print("Aplic standard scaler.\n")
        scaler = StandardScaler()
        x_antrenare_scalat = scaler.fit_transform(x_antrenare)
        x_validare_scalat = scaler.transform(x_validare)

        # implementam SVMul linira 
        # care are parametrul de regulizare setat la 1
        # si 200 de iteratii maxime per eticheta
        print("Incep antrenarea")
        svm = LinearSVC(C=1.0, max_iter=2, verbose=1)
        svm.fit(x_antrenare_scalat, y_antrenare)

        # dupa antrenarea modelului anterior putem face testul pe datele de validare
        
        print("Intram pe datele de validare")
        y_validare_prezis = svm.predict(x_validare_scalat)
        acuratete_validare = accuracy_score(y_validare, y_validare_prezis)
        print("Acuratete validare:", round(acuratete_validare * 100, 5), "\n")
        print("Raport de clasificare")
        print(classification_report(y_validare, y_validare_prezis, digits=4))

        # dupa ce am afisat acuratetea pe datele de validare putem salva modelul antrenat
        print("Salvez model")
        joblib.dump(svm, cale_svm)
        joblib.dump(scaler, cale_scaler)
        print("Modele salvate cu succes.")

    # putem trece acum la predictiile pe datele de test
    print("Incep sa contruiesc featureuri pe datele de test\n")
    x_test, y_test, iduri_imagini_test = construire_matrice_featureuri("../data/test.csv", "../data/test/", include_etichete=False)
    print("x_test shape:", x_test.shape)
    print("x_test shape:", x_test.shape)

    print("Scalez datele de test cu acelasi StandardScaler\n")
    x_test_scalat = scaler.transform(x_test)

    print("Folosind modelul SVM fac predictii pe datele de test\n")
    y_test_pred = svm.predict(x_test_scalat)

    print("Construiesc dataframeul pentru trimitere.")
    submission_df = pd.DataFrame({
        "image_id": iduri_imagini_test,
        "label": y_test_pred
    })

    print("Creez CSV-ul specific")
    submission_df.to_csv("submission.csv", index=False)
    print("Fișierul 'submission.csv' a fost generat cu succes.")


if __name__ == "__main__":
    main()
