import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from multiprocessing import freeze_support

class SetDate(Dataset):

    # Clasa care are scopul de a incarca
    # preprocesa si oferi imagini si etichetele asociate - daca este cazul
    # pentru antrenarea/validarea/testarea unui model

    # constructor pentru SetDate
    # primeste in dataframe din biblioteca pandas
    # cu coloane de tipul
    # image_id, label
    # director_imagini este calea pentru fisierul de imagini
    # transform sunt augmentarile datelor
    def __init__(self, dataframe, director_imagini, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.director_imagini = director_imagini
        self.transform = transform
        self.etichete = 'label' in self.dataframe.columns

    def __len__(self):
        return len(self.dataframe)

    # preia informatii despre imaginea
    # de la pozitia idx
    # apoi extrage id-ul sau unic
    # cale va reprezenta variabila care navigheaza la acea imagine
    # imaginea este deschisa si apoi transformata
    # conform transformarile definite in functia main
    def __getitem__(self, idx):
        linie = self.dataframe.iloc[idx]
        id_imagine = linie.image_id
        cale = os.path.join(self.director_imagini, f"{id_imagine}.png")
        imagine = Image.open(cale).convert("RGB")
        if self.transform:
            imagine = self.transform(imagine)


        # daca exista etichete vom intoarce
        # imaginea si eticheta sa
        if self.etichete:
            eticheta = torch.tensor(linie.label, dtype=torch.long)
            return imagine, eticheta
        # altfel vom intoarce id-ul imaginii
        # si imaginea in sine
        else:
            return id_imagine, imagine



class CNN(nn.Module):
    # clasa care defineste o structura
    # pentru o CNN destinata clasificarii imaginilor
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),         # transforma imaginea, luand caracteristici locale
                                                                                                  # initial avem 3 canale pentru intrare si progresiv 32 64 128 256 512
                                                                                                  # adaugam si un padding per imagine practic o bordam cu 0 peste tot
            nn.BatchNorm2d(32),                                                                   # apoi iesirea este normalizata
            nn.LeakyReLU(inplace=True),                                                           # permite scurgeri mici pentru valori negative
                                                                                                  # inplace = True punem ca transformarea sa se faca in tensorul de intrare
            nn.MaxPool2d(2, 2),                                                   # reduce dimensiunea la jumatate
                                                                                                  # practic luam cate 4 pixeli odata, o casuta de 2x2 si o mutam
                                                                                                  # cu 2 pe orizontala si verticala ca sa nu existe suprapuneri

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1,1)),                            # realizeaza o mediere adaptiva si vom avea o iesire de dimensiune fixa
            nn.Flatten(),                                           # transforma iesirea intr-un vector unidimensional pentru stratul complet conectat
            nn.Dropout(0.5),                                        # se opresc aleatoriu 50% din neuroni
            nn.Linear(512, 256, bias=False),   # reducea dimensiunea de la 512 la 256
            nn.BatchNorm1d(256), nn.LeakyReLU(inplace=True),        # aplica activarea non-liniara pe vectorul de 256
            nn.Dropout(0.5),                                        # regularizare inaintea stratului final
            nn.Linear(256, num_classes)                   # produce scorurile finale pentru fiecare clasa
        )

    # aplica intreaga structura asupra imaginii de intrare x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def proceseaza_epoca(dataLoader, cnn, pierdereEntropie, obiectOptimizare, componenta):
    # functie care face o parcurgere completa
    #a setului de date de antrenare
    # actualizand parametrii modelului pentru a imbunatati
    # capacitatea de predictie

    cnn.train()         # comuta modelul in modul de antrenare
    totPierdere = 0     # suma pierderilor
    totCorect = 0       # nr de predictii corecte per epoca
    nr_exemple = 0      # nr de imagini procesate la epoca

    # parcurgem fiecare batch de date
    # pentru a afisa progresul
    for id_cos, (imagini, etichete) in enumerate(dataLoader, 1):
        imagini, etichete = imagini.to(componenta), etichete.to(componenta)  # transfera lotul curent pe deviceul pe care se face antrenarea
        obiectOptimizare.zero_grad()                                         # sterge gradientii anteriori
        rezultate = cnn(imagini)                                             # predictiile pt lotul curent
        pierdere = pierdereEntropie(rezultate, etichete)                     # diferenta dintre predictiile retelei si etichetele reale
        pierdere.backward()                                                  # gradientii pentru parametrii modelului
        obiectOptimizare.step()                                              # ajusteaza parametrii retelei folosind optimizatorul definit

        # aici sunt comparate predictiile cu etichetele reale
        # iar apoi se calculeaza proportia predictiilor corecte in lot
        predictii = rezultate.argmax(dim=1)
        comp = predictii == etichete
        comp_float = comp.float()
        acuratete_medie = comp_float.mean()
        acuratete_cos = acuratete_medie.item()

        pierdere_cos= pierdere.item()

        print("Cos", id_cos, "din", len(dataLoader), "|| Pierdere:", round(pierdere_cos,5),  "|| Acuratete:", acuratete_cos)

        # actualizeaza suma totala a pierderii si nr total de predictii corecte
        # totodata se actualizeaza nr total de exemple procesate
        marime_cos = etichete.size(0)
        totPierdere += pierdere_cos * marime_cos
        comp = predictii == etichete
        corect_bool = comp.sum()
        ct = corect_bool.item()
        totCorect += ct
        nr_exemple += marime_cos

    # calculam media pentru intreaga epoca
    pierdere_epoca = totPierdere / nr_exemple
    acuratete_epoca = totCorect / nr_exemple
    print("------------------------------------------------------------")
    print("Epoca finalizata. || Pierdere medie:", round(pierdere_epoca, 5), "|| Acuratete:", acuratete_epoca)
    return pierdere_epoca, acuratete_epoca

def evaluare_epoca(dataLoader, cnn, pierdereEntropie, componenta):
    # functie care permite evaluarea obiectiva a performantei modelului
    # oferind informatii precise despre capacitatea sa de generalizare
    # si corectitudine

    cnn.eval()                    # seteaza modelul in modul de evaluare
    totPierdere = 0               # suma pierderilor pe toate loturile
    totCorect = 0                 # nr total de predictii efectuate corect
    with torch.no_grad():         # pytorch nu va stoca gradienti pe timpul evaluarii, reducand consumul de memorie
        for imagini, etichete in dataLoader:                                       # iteram prin seturile de date
            imagini, etichete = imagini.to(componenta), etichete.to(componenta)    # trimitem datele cate componenta specificata
            rezultate = cnn(imagini)                                               # obtine predictiile
            pierdere = pierdereEntropie(rezultate, etichete)                       # calcularea pierdetii pentru lotul curent
            totPierdere += pierdere.item() * imagini.size(0)                       # si o acumuleaza ponderat la media finala
            predictii = rezultate.argmax(dim=1)                                    # se identifica clasa prezisa cu cea mai mare probabilitate
            rez_corecte = (predictii == etichete).sum().item()                     # verifica cate predictii corespund etichetelor reale
            totCorect += rez_corecte

    # determina pierderea si acuratetea medie pe tot setul de date
    nr_exemple = len(dataLoader.dataset)
    return totPierdere/nr_exemple, totCorect/nr_exemple

def main():
    # se verifica daca antrenarea modelului se poate face pe GPU
    # iar daca da, se muta pe acolo
    componenta = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Modelul se va antrena pe PLACA VIDEO (GPU)")
    else:
        print("Modelul se va antrena pe PROCESOR (CPU)")

    # numarul de epoci
    numar_epoci = 2

    # transformarile facute pe imagini
    transformare = transforms.Compose([
        transforms.Resize((224, 224)),                                                   # duce toate imaginile la dimensiunea de 224x224
        transforms.RandomHorizontalFlip(p=0.5),                                          # cu o probabilitate de 50%, imaginea se poate
        transforms.RandomRotation(degrees=15),                                           # in mod randomizat imaginea are sansa de a se invarti la 15 grade
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),                                                           # converteste imaginile in tensori
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])      #se aplica normalizarea standard ImageNet
    ])

    dataframe_antrenare = pd.read_csv("../data/train.csv")
    dataframe_validare   = pd.read_csv("../data/validation.csv")
    dataframe_test  = pd.read_csv("../data/test.csv")

    dataset_antrenare = SetDate(dataframe_antrenare,"../data/train",transformare)
    dataset_validare = SetDate(dataframe_validare,"../data/validation",transformare)
    loader_antrenare = DataLoader(dataset_antrenare,batch_size=32, shuffle=True,num_workers=4)  # punem shuffle true ca imaginile sa fie amestecate la inceputul fiecarei epoci
                                                                                                # si asignam 4 threaduri pentru dataloader
    loader_validare = DataLoader(dataset_validare,batch_size=32,shuffle=False,num_workers=4)

    cnn = CNN(num_classes=5).to(componenta)
    pierdereEntropie = nn.CrossEntropyLoss()                     # folosim functia de pierdere cross-entropy
    obiectOptimizare = optim.Adam(cnn.parameters(), lr=1e-3)     # optimizatorul adam

    # checkpoint in cazul in care nu am avut timp
    # sa antrenez constant un model
    checkpoint = "best_model.pth"

    # in cazul in care numarul de epoci era 0
    # doar incarcam modelul pentru a scrie in fisierul CSV
    if numar_epoci == 0:
        if os.path.exists(checkpoint):
            incarcare = torch.load(checkpoint, map_location=componenta, weights_only=True)
            cnn.load_state_dict(incarcare)
            print("Model incarcarat")

    # altfel incepeam sa antrenez modelul
    # pe numarul de epoci setat de mine
    else:

        # daca aveam deja un model in memorie
        # il puteam prelua pe acela pentru a continua sa il antrenez
        acuratete_maxima = 0.0
        if os.path.exists(checkpoint):
            incarcare = torch.load(checkpoint, map_location=componenta, weights_only=True)
            cnn.load_state_dict(incarcare)
            print("Modelul a fost incarcat si este pregatit pentru antrenare.")
        # indiferent daca gaseam sau nu un model in memorie
        #incepeam sa antrenez noul model pana la nr de epoci setat de mine
        # folosind functiile definite anterior
        for epoca in range(1, numar_epoci+1):
            proceseaza_epoca(loader_antrenare, cnn, pierdereEntropie, obiectOptimizare, componenta)
            pierdere, acuratete = evaluare_epoca(loader_validare, cnn, pierdereEntropie, componenta)
            print("Epoca", epoca, "|| Pierdere validare:", round(pierdere, 5), "|| Acuratete validare:", acuratete)
            if acuratete > acuratete_maxima:
                acuratete_maxima = acuratete
                torch.save(cnn.state_dict(), checkpoint)
                print("A fost salvat noul model cu acuratetea", acuratete)
            print("------------------------------------------------------------")

    # definit datele de test
    dataset_test = SetDate(dataframe_test, "../data/test", transformare)
    loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    cnn.eval()
    predictii, ids = [], []
    with torch.no_grad():                                               # pytorch nu va stoca gradienti
        for ids_imagini, imagini in loader_test:                        # iteram prin imaginile de test si facem predictii in functie de etichete
            imagini = imagini.to(componenta)
            rezultate = cnn(imagini)
            etichete = rezultate.argmax(dim=1).cpu().numpy().tolist()
            predictii.extend(etichete)
            ids.extend(ids_imagini)

    # la final scriem in fisierul CSV datele
    sub = pd.DataFrame({"image_id": ids, "label": predictii})
    sub.to_csv("submission.csv", index=False)
    print("Fisierul de submisie a fost salvat cu succes.")

if __name__ == "__main__":
    freeze_support() # compatbilitate multiprocessing
    main()
