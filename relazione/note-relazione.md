## Monk Task 2 in generale (Paolo)

- Il dataset era discretamente sbilanciato, per cui è importante notare che la misura di accuracy
non è totalmente veritiera. Per questo abbiamo utilizzato anche la balanced accuracy, che tiene conto
delle diverse proporzioni tra le classi.

    Esempio: 99% dei dati appartengono a classe A, l'1% appartiene alla classe B. Un modello che assegna
    tutti i dati sempre e solo ad A ottiene una accuracy del 99%, quando quella reale è del 50%.

- Un dataset sbilanciato può anche influire negativamente sul training di un modello perché può portare
il modello a sviluppare un bias verso la majority class, rendendo difficile l'apprendimento e la
corretta classificazione della minority class.

---  

## SVM per Monk Task 2 (Paolo)
- SVM per il Task 2 per fortuna da un 100% di accuracy, per cui il modello funziona bene. Nel caso un cui
avessimo avuto una accuracy più bassa, avremmo potuto provare tecniche di undersampling od oversampling,
rispettivamente Condensed Nearest Neighbor (CNN) e SMOTE.

- CNN è un algoritmo di undersampling (quindi agisce sulla majority class) che tende a preservare i dati
che si trovano sul decision boundary (ritenuti importanti per la classificazione) e a toglierne alcuni più
lontani da esso (meno importanti) al fine di bilanciare le due classi

- SMOTE è un algoritmo di oversampling (quindi agisce sulla minority class) che, in parole molto povere,
finché non ha bilanciato i dati delle due classi sceglie due punti della minority class, traccia una riga
immaginaria tra i due e crea un dato lungo questa linea (assegnato alla minority class).

- CNN è buono per SVM perché tende a preservare i dati sul decision boundary (per SVM, i support vector). SMOTE è
buono perché aggiunge dati che sono o inutili oppure sono dei support vector (nel caso in cui i due punti scelti
siano appunto dei vettori di supporto).

- C'è da dire che i dati sono molto pochi, per cui sarebbe stato preferibile SMOTE rispetto a CNN

- Ovviamente non serve scrivere tutto questo visto che non fa parte del programma, ma giusto un
accenno


---

## Random Forest e NN per Monk Task 2 (Paolo)

- Il Random Forest è ritenuto abbastanza robusto nei confronti dei dataset sbilanciati

- In generale comunque, sia Random Forest che NN possono essere provati con tecniche di undersampling/oversampling.

---

## NN per cup2023 (Paolo)

- Ci sono due notebook: in uno è stata fatta una grid search con CV=5, nell'altro è stato utilizzato Optuna.
- Optuna è una libreria per il tuning degli iperparametri, che utilizza l'ottimizzazione bayesiana. 
Consente di trovare la migliore configurazione di iperparametri equilibrando tra l'esplorazione di nuove 
combinazioni e l'approfondimento di quelle promettenti. Inoltre, migliora l'efficienza della ricerca con una tecnica 
detta "pruning", interrompendo precocemente i trial meno promettenti.
- (Andrea) Questo è molto interessante, però il Micheli una full grid search la vuole vedere. Può essere nidificata, ma un'esplorazione completa la vuole.
---



## Altre note "metodologiche" (Andrea)

### Osservazioni:
- Se riscaliamo/normalizziamo i valori target per la CUP, bisogna ricordarsi di riportare i risultati (ovvero il MEE ottenuto sui test) nella scala originale, sia sulle slides, sia nel file con i risultati finali

      (Paolo) I valori target non vengono mai scalati, vengono scalate solo le feature, quindi non ci sono problemi 
      (Andrea) Ah ok, quindi le predizioni del modello hanno la stessa scala, per così dire, dei valori originali
      (Paolo) Esatto, feature scalate e output nella scala originale
      (Andrea) Ottimo allora! :D

- Pulire le celle nel notebook fa risparmiare spazio. Questo è importante perché il pacchetto da consegnare deve avere dimensione massima 20MB, tentando di mantenersi sui 5MB
      
      (Paolo) Pulire in che senso?
      (Andrea) Vicino all'output della cella, in un angolino in alto a sx, ci sono tre pallini e compare l'opzione "clear cell output". Almeno, su VSCode è così, ma credo ci sia una cosa analoga anche su browser. Quando si esegue la cella sotto compare l'output, con pulire le celle del notebook intendo "cancellare" questi output
      (Paolo) Non penso sia una buona idea pulire gli output perché ci sono tutti i risultati delle metriche, fossi nel prof vorrei vedere che nel report non mi dici cazzate. Direi di preoccuparcene quando e se effettivamente avremo problemi di dimensioni dello zip
      (Andrea) Quello che dici ha molto senso, ho scritto quel commento perché ho trovato scritta questa cosa nelle istruzioni del professore. Comunque, i risultati delle metriche, almeno quelli più importanti, vanno messi nel report (quelli non strettamente necessari/non richiesti possono essere messi in appendice). Idem i plots. Ad ogni modo, certo, possiamo preoccuparcene se e quando avremo problemi di spazio.

### Cosa consegnare (Andrea)

Faccio un'estrema sintesi delle indicazioni fornite dal professore (su Moodle c'è un pacchetto con tutto) per quanto riguarda la consegna:

- [] Bisogna consegnare un pacchetto in formato .zip, contenente:
    - [] Il **codice** (c'è scritto: "se originale, altrimenti citare le librerie". Secondo me quindi si può non mettere per risparmiare spazio se necessario. Comunque all'orale bisogna averlo)
    - [] **team-name_ML-CUP23-TS.csv**, il file con i risultati per la coppa. Indicazioni precise sul formato sono sulle slides, o in short-info.txt (nulla di complicato comunque)
    - [] **team-name_abstract.txt**, un riassunto conciso (max 5 righe) del modello scelto
    - [] Il **Report**, sotto forma di slides

### Report:





