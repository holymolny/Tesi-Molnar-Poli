
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy.stats import spearmanr
#sbert
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Legge il file Excel
file_excel = 'CompletiMolnarPoli.xlsx'

# Carica il file Excel in un DataFrame
df = pd.read_excel(file_excel)

#file1 = pd.read_excel('Frase1_vec.xlsx')
#file2 = pd.read_excel('Frase2_vec.xlsx')

#header
#[0]Numero Quesito 
#[1]Numero Coppia  
#[2]Frase 1 
#[3]Frase 2 
#[4]Giudizio Molnar 
#[5]Giudizio Poli  
#[6]Giudizio A  
#[7]Giudizio B  
#[8]Giudizio C  
#[9]Giudizio D  
#[10]Giudizio E  
#[11]Descrizione Molnar  
#[12]Descrizione Poli   
#[13]BLEU-SCORE Molnar-Poli  
#[14]SBERT-SCORE    


def bleuScore(df):
    for index, row in df.iterrows():
        sent1 = str(row['Frase 1'])
        sent2 = str(row['Frase 2'])
        print(sentence_bleu([sent1], sent2, weights = [1],))


def kCohen(df):
    arrayMolny = []
    arrayIrene = []
    for index, row in df.iterrows():
        val1 = row['Giudizio Molnar']
        val2 = row['Giudizio Poli']
        arrayMolny.append(val1)
        arrayIrene.append(val2)
    k = cohen_kappa_score(arrayMolny, arrayIrene)
    print(k)

def SpearmanSbertGiudizi(df):
    array = []
    sbert = []
    for index, row in df.iterrows():
        val1 = row['Giudizio Molnar']
        val2 = row['Giudizio Poli']
        diff = abs(int(val1) - int(val2))
        array.append(diff)
        sbert.append(float(row['SBERT-SCORE']))

    corr, p_value = spearmanr(array, sbert)
    print("Correlazione di Spearman: ", corr, "P-value:", p_value)

def SpearmanBleuGiudizi(df):
    array = []
    bleu = []
    for index, row in df.iterrows():
        val1 = row['Giudizio Molnar']
        val2 = row['Giudizio Poli']
        diff = abs(int(val1) - int(val2))
        array.append(diff)
        bleu.append(float(row['BLEU-SCORE Molnar-Poli']))

    corr, p_value = spearmanr(array, bleu)
    print("Correlazione di Spearman: ", corr, "P-value:", p_value)


def SpearmanSbertBleu(df):
    bleu = []
    sbert = []
    for index, row in df.iterrows():
        bleu.append(float(row['BLEU-SCORE Frase1-Frase2']))
        sbert.append(float(row['SBERT-SCORE Frase1-Frase2']))

    corr, p_value = spearmanr(bleu, sbert)
    print("Correlazione di Spearman: ", corr, "P-value:", p_value)


def SpearmanMolnarPoli(df):
    molnar = []
    poli = []
    for index, row in df.iterrows():
        molnar.append(int(row['BLEU-SCORE Frase1-Frase2']))
        poli.append(int(row['SBERT-SCORE Frase1-Frase2']))

    corr, p_value = spearmanr(molnar, poli)
    print("Correlazione di Spearman: ", corr, "P-value:", p_value)


def sbert(df):
    descM = []
    descI = []
    for index, row in df.iterrows():
        descM.append(row['Frase 1'])
        descI.append(row['Frase 2'])

    embeddings1 = model.encode(descM, convert_to_tensor=True)
    embeddings2 = model.encode(descI, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    #Output the pairs with their score
    for i in range(len(descI)):
        #print("{} \t\t {} \t\t Score: {:.4f}".format(descM[i], descI[i], cosine_scores[i][i]))
        print("{:.4f}".format(cosine_scores[i][i]))


def spearmanFeatures(file1, file2, df):
    # Inizializza un dizionario per i risultati
    risultati = []

    start_colonna = 3
    end_colonna = 112
    giudiziMolny = []
    giudiziIrene = []
    for index, row in df.iterrows():
        val1 = row['Giudizio Molnar']
        val2 = row['Giudizio Poli']
        giudiziMolny.append(val1)
        giudiziIrene.append(val2)

    # Calcola la differenza in valore assoluto tra le colonne specificate
    for colonna in range(start_colonna, end_colonna + 1):
        colonna_file1 = file1.iloc[:, colonna]
        colonna_file2 = file2.iloc[:, colonna]
        diff = (colonna_file1 - colonna_file2).abs()

        corr, p_value = spearmanr(diff, giudiziMolny[0])
        print("Correlazione di Spearman: ", corr, "P-value:", p_value)

        #for x in range(len(giudiziMolny)):
            #corr, p_value = spearmanr(diff, giudiziMolny[x])
            #print("Correlazione di Spearman: ", corr, "P-value:", p_value)

        #risultati[f'Colonna_{colonna + 1}'] = risultato_colonna
	# Crea un DataFrame con i risultati
	#risultato_finale = pd.DataFrame(risultati)

	# Scrivi il risultato su un file CSV
	#risultato_file = '/Scrivania/Tirocinio/risultato.csv'
	#risultato_finale.to_csv(risultato_file, index=False)


	#array = []
        #bleu = []
        #for index, row in df.iterrows():
        #val1 = row['Giudizio Molnar']
        #val2 = row['Giudizio Poli']
        #diff = abs(int(val1) - int(val2))
        #array.append(diff)
        #bleu.append(float(row['BLEU-SCORE Molnar-Poli']))

    #corr, p_value = spearmanr(array, bleu)
    #print("Correlazione di Spearman: ", corr, "P-value:", p_value)


def differenza_valore_assoluto(file1, file2, output_file):

    # Verifica che i due dataframe abbiano le stesse dimensioni
    if file1.shape != file2.shape:
        raise ValueError("I due file devono avere le stesse dimensioni.")

    # Calcola la differenza in valore assoluto per ogni cella
    df_diff = file1.subtract(file2).abs()
    print (df_diff)

    # Salva il risultato in un nuovo file Excel
    #df_diff.to_excel(output_file, index=False)

    #try:
        #differenza_valore_assoluto(file1, file2, output_file)
        #print("Differenza in valore assoluto calcolata con successo e salvata in:", output_file)
    #except Exception as e:
        #print("Si è verificato un errore:", str(e))


def differenza_celle_riga_per_riga(file1, file2, start_colonna, end_colonna, risultato_file):
    # Leggi i dati dai file Excel
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Stampa le colonne effettive presenti nei DataFrame
    print("Colonnes in df1:", df1.columns)
    print("Colonnes in df2:", df2.columns)

    # Inizializza una lista per i risultati
    risultati = []

    # Calcola la differenza in valore assoluto tra le coppie di celle (riga per riga) specificate
    for colonna in range(start_colonna, end_colonna + 1):
        differenza_celle = (df1.iloc[1:, colonna] - df2.iloc[1:, colonna]).abs()
        risultati.append(differenza_celle)

    # Combina i risultati in un unico DataFrame
    risultato_finale = pd.concat(risultati, axis=1)

    # Scrivi il risultato su un file di testo
    risultato_finale.to_csv(risultato_file, sep='\t', index=False, header=True)

# Esempio di utilizzo
file1 = 'frase1.xlsx'
file2 = 'frase2.xlsx'
start_colonna = 2  # Se la tua numerazione delle colonne inizia da 1
end_colonna = 129  # Adattare al numero reale delle colonne
risultato_file = 'risultatiF.txt'

differenza_celle_riga_per_riga(file1, file2, start_colonna, end_colonna, risultato_file)


#FUNZIONI RICHIAMATE
#file_path = "dati_q20.txt"
#data = creaDati(file_path)
#bleuScore(df)
#kCohen(df)
#SpearmanSbertGiudizi(df)
#SpearmanMolnarPoli(df)
#SpearmanBleuGiudizi(df)
#SpearmanSbertBleu(df)
#sbert(df)
#spearmanFeatures(file1, file2, df)
#differenza_valore_assoluto("frase1.xlsx", "frase2.xlsx", 'output.ods')
