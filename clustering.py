
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt

df = pd.read_excel('CompletiMolnarPoli.xlsx', index_col=None) 
df

sent1 = set(df['Descrizione Molnar'][:400]) # con 'set' rimuovo i duplicati
sent2 = set(df['Descrizione Molnar']) # stesse frasi

bleu_scores = []

for s1 in sent1:
    bleu_scores.append([1 - sentence_bleu([s1], s2, weights = [1],) for s2 in sent2]) 
    ### con 1 - BLEU calcolo la distanza tra le frasi
    
df1 = pd.DataFrame(bleu_scores, index=list(sent1), columns=list(sent2))
df1



df1.values # matrice di distanza
 #https://stackoverflow.com/questions/35873273/display-cluster-labels-for-a-scipy-dendrogram
labelList = df1.index
Z = sch.linkage(df1.values, 'complete')
R = sch.dendrogram(Z,no_plot=True)
labelDict = {leaf: labelList[leaf] for leaf in R["leaves"]} #salva le frasi/label in un dizionario per stamparle dopo

plt.figure(figsize=(7,30))
sch.dendrogram(Z,leaf_label_func=lambda x:labelDict[x], leaf_font_size=12, orientation= 'left')
plt.show()
