import pandas as pd
from scipy.stats import spearmanr

def calculate_spearmanr(file1_col, file2_col):
    spearman_corr, p_value = spearmanr(file1_col, file2_col)
    return spearman_corr, p_value

def main():
    # Imposta i percorsi dei file
    file1_path = 'CompletiMolnarPoli.xlsx'
    file2_path = 'risultatiF.xlsx'
    output_path = 'output_file.xlsx'

    #df
    file1_data = pd.read_excel(file1_path, usecols=['Giudizio Molnar'], nrows=1022)
    file2_data = pd.read_excel(file2_path, nrows=1022)

    #lista per i risultati
    result_list = []

    # Itera sulle colonne del secondo file
    for col_name in file2_data.columns:
        # Calcola il coefficiente di correlazione di Spearman e il p-value
        spearman_corr, p_value = calculate_spearmanr(file1_data['Giudizio Molnar'], file2_data[col_name])

        # Aggiungi i risultati alla lista
        result_list.append({'nome feature': col_name, 'Spearman': spearman_corr, 'p-value': p_value})

    #df risultati
    output_df = pd.DataFrame(result_list)

    #output
    output_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    main()

