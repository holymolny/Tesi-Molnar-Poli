import pandas as pd
from scipy.stats import spearmanr

def calculate_spearmanr(file1_col, file2_col):
    spearman_corr, p_value = spearmanr(file1_col, file2_col)
    return spearman_corr, p_value

def main():
    # Imposta i percorsi dei file
    file1_path = 'CompletiMolnarPoli.xlsx'
    file2_path = 'risultatiDesc.xlsx'
    output_path = 'features_desc.xlsx'

    # Carica i dati dai file
    file1_data = pd.read_excel(file1_path, usecols=['Giudizio Molnar', 'Giudizio Poli'], nrows=1022)
    file2_data = pd.read_excel(file2_path, nrows=1022)

    # Calcola la differenza assoluta tra 'Giudizio Molnar' e 'Giudizio Poli'
    file1_data['Differenza Assoluta'] = abs(file1_data['Giudizio Molnar'] - file1_data['Giudizio Poli'])

    # Inizializza il dataframe per l'output
    output_df = []

    # Itera sulle colonne del secondo file
    for col_name in file2_data.columns:
        # Calcola il coefficiente di correlazione di Spearman e il p-value
        spearman_corr, p_value = calculate_spearmanr(file1_data['Differenza Assoluta'], file2_data[col_name])

        # Aggiungi i risultati alla lista solo se non sono tutte-NA
        if not pd.isna(spearman_corr) and not pd.isna(p_value):
            output_df.append({'features': col_name, 'spearman': spearman_corr, 'p-value': p_value})

    # Crea un DataFrame dai risultati
    output_df = pd.DataFrame(output_df)

    # Salva il dataframe di output in un nuovo file Excel
    output_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    main()
