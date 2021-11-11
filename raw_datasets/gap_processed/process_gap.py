import pandas as pd

def process(inputFileName, A_or_B):
    start_token = "<bop>"
    end_token = "<eop>"
    df = pd.read_csv(f'{inputFileName}.tsv', sep = "\t", header=0, names=["ID", "Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref", "B", "B-offset", "B-coref", "URL"])
    df['new_text'] = ""
    for i in range(len(df)):
        row = df.iloc[i]
        end = row['Pronoun-offset'] + len(row['Pronoun'])
        assert row['Text'][row['Pronoun-offset']:end] == row['Pronoun']
        new_text = row['Text'][:row['Pronoun-offset']] + start_token + row['Pronoun'] + end_token + row['Text'][end:]
        df.at[i, 'new_text'] = new_text

    # A_or_B = "B"
    start_token = "<bop>"
    end_token = "<eop>"
    # df = df[['new_text', 'A', 'A-offset', 'A-coref', "Pronoun-offset"]]
    for i in range(len(df)):
        row = df.iloc[i]
        start = row[f'{A_or_B}-offset']
        if row[f'{A_or_B}-offset'] > row['Pronoun-offset']:
            start += len(start_token) + len(end_token)
        end = start + len(row[A_or_B])
        assert row['new_text'][start:end] == row[A_or_B]
        new_text = row['new_text'][:start] + start_token + row[A_or_B] + end_token + row['new_text'][end:]
        df.at[i, f'new_text{A_or_B}'] = new_text
    df = df[[f'{A_or_B}-coref', f'new_text{A_or_B}']]

    # mapping = {True: 1, False: 0}
    # df = df.replace({f'{A_or_B}-coref': mapping})
    df.to_csv(f"{A_or_B}.txt", header=None, index=None, sep='\t', mode='w')
inputFileName = '../gap/gap-validation'
process(inputFileName, "A")
process(inputFileName, "B")
df_A = pd.read_csv(f"A.txt", sep = "\t", header=None, names=["Label", "Text"])
df_B = pd.read_csv(f"B.txt", sep = "\t", header=None, names=["Label", "Text"])
df = df_A.append(df_B)
df.to_csv(f"valid.txt", header=None, index=None, sep='\t', mode='w')