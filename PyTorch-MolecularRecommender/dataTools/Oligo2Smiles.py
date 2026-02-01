import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="data/sequence.csv", help="Path to a csv file with containing Oligo-nulcleotide sequence")
parser.add_argument("--out_path", type=str, default="data/seq_SMILES.csv", help="Path to create updated csv file with Oligo-nucleotide SMILE")

# Converts nucleotide sequence to SMILES string
def convert_sequence2SMILES(seq):
    output = ""
    converter = {
        'A': "C1=NC2=NC=NC(=C2N1)N",
        'C': "C1=C(NC(=O)N=C1)N",
        'G': "C1=NC2=C(N1)C(=O)NC(=N2)N",
        'T': "CC1=CNC(=O)NC1=O"}
    
    for char in seq:
        output += converter[char]
    
    return output

args = parser.parse_args()
df = pd.read_csv(args.data_path)

target_dir = {
    'Cocaine': "CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC",
    'Methamphetamine': 'C[C@@H](CC1=CC=CC=C1)NC',
    'Flunixin': 'CC1=C(C=CC=C1NC2=C(C=CC=N2)C(=O)O)C(F)(F)F',
    'AB-Fubinaca': 'CC(C)[C@@H](C(=O)N)NC(=O)C1=NN(C2=CC=CC=C21)CC3=CC=C(C=C3)F',
    'Heroin': "CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)OC(=O)C)CCN3C",
    'Oxycodone': 'CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O',
    'Morphine': 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O',
    'Fentanyl': 'CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3',
    'Acetyl Fentanyl': 'CC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3',
    'Furanyl Fentanyl': 'C1CN(CCC1N(C2=CC=CC=C2)C(=O)C3=CC=CO3)CCC4=CC=CC=C4',
    'XLR-11': 'CC1(C(C1(C)C)C(=O)C2=CN(C3=CC=CC=C32)CCCCCF)C',
    'THC': 'CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1)(C)C)C)O',
    'Mephedrone': "CC1=CC=C(C=C1)C(=O)C(C)NC",
    '(-)-MDPV': "CCCC(C(=O)C1=CC2=C(C=C1)OCO2)N3CCCC3.Cl"
}

output = []
max_len = 0
offset = 0
for i, row in df.iterrows():
    if pd.notna(row['Target']):
        output.append({
            'AptamerID': row['AptamerID'],
            'Sequence': row['Sequence'],
            'Seq_SMILES': convert_sequence2SMILES(row['Sequence']),
            'Target': row['Target'],
            'Target_SMILES': target_dir[row['Target']],
            'Affinity': int(row['Affinity'].replace(',',''))
        })
        if len(output[i-offset]['Seq_SMILES'])>max_len:
            max_len=len(output[i-offset]['Seq_SMILES'])
    else:
        offset += 1

pd.DataFrame(output).to_csv(args.out_path, index=False)
print("Maximum length of SMILES string created is", max_len)