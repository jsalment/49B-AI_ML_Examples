""" 
This file systematically estimates the affinity scores between aptamer 
sequences and target molecules. The purpose is to generate additional data for 
neural network training. This algorithm estimates an affinity score between a 
target and aptamer by dividing the known affinity between a reference target and 
aptamer by the specificity score of the new target with respect to the referece 
target. 

For example: estimated affinity of morphine = 
(known affinity between aptamer and cocaine) / 
(morphine's specificity with aptamer normalized 
against cocaine's specificity with aptamer) 

"""
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--affin_path", type=str, default="data/sequence.csv",
                    help="Path to csv with aptamer-target affinities")
parser.add_argument("--spec_dir", type=str, 
                    default="data/Specificities",
                    help="Directory containing specificity csv files")
parser.add_argument("--out_path", type=str, default="data/seq_SMILES_large.csv",
                    help="Path to save combined estimated affinity file")

args = parser.parse_args()

# --- Utility: nucleotide sequence to SMILES ---
def convert_sequence2SMILES(seq):
    converter = {
        'A': "C1=NC2=NC=NC(=C2N1)N",
        'C': "C1=C(NC(=O)N=C1)N",
        'G': "C1=NC2=C(N1)C(=O)NC(=N2)N",
        'T': "CC1=CNC(=O)NC1=O"
    }
    return "".join(converter.get(char, "") for char in seq)

# --- Known target SMILES directory ---
target_dir = {
    'Cocaine': "CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)OC",
    '(+)-Methamphetamine': 'C[C@@H](CC1=CC=CC=C1)NC',
    'Methamphetamine': 'C[C@@H](CC1=CC=CC=C1)NC',
    'Flunixin': 'CC1=C(C=CC=C1NC2=C(C=CC=N2)C(=O)O)C(F)(F)F',
    'AB-Fubinaca': 'CC(C)[C@@H](C(=O)N)NC(=O)C1=NN(C2=CC=CC=C21)CC3=CC=C(C=C3)F',
    'Heroin': "CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5[C@]2([C@H]1OC5=C(C=C4)OC(=O)C)CCN3C",
    'Oxycodone': 'CN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)OC)O4)O',
    'Morphine': 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O',
    'Fentanyl': 'CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3',
    'Acetyl fentanyl': 'CC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3',
    'Furanyl fentanyl': 'C1CN(CCC1N(C2=CC=CC=C2)C(=O)C3=CC=CO3)CCC4=CC=CC=C4',
    'XLR-11': 'CC1(C(C1(C)C)C(=O)C2=CN(C3=CC=CC=C32)CCCCCF)C',
    'THC': 'CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1)(C)C)C)O',
    'Mephedrone': 'CC1=CC=C(C=C1)C(=O)C(C)NC',
    '(-)-MDPV': 'CCCC(C(=O)C1=CC2=C(C=C1)OCO2)N3CCCC3.Cl',
    'AB-FUBINACA': 'CC(C)[C@@H](C(=O)N)NC(=O)C1=NN(C2=CC=CC=C21)CC3=CC=C(C=C3)F',
    'Lysergic acid diethylamide': 'CCN(CC)C(=O)[C@H]1CN([C@@H]2CC3=CNC4=CC=CC(=C34)C2=C1)C',
    'Tetrahydrocannabinol': 'CCCCCC1=CC(=C2[C@@H]3C=C(CC[C@H]3C(OC2=C1)(C)C)C)O',
    'Cannabidiol': 'CCCCCC1=CC(=C(C(=C1)O)[C@@H]2C=C(CC[C@H]2C(=C)C)C)O',
    'Cannabinol': 'CCCCCC1=CC(=C2C(=C1)OC(C3=C2C=C(C=C3)C)(C)C)O',
    'Alprazolam': 'CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4',
    'Diazepam': 'CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3',
    'Quinine': 'COC1=CC2=C(C=CN=C2C=C1)[C@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O',
    'Sumatriptan': 'CNS(=O)(=O)CC1=CC2=C(C=C1)NC=C2CCN(C)C',
    'Serotonin': 'C1=CC2=C(C=C1O)C(=CN2)CCN',
    'L-Tryptophan': 'C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N',
    'Dehydroepiandrosterone sulfate': 'C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CC=C4[C@@]3(CC[C@@H](C4)OS(=O)(=O)O)C',
    'N,N-Dimethyltryptamine': 'CN(C)CCC1=CNC2=CC=CC=C21',
    'Benzocaine': 'CCOC(=O)C1=CC=C(C=C1)N',
    'Procaine': 'CCN(CC)CCOC(=O)C1=CC=C(C=C1)N',
    'Diphenhydramine': 'CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2',
    'Lidocaine': 'CCN(CC)CC(=O)NC1=C(C=CC=C1C)C',
    'MDMA': 'CC(CC1=CC2=C(C=C1)OCO2)NC',
    'Amphetamine': 'CC(CC1=CC=CC=C1)N',
    'Methadone': 'CCC(=O)C(CC(C)N(C)C)(C1=CC=CC=C1)C2=CC=CC=C2',
    'Acetaminophen': 'CC(=O)NC1=CC=C(C=C1)O',
    '(+)-Pseudoephedrine': 'C[C@@H]([C@H](C1CC=CC=C1)O)NC',
    'Pseudoephedrine': 'C[C@@H]([C@H](C1=CC=CC=C1)O)NC',
    'Codeine': 'CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O',
    'Chlorpromazine': 'CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl',
    'Lactose': 'C([C@@H]1[C@@H]([C@@H]([C@H]([C@@H](O1)O[C@@H]2[C@H](OC([C@@H]([C@H]2O)O)O)CO)O)O)O)O',
    'Mannitol': 'C([C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)O',
    'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'Lorazepam': 'C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl',
    'Papaverine': 'COC1=C(C=C(C=C1)CC2=NC=CC3=CC(=C(C=C32)OC)OC)OC',
    'Noscapine': 'CN1CCC2=CC3=C(C(=C2[C@@H]1[C@@H]4C5=C(C(=C(C=C5)OC)OC)C(=O)O4)OC)OCO3',
    'Benzoylecgonine': 'CN1[C@H]2CC[C@@H]1[C@H]([C@H](C2)OC(=O)C3=CC=CC=C3)C(=O)O',
    'Methylphenidate': 'COC(=O)C(C1CCCCN1)C2=CC=CC=C2',
    'Levamisole': 'C1CSC2=N[C@H](CN21)C3=CC=CC=C3',
    'Scopolamine': 'CN1[C@@H]2CC(C[C@H]1[C@@H]3[C@H]2O3)OC(=O)[C@@H](CO)C4=CC=CC=C4',
    'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'Fluoxetine': 'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',
    'Nicotine': 'CN1CCC[C@H]1C2=CN=CC=C2',
    'Dopamine': 'C1=CC(=C(C=C1CCN)O)O',
    'UR-144': 'CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3C(C3(C)C)(C)C',
    'Methylenedioxypyrovalerone': 'CCCC(C(=O)C1=CC2=C(C=C1)OCO2)N3CCCC3',
    'p-Hydroxymethamphetamine': 'C[C@H](CC1=CC=C(C=C1)O)NC',
    'Homovanillic acid': 'COC1=C(C=CC(=C1)CC(=O)O)O',
    'Phenylalanine': 'C1=CC=C(C=C1)C[C@@H](C(=O)O)N',
    '3,4-Dihydroxyphenylacetic acid': 'C1=CC(=C(C=C1CC(=O)O)O)O',
    'Norepinephrine': 'C1=CC(=C(C=C1[C@H](CN)O)O)O',
    'Epinephrine': 'CNC[C@@H](C1=CC(=C(C=C1)O)O)O',
    'Tyrosine': 'C1=CC(=CC=C1C[C@@H](C(=O)O)N)O',
    'Tyramine': 'C1=CC(=CC=C1CCN)O',
    'Naloxone': 'C=CCN1CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O',
    'Naltrexone': 'C1CC1CN2CC[C@]34[C@@H]5C(=O)CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O',
    'Methylnaltrexone': 'C[N+]1(CC[C@]23[C@@H]4C(=O)CC[C@]2([C@H]1CC5=C3C(=C(C=C5)O)O4)O)CC6CC6',
    'Levamisole': 'C1CSC2=N[C@H](CN21)C3=CC=CC=C3',
    'Xylazine': 'CC1=C(C(=CC=C1)C)NC2=NCCCS2',
    'Ethylone': 'CCNC(C)C(=O)C1=CC2=C(C=C1)OCO2',
    'Clonazepam': 'C1C(=O)NC2=C(C=C(C=C2)[N+](=O)[O-])C(=N1)C3=CC=CC=C3Cl',
    'Bupropion': 'CC(C(=O)C1=CC(=CC=C1)Cl)NC(C)(C)C',
    '5F-AMB': 'CC(C)[C@@H](C(=O)OC)NC(=O)C1=NN(C2=CC=CC=C21)CCCCCF',
    'cis-Tramadol': 'CN(C)C[C@H]1CCCC[C@@]1(C2=CC(=CC=C2)OC)O',
    'Clomipramine': 'CN(C)CCCN1C2=CC=CC=C2CCC3=C1C=C(C=C3)Cl',
    'Amoxapine': 'C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl',
    'Citalopram': 'CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F'
}

# --- Load the base affinity data ---
affin_df = pd.read_csv(args.affin_path)
affin_df["Affinity"] = affin_df["Affinity"].astype(str).str.replace(",", "").astype(float)

output_rows = []

# --- Loop through all specificity CSVs ---
for file in os.listdir(args.spec_dir):
    if not file.endswith(".csv"):
        print(f"Skipping {file}. Not a .CSV file type.")
        continue
    print(f"Performing estimates for {file}")
    spec_path = os.path.join(args.spec_dir, file)
    spec_df = pd.read_csv(spec_path)

    # Skip if empty or malformed
    if spec_df.empty or "Target" not in spec_df.columns:
        print(f"Skipping {file}: missing Target column")
        continue

    for _, a_row in affin_df.iterrows():
        # Find corresponding aptamer column in spec_df
        aptamer_id = a_row.get("AptamerID")
        if aptamer_id not in spec_df.columns: continue
        ref_target = str(a_row["Target"]).strip()
        ref_affinity = a_row["Affinity"]

        # perform estimation against every target in spec_df
        for _, s_row in spec_df.iterrows():
            target = str(s_row["Target"]).strip()
            if target.lower() == "specificity score": continue  # skip bottom row
            target_spec = s_row[aptamer_id]
            if target_spec == 0: target_spec = 0.001  # avoid div by zero
            est_affinity = ref_affinity / target_spec

            # get the remaining data for the row, such as SMILES, sequence, etc.
            sequence = a_row["Sequence"] # sequence of the aptamer
            seq_smiles = convert_sequence2SMILES(sequence)
            target_smiles = target_dir.get(target, np.nan)

            # Append the new data as a new row to output_rows
            output_rows.append({
                "AptamerID": aptamer_id,
                "Sequence": sequence,
                "Seq_SMILES": seq_smiles,
                "Target": target,
                "Target_SMILES": target_smiles,
                "Affinity": est_affinity
            })

# # Scan output_rows for missing Target_SMILES values
# missing_targets = set()

# for row in output_rows:
#     tsmiles = str(row.get("Target_SMILES", "")).strip()
#     if not tsmiles or tsmiles.lower() == "nan":
#         target = row.get("Target", "Unknown")
#         missing_targets.add(target)

# print("Targets missing SMILES strings:")
# for t in missing_targets:
#     print(f"- {t}")

# --- Save combined results ---
out_df = pd.DataFrame(output_rows)
out_df.to_csv(args.out_path, index=False)
print(f"Saved {len(out_df)} rows to {args.out_path}")
