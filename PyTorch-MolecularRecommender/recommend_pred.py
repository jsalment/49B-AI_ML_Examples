import argparse
import torch
import numpy as np
import pandas as pd
import recommender
import json
import os
from datetime import datetime
from load_data import decode_char, PRINTABLE_ASCII_CHARS
from converter import target_dir, convert_SMILES2sequence
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",type=str,default="data/seq_SMILES_large.csv",
help="Path to a csv file with nucleotide SMILES, target SMILES string and Affinity score per line. These pairs will form the recommender.")
parser.add_argument("--checkpoint_path",type=str,default="checkpoints/recommender_epoch_best.ckpt",help="Path to a binary file containing pretrained model weights.")
parser.add_argument("--k",type=int,default=5,help="Number of molecules to recommend.")
parser.add_argument("--target",type=str,default="Cocaine",help="Target molecule to provide recommended Oligos for.")
parser.add_argument("--output_path",type=str,required=True,help="Path to output file where results will be saved (e.g., results.json or results.txt).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

df = pd.read_csv(args.data_path)

# Drop Affinities greater than threshold
df = df[df['Affinity'] <= 3000]

# Drop targets with 10 or less entries
value_counts = df["Target"].value_counts()
values_to_drop = value_counts[value_counts < 11].index
df = df[~df['Target'].isin(values_to_drop)]

relevatant_df = df[df['Target'] == args.target]
relevatant_df = relevatant_df[relevatant_df['Affinity'] <= 200]
len_rel = len(relevatant_df)

# Map Seq_SMILES -> Sequence (original sequence string)
sequence_dict = pd.Series(df.Sequence.values, index=df.Seq_SMILES).to_dict()

target = target_dir[args.target]

print("Initializing Recommender...")
recommend_model = recommender.MolecularRecommender().to(device)
checkpoint = torch.load(args.checkpoint_path, weights_only=False)
recommend_model.load_state_dict(checkpoint['state_dict'])
recommend_model.eval()
print("Recommender Initialized.")

dataset = recommender.MolecularDataset(
    targets=np.repeat(target, len(df.Seq_SMILES.values), axis=0),
    nucleotides=df.Seq_SMILES.values,
    affinities=df.Affinity.values
)

loader = DataLoader(
    dataset, batch_size=100, shuffle=False, num_workers=8, collate_fn=recommender.targets_nucleotides_batch
)

def recommend_top_target(model, device, loader, seq_smiles_list):
    """
    Return a list of tuples (seq_smiles, target_name, predicted_affinity)
    in ascending order of predicted_affinity.
    """
    model.eval()
    predictions = []
    pointer = 0

    with torch.no_grad():
        for batch in loader:
            targets_tensor = batch[0]
            nuc_tensor = batch[1]
            preds = model(targets_tensor.to(device), nuc_tensor.to(device))
            preds = preds.cpu().detach().numpy().reshape(-1)

            batch_size = len(preds)
            batch_seq_smiles = seq_smiles_list[pointer: pointer + batch_size]
            pointer += batch_size

            for seq_smiles, affinity_pred in zip(batch_seq_smiles, preds):
                predictions.append((seq_smiles, args.target, float(affinity_pred)))

    predictions.sort(key=lambda x: x[2])
    return predictions

seq_smiles_list = df.Seq_SMILES.values.tolist()

print(f"Begin search of {len(seq_smiles_list)} Oligos.")
all_predictions = recommend_top_target(recommend_model, device, loader, seq_smiles_list)

# Deduplicate by Seq_SMILES
seen = set()
unique_recommendations = []
for seq_smiles, tgt, pred_aff in all_predictions:
    if seq_smiles in seen:
        continue
    seen.add(seq_smiles)
    seq = sequence_dict.get(seq_smiles, None)
    if seq is None:
        try:
            seq = convert_SMILES2sequence(seq_smiles)
        except Exception:
            seq = seq_smiles
    unique_recommendations.append((seq, seq_smiles, tgt, pred_aff))

num_rel_k = 0
num_rel = 0
for i, rec in enumerate(unique_recommendations):
    sequence = rec[0]
    is_relevant = relevatant_df['Sequence'].eq(sequence).any()
    if is_relevant and i < args.k:
        num_rel_k += 1
        num_rel += 1
    elif is_relevant and i < len_rel:
        num_rel += 1

precision_at_k = num_rel_k / args.k if args.k > 0 else 0.0
print(f"Precision@{args.k}: {precision_at_k:.2f}")
if len_rel:
    print(f'Precision@{len_rel}: {num_rel/len_rel:.2f}')

top_k = [
    {"Sequence": seq, "Target": tgt, "PredAffinity": aff}
    for seq, seq_smiles, tgt, aff in unique_recommendations[:args.k]
]
bottom_k = [
    {"Sequence": seq, "Target": tgt, "PredAffinity": aff}
    for seq, seq_smiles, tgt, aff in unique_recommendations[-args.k:]
]

print(f"Top {args.k} unique Oligos for {args.target}:")
for entry in top_k:
    print(entry)

print(f"\nBottom {args.k} unique Oligos for {args.target}:")
for entry in bottom_k:
    print(entry)

# Save output to file
output_data = {
    f"Target: {args.target}\n" + \
    f"Precision@k: {precision_at_k}\n" + \
    f"Top_k: {top_k}\n" + \
    f"Bottom_k: {bottom_k}\n" 
}

with open(args.output_path, "a") as f:
    f.write(str(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Checkpoint: {args.checkpoint_path}\n\n"))
    f.write(str(output_data))

print(f"\nResults saved to: {args.output_path}")
