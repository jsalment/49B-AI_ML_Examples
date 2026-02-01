import argparse
from math import ceil
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import recommender

from transformer import CosineWithRestarts
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from torch.utils.data import DataLoader
from multiprocessing import freeze_support
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True, help="Path to a csv file with nucleotide SMILES, target SMILES string and Affinity score per line. These pairs will form the recommender.")
parser.add_argument("--ckpt_path", type=str, default="checkpoints/pretrained.ckpt", help="Path to a binary file containing pretrained model weights.")
parser.add_argument("--ckpt_save", type=str, required=True, help="Path to a file to save model weights.")
parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size used in the pretrained Transformer.")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers used in the Encoder and Decoder of the pretrained Transformer.")
parser.add_argument("--b_thres", type=int, default=200, help="Threshold below which an affinity score will assumed bind with a target molecule.")
parser.add_argument("--a_thres", type=int, default=3000, help="Affinity upper limit — samples with affinity values above this are dropped entirely from the dataset.")
parser.add_argument("--s_thres", type=int, default=300, help="Affinity cutoff for class labeling — samples with affinity <= this value are marked as binding for stratified cross-validation.")
parser.add_argument("--results_path", type=str, required=True, help="Path to store validation summary data.")

def cross_validation(k: int, df: pd.DataFrame, shuffle=True, random_state=None) -> list[(pd.DataFrame, pd.DataFrame)]:
    """
    Performs k-fold cross-validation on the given DataFrame.

    Parameters:
    - k: int, number of folds
    - df: pandas DataFrame
    - shuffle: bool, whether to shuffle data before splitting
    - random_state: int, random seed for reproducibility

    Returns:
    - folds: List of size k. Each element is a tuple (train_df, validation_df)
    """
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = len(df)
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i != k - 1 else n  # last fold takes any remainder
        val_df = df.iloc[start:end]
        train_df = pd.concat([df.iloc[:start], df.iloc[end:]]).reset_index(drop=True)
        folds.append((train_df, val_df))

    return folds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

df = pd.read_csv(args.data_path)

# Drop Affinities greater than threshold
threshold = args.a_thres
df = df[df['Affinity']<= threshold]

# Drop targets with 10 or less entires
value_counts = df["Target"].value_counts()
values_to_drop = value_counts[value_counts < 11].index
df = df[~df['Target'].isin(values_to_drop)]

bind_threshold = args.b_thres
stratify_threshold = args.s_thres
df['Binds'] = df['Affinity'] <= stratify_threshold

# Remove Fentanyl for being an unbalaced Class
df = df[~df["Target"].eq("Fentanyl")]

if __name__ == '__main__':    
    # allows for multiprocessing
    freeze_support()

    le_target = preprocessing.LabelEncoder()
    df.Target = le_target.fit_transform(df.Target.values)
    df['Classes'] = df['Target'].astype(str) + df['Binds'].astype(str)

    batch_size = 10

    # cross validation setup

    x = df.drop(columns=['Affinity'])
    y = df['Affinity']

    # Bin affinity into quantiles (say 20 bins for smoother averages).
    # Binning is a workaround to allow stratified sampling on a continuous variable.
    # The goal is to ensure that each fold has a similar distribution of affinity values.
    y_binned = pd.qcut(y, q=20, labels=False, duplicates="drop")

    num_folds = 10 
    train_dataset_size = len(df) * (num_folds - 1) / num_folds

    log_progress_step = ceil(train_dataset_size / batch_size / 5)  # Log progress 5 times per epoch 
    
    #skf is a StratifiedKFold object. StratifiedKFold is a class that allows us to split data into train and test sets. 
    skf = StratifiedKFold(n_splits = num_folds, random_state=None, shuffle=True,) 

    # creates a of tuples of (train_df, validation_df) for each fold.
    folds = [
        (df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True))
        for train_idx, val_idx in skf.split(x, y_binned)
        ]  

    # Training loop and logging setup

    # this array will keep track of the mean absolute error for each fold. Useful for summary statistics.
    mae_per_fold = np.zeros(num_folds)
    # this array will keep track of the accuracy for each fold. Useful for summary statistics.
    acc_per_fold = np.zeros(num_folds)   

    loss_func = nn.MSELoss()

    epochs = 200
    early_stopping = 10

    # Training loop
    print(f"Training on {train_dataset_size} samples...")

    fold_num = 0
    for (df_train, df_val) in folds:
        print("Initializing Recommender on fold ", fold_num)
        # load weights from checkpoint. Need to reload weights before every fold
        recommend_model = recommender.MolecularRecommender(ckpt_path=args.ckpt_path, embedding_size=args.embedding_size, num_layers=args.num_layers).to(device)
        print("Recommender Initialized.")

        losses = []
        acc_history = []   # this list will keep a running list of accuracies for each epoch. It will be averaged later for each fold. Useful for summary statistics.
        total_loss = 0
        stopping_counter = 0
        best_val_mae = float('inf')
        optimizer = torch.optim.Adam(recommend_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        sched = CosineWithRestarts(optimizer, T_max= 10) #T_max =  number of epochs before learning rate resets.

        train_dataset = recommender.MolecularDataset(
            targets=df_train.Target_SMILES.values,
            nucleotides=df_train.Seq_SMILES.values,
            affinities=df_train.Affinity.values
        )
        

        valid_dataset = recommender.MolecularDataset(
            targets=df_val.Target_SMILES.values,
            nucleotides=df_val.Seq_SMILES.values,
            affinities=df_val.Affinity.values
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=recommender.targets_nucleotides_batch
        )
        val_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=recommender.targets_nucleotides_batch
        )

        for e in range(epochs):

            recommend_model.train() # set to training mode.
            step_count = 0  # Reset step count at the beginning of each epoch
            start = time.time()
            for i, train_data in enumerate(train_loader):
                targets = train_data[0].to(device)
                nucleotides = train_data[1].to(device)

                output = recommend_model(targets, nucleotides)
                # Reshape the model output to match the target's shape
                output = output.squeeze()  # Removes the singleton dimension

                ratings = train_data[2].to(torch.float32).squeeze().to(device)  # Assuming ratings is already 1D

                loss = loss_func(output, ratings)
                total_loss += loss.sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Increment step count by the actual size of the batch
                step_count += len(train_data[1])

                # Check if it's time to log progress
                if (step_count % log_progress_step == 0 or i == len(train_loader) - 1):  # Log at the end of each epoch
                    recommender.log_progress(
                        e, fold_num, step_count, total_loss, log_progress_step, train_dataset_size, losses, epochs
                    )
                    total_loss = 0
            
            y_pred = []
            y_true = []
            acc_y_pred = []
            acc_y_true = []
            base_y_pred = []

            recommend_model.eval()

            with torch.no_grad():
                for i, valid_data in enumerate(val_loader):
                    output = recommend_model(
                        valid_data[0].to(device), valid_data[1].to(device)
                    )
                    ratings = valid_data[2].to(device)
                    bool_output = output <= bind_threshold
                    bool_ratings = ratings <= bind_threshold
                    y_pred.extend(output.cpu().detach().numpy())
                    acc_y_pred.extend(bool_output.cpu().detach().numpy())
                    y_true.extend(ratings.cpu().detach().numpy())
                    acc_y_true.extend(bool_ratings.cpu().detach().numpy())

            # Calculate Validation Metrics
            mae = mean_absolute_error(y_true, y_pred)
            acc = accuracy_score(acc_y_true, acc_y_pred)
            acc_history.append(acc)    
            avg_fold_acc = np.mean(acc_history)    # update the accuracy for this fold. Going to be printed in the summary table.
            print(f"epoch: {e+1}, Val MAE: {mae:.2f}, Acc: {acc:.2f}")


            # Stop training early if the absolute error does not do better than best_val_mae in 10 iterations.
            if mae <= best_val_mae:
                best_val_mae = mae
                checkpoint_name = f"{args.ckpt_save}"
                torch.save({
                            'epoch': e,
                            'state_dict': recommend_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr': optimizer.param_groups[0]['lr']
                            }, checkpoint_name)
                stopping_counter = 0
            else:
                stopping_counter += 1
            
            if stopping_counter >= early_stopping:
                print("Stopping fold early.")
                break

            previous_val_mae = mae

        mae_per_fold[fold_num] = best_val_mae    # update the mean absolute error for this fold. Going to be printed in the summary table   
        acc_per_fold[fold_num] = avg_fold_acc
        fold_num += 1
        sched.step()

    # Calculate average loss between all the folds. 
    print("Training complete. Summary statistics:")
    print("folds: ", num_folds)
    print("epochs per fold: ", epochs)
    print( "       |best MAE|  ACC  |")
    for i in range(num_folds):
        print(f"Fold {i} | {mae_per_fold[i]:.2f} | {acc_per_fold[i]:.2f}  |")
    print(f"Avg    | {np.mean(mae_per_fold):.2f} | {np.mean(acc_per_fold):.2f}  |")

    with open(args.results_path, "a") as file_results:
        file_results.write(f"\nValidation results ({datetime.now()})\n")
        file_results.write(f"folds:  {num_folds}\n")
        file_results.write(f"train set size:  {train_dataset_size}\n")
        file_results.write(f"epochs per fold: {epochs}\n")
        file_results.write("       |best MAE|  ACC  |\n")
        for i in range(num_folds):
            file_results.write(f"Fold {i} | {mae_per_fold[i]:.2f} | {acc_per_fold[i]:.2f}  |\n")
        file_results.write(f"Avg    | {np.mean(mae_per_fold):.2f} | {np.mean(acc_per_fold):.2f}  |\n")