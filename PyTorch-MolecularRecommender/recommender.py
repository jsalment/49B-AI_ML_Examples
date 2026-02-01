import torch
import torch.nn as nn

from torch.utils.data import Dataset
from transformer import Transformer, create_masks
from load_data import ALPHABET_SIZE, EXTRA_CHARS

def encode_char(c):
    return ord(c) - 32

def encode_smiles(string, start_char=EXTRA_CHARS['seq_start']):
    return torch.tensor([ord(start_char)] + [encode_char(c) for c in string], dtype=torch.long)[:1300].unsqueeze(0)

def targets_nucleotides_batch(instances): 
    # Must return an additional batch for each column you add to training data

    targets_lens = torch.tensor([s["targets"].shape[0] + 1 for s in instances], dtype=torch.long)
    nucleotides_lens = torch.tensor([s["nucleotides"].shape[0] + 1 for s in instances], dtype=torch.long)
    
    max_len_targets = targets_lens.max().item()
    max_len_nucleotides = nucleotides_lens.max().item()
    
    batch_targets = torch.full((len(instances), max_len_targets), ord(EXTRA_CHARS['pad']), dtype=torch.long)
    batch_nucleotides = torch.full((len(instances), max_len_nucleotides), ord(EXTRA_CHARS['pad']), dtype=torch.long)
    batch_affinities = torch.zeros((len(instances),1), dtype=torch.float)

    for i, instance in enumerate(instances):
        batch_targets[i, 0] = ord(EXTRA_CHARS['seq_start'])
        batch_targets[i, 1:targets_lens[i]] = instance['targets']

        batch_nucleotides[i, 0] = ord(EXTRA_CHARS['seq_start'])
        batch_nucleotides[i, 1:nucleotides_lens[i]] = instance["nucleotides"]

        batch_affinities[i] = instance["affinities"]
    
    return batch_targets, batch_nucleotides, batch_affinities

class MolecularDataset(Dataset):
    # Molecular data class. This class prepares the dataset for training and validation.

    def __init__(self, targets, nucleotides, affinities):
        # Initialize the dataset
        self.targets = targets
        self.nucleotides = nucleotides
        self.affinities = affinities
        # add new fields here to experiment with different data embeddings

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.nucleotides)
    
    def string_to_tensor(self, string):
        tensor = torch.tensor(list(map(encode_char, string)), dtype=torch.uint8)
               
        return tensor
    
    def __getitem__(self, item):
        # Retrieves a samle from the datasset at the specified index
        targets = self.targets[item]
        nucleotides = self.nucleotides[item]
        affinities = self.affinities[item]

        return {
            "targets": self.string_to_tensor(targets),
            "nucleotides": self.string_to_tensor(nucleotides),
            "affinities": torch.tensor(affinities, dtype=torch.float)
            # add additional entries for any other dimensions you add to training data
        }

class MolecularRecommender(nn.Module):
    def __init__(self, ckpt_path = "checkpoints/pretrained.ckpt", embedding_size=512, num_layers=6, hidden_dim=512, dropout_rate=.2):
        super(MolecularRecommender, self).__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Embedding Layers
        print("Loading pretrained weights from", ckpt_path)
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.target_embedding = Transformer(ALPHABET_SIZE, self.embedding_size, self.num_layers).eval()
        self.target_embedding = torch.nn.DataParallel(self.target_embedding)
        self.target_embedding.load_state_dict(checkpoint['state_dict'])
        for param in self.target_embedding.parameters():
            param.requires_grad = False 
        self.target_model = self.target_embedding.module
        self.target_encoder = self.target_embedding.module.encoder
        self.nucleotide_embedding = Transformer(ALPHABET_SIZE, self.embedding_size, self.num_layers).eval()
        self.nucleotide_embedding = torch.nn.DataParallel(self.nucleotide_embedding)
        """ for param in self.nucleotide_embedding.parameters():
            param.requires_grad = False """ #try uncommenting to freeze aptamer weights. Might improve results 
        self.nucleotide_embedding.load_state_dict(checkpoint['state_dict'])
        self.nucleotide_model = self.nucleotide_embedding.module
        self.nucleotide_encoder = self.nucleotide_embedding.module.encoder
        print("Pretrained weights loaded")

        # Hidden Layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim) #(input, output)
        self.fc2 = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.fc3 = nn.Linear(int(self.hidden_dim/2), int(self.hidden_dim/8))
        self.fc4 = nn.Linear(int(self.hidden_dim/8), 1) #last layer is size one because we want a single value prediction

        # Dropout Layer - randomly turns of some neurons across network
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, target, nucleotide):
        # Embeddings
        mask_target = create_masks(target)
        target_embedded = self.target_encoder(target, mask_target).mean(axis=1)

        mask_nucleotide = create_masks(nucleotide)
        nucleotide_embedded = self.nucleotide_encoder(nucleotide, mask_nucleotide).mean(axis=1)

        # Concatenate nucleotide and target embeddings
        combined = torch.cat([nucleotide_embedded,target_embedded], axis=1)

        # Pass through hidden layers with RelU and dropout
        x = self.dropout(self.relu(self.fc1(combined)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.fc4(x)

        return output

# Function to log progress
def log_progress(epoch, fold, step, total_loss, log_progress_step, data_size, losses, epochs):
    avg_loss = total_loss / log_progress_step
    print(f"fold: {fold} | epoch: {epoch+1:02d}/{epochs:02d} | Step: {step}/{data_size} | Avg Loss: {avg_loss:<6.9f}")
    losses.append(avg_loss)
