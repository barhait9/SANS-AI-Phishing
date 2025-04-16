import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import DataFormatter as df

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sample dataset (Replace with actual preprocessed dataset)
emails = df.get_cleaned_dataset()

# Vocabulary
vocab = list(set(word for email in emails for sentence in email for word in sentence))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}
vocab_size = len(vocab)


# Generate Training Data (Skip-Gram with Negative Sampling)
def generate_training_data(emails_words, word2idx, window_size=3, neg_samples=2):
    training_data = []
    for sentence in emails_words:
        for i, target_word in enumerate(sentence):
            target_idx = word2idx[target_word]

            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            context_words = [sentence[j] for j in range(start, end) if j != i]

            for context_word in context_words:
                context_idx = word2idx[context_word]
                training_data.append((target_idx, context_idx, 1))

                for _ in range(neg_samples):
                    neg_word = random.choice(vocab)
                    while neg_word in context_words:
                        neg_word = random.choice(vocab)
                    neg_idx = word2idx[neg_word]
                    training_data.append((target_idx, neg_idx, 0))
    return training_data


training_data = generate_training_data(emails, word2idx)


# Convert to PyTorch Tensor
def to_tensor(pairs):
    return torch.tensor(pairs, dtype=torch.long, device=device)


targets, contexts, labels = zip(*training_data)
targets, contexts, labels = to_tensor(targets), to_tensor(contexts), to_tensor(labels, dtype=torch.float)

dataset = torch.utils.data.TensorDataset(targets, contexts, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)


# Skip-Gram Model with PyTorch (Momentum Optimized)
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_idx):
        embed = self.embeddings(target_idx)
        output = self.output_layer(embed)
        return output


# Initialize Model and Optimizer
embedding_dim = 300
model = SkipGram(vocab_size, embedding_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Added momentum


# Training Loop
def train(model, dataloader, epochs=1):
    model.train()
    progress = 0
    for epoch in range(epochs):
        total_loss = 0
        for target_idx, context_idx, label in dataloader:
            optimizer.zero_grad()
            output = model(target_idx)

            # Extract only the relevant context words for loss computation
            logits = output[torch.arange(output.shape[0]), context_idx]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            progress += 1
            print((progress / len(training_data))*100,"%")
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


train(model, dataloader)  # Train with Momentum

# Print Word Embeddings
print("\nWord Embeddings:")
for word, idx in word2idx.items():
    print(word, model.embeddings.weight[idx].detach().cpu().numpy())
