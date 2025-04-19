import random
import numpy as np
import DataFormatter as df
import cupy as cp
print(cp.__version__)
print(cp.cuda.runtime.getDeviceCount())

emails = df.get_cleaned_dataset()

vocab = list(set(word for email in emails for sentence in email for word in sentence))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

def generate_training_data(word2idx, window_size=3, neg_samples=2):
    '''
    Generates training data for word2vec
    
    Args:
        word2idx (dict): a dictionary that gives a key to each word
        window_size(int): the size of the nidow used in skipgram
        neg_samples(int): the amount of words in a skipgram that are words that don't associate
        with the target word
    '''
    training_data = []
    for email in emails:
        for sentence in email:
            for i,target_word in enumerate(sentence):
                target_idx = word2idx[target_word]

                start = max(0,i - window_size)
                end = min(len(sentence),i + window_size + 1)

                context_words = [sentence[j] for j in range(start, end) if j != i]

                for context_word in context_words:
                    context_idx = word2idx[context_word]
                    training_data.append((target_idx, context_idx, 1))

                for negative in range(neg_samples):
                    # Choose a random word for the negative sample
                    neg_word = random.choice(vocab)
                    while neg_word in context_words:  # Avoid choosing context words
                        neg_word = random.choice(vocab)
                    neg_idx = word2idx[neg_word]
                    training_data.append((target_idx, neg_idx, 0))  # Negative sample


    return training_data

training_data = generate_training_data(emails,word2idx)

class SkipGram:
    '''
    Self made skipgram class that preforms the skipgram algorithms
    '''
    def __init__(self,vocab_size,embedding_dim=300,learning_rate=0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.W1 = cp.random.randn(vocab_size, embedding_dim).astype(cp.float32)
        self.W2 = cp.random.randn(vocab_size, embedding_dim).astype(cp.float32)

    def softmax(self,x):
        
        exp_x = cp.exp(x - cp.max(x))
        return exp_x / cp.sum(exp_x, axis=0)

    def forward(self,target_idx):
        h = self.W1[target_idx]
        u = cp.dot(h, self.W2)
        y_pred = self.softmax(u)
        return y_pred, h

    def backprop(self,target_idx,context_idx,y_pred,h):
        y_true = cp.zeros(self.vocab_size)
        y_true[context_idx] = 1

        error = y_pred - y_true

        self.W2 -= self.learning_rate * cp.outer(h, error)
        self.W1[target_idx] -= self.learning_rate * cp.dot(self.W2, error)

        
    # Now train Neural network using window array and target word


    def train(self, training_data, batch_size=1024, epochs=10):
        total = len(training_data)

        for epoch in range(epochs):
            for i in range(0, total, batch_size):
                batch = training_data[i:i + batch_size]

                # Exctracts target and context
                target_batch = cp.array([pair[0] for pair in batch], dtype=cp.int32)
                context_batch = cp.array([pair[1] for pair in batch], dtype=cp.int32)

                current_batch_size = len(batch)
                v = self.W1[target_batch]
                y_pred_batch = cp.matmul(v, self.W2.T)

                # One-hot true labels
                y_true_batch = cp.zeros_like(y_pred_batch)
                for j in range(current_batch_size):
                    if context_batch[j] < self.vocab_size:
                        y_true_batch[j, context_batch[j]] = 1

                # Error and gradients
                y_error_batch = y_pred_batch - y_true_batch
                dW2 = cp.matmul(y_error_batch.T, v)
                dW1 = cp.matmul(y_error_batch, self.W2)
                max_grad_value = 5.0
                dW1 = cp.clip(dW1, -max_grad_value, max_grad_value)
                dW2 = cp.clip(dW2, -max_grad_value, max_grad_value)

                # Updating weights
                self.W1[target_batch] -= self.learning_rate * dW1
                self.W2 -= self.learning_rate * dW2

                # Show progress for the current epoch
                progress = (i / total) * 100
                print(f"Epoch {epoch + 1}/{epochs} - Progress: {i} / {total} ({progress:.2f}%)")
        self.save_embeddings()

    def save_embeddings(self):
        word_embeddings = cp.asnumpy(self.W1)
        with open("word_embeddings.txt", "w") as f:
            for word, idx in word2idx.items():
                embedding = " ".join(map(str, word_embeddings[idx]))  
                f.write(f"{word} {embedding}\n")


skipgram = SkipGram(vocab_size=len(vocab))
skipgram.train(training_data)






