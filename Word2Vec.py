import random
import numpy as np
import DataFormatter as df


body = "hello this will be a test of the email today semantic analysis of words for skipgram this is needed because why not type type"
emails = df.get_cleaned_dataset()

words = body.split()
vocab = list(set(word for email in emails for sentence in email for word in sentence)) # only shows unique words
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

def generate_training_data(emails_words, word2idx, window_size=3, neg_samples=0):
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


    '''
    for i, target_word in enumerate(words):
        target_idx = word2idx[target_word]
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_word = words[j]
                context_idx = word2idx[context_word]
                training_data.append((target_idx, context_idx))
                '''
    return training_data

training_data = generate_training_data(emails,word2idx)

class SkipGram:
    def __init__(self,vocab_size,embedding_dim=300,learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.W1 = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-1, 1, (embedding_dim, vocab_size))
    def softmax(self,x):
        exp_x = np.exp(x - max(x))
        return exp_x / exp_x.sum(axis=0)

    def forward(self,target_idx):
        h = self.W1[target_idx]
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        return y_pred, h

    def backprop(self,target_idx,context_idx,y_pred,h):
        y_true = np.zeros(self.vocab_size)
        y_true[context_idx] = 1

        error = y_pred - y_true

        self.W2 -= self.learning_rate * np.outer(h, error)
        self.W1[target_idx] -= self.learning_rate * np.dot(self.W2, error)

        
    # Now train Neural network using window array and target word

    def train(self,training_data,epochs=1):
        for epoch in range(epochs):
            total_loss = 0
            progress = 0
            for target_idx, context_idx, _ in training_data:
                y_pred, h = self.forward(target_idx)
                self.backprop(target_idx,context_idx,y_pred,h)
                progress += 1
                print(((progress / 5556) * 100),"%")
                total_loss += -np.log(y_pred[context_idx])
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
print(f"Sample training data: {training_data[:5]}")
print(f"Vocab size:{len(vocab)}")
print("num of pairs below")
print(len(training_data))
skipgram = SkipGram(vocab_size=len(vocab))
skipgram.train(training_data)

print("\nWord Embeddings:")
for word, idx in word2idx.items():
    print(word, skipgram.W1[idx])





