import numpy as np
import DataFormatter as df




def load_custom_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()  # Split by spaces
            word = parts[0]  # The word itself
            vector = list(map(float, parts[1:]))  # Convert remaining parts to floats (embedding)
            embeddings[word] = vector
    return embeddings



def text_to_embedding(word_to_vec,emails):
    embeddings = []
    labels = []
    for email in emails:
        labels.append(email[-1])
        words = email[:-1].split()
        email_embeddings = []
        for word in words:
            if word in word_to_vec:
                email_embeddings.append(word_to_vec[word])
            else:
                email_embeddings.append([0.0] * len(next(iter(word_to_vec.values()))))
        embeddings.append(email_embeddings)
    return embeddings,labels

def inputToEmbedding(word_to_vec,input):
    inputArr = input.strip().split()
    embeddings = []
    for word in inputArr:
        if word in word_to_vec:
            embeddings.append(word_to_vec[word])
        else:
            embeddings.append([0.0] * len(next(iter(word_to_vec.values()))))
    return embeddings


def embedEmails(embeddings):
    oneEmbedding = []

    for email in embeddings:
        if len(email) == 0:
            oneEmbedding.append(np.zeros(300))  # Return a zero vector if no embeddings
        else:
            oneEmbedding.append(np.mean(email, axis=0))
    return oneEmbedding

# Example
def embedEmail(embeddings):
    if len(embeddings) == 0:
        return np.zeros(300)  # Return a zero vector if no embeddings
    else:
        return np.mean(embeddings, axis=0)

def getTrainingData():
    embeddings_path = 'word_embeddings.txt'
    word_to_vec = load_custom_embeddings(embeddings_path)
    emailsShuffled,emailsOriginal = df.getAllEmails()
    embeddings, labels = text_to_embedding(word_to_vec, emailsShuffled)
    email_embeddingsShuffled = embedEmails(embeddings)
    embeddings, labelsNotUse = text_to_embedding(word_to_vec, emailsOriginal)
    email_embeddings = embedEmails(embeddings)
    return email_embeddingsShuffled,email_embeddings,labels

def createEmbeddingFromEmail(email):
    embeddings_path = 'word_embeddings.txt'
    word_to_vec = load_custom_embeddings(embeddings_path)
    embedding = embedEmail(inputToEmbedding(word_to_vec,email))
    return embedding