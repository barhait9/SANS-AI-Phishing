import numpy as np
from DataHandle.DataFormatter import getAllEmails
from PathResolver import path_resolver as pr

def load_custom_embeddings(file_path):
    '''
    Loads embeddings to a variable

    Args:
        file_path (str): the path of the file the embeddings are stored on
    '''
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
    return embeddings



def text_to_embedding(word_to_vec,emails):
    '''
    Converts a bunch of emails to embeddings into an array used in training

    Args:
        word_to_vec (embeddings): a loaded set of word embeddings loaded in with a function
        emails (array(strings)): An array of emails with a number as the last character 
            which is the label of the email determining if it's spam or not
    '''
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
    '''
    Will convert a singular email into embeddings

    Args:
        word_to_vec (embeddings): a loaded set of word embeddings loaded in with a function
        input (str): a string which is the body of the email
    '''
    inputArr = input.strip().split()
    embeddings = []
    for word in inputArr:
        if word in word_to_vec:
            embeddings.append(word_to_vec[word])
        else:
            embeddings.append([0.0] * len(next(iter(word_to_vec.values()))))
    return embeddings


def embedEmails(embeddings):
    '''
    Converts all word embeddings in an array of  emails to one vector

    Args:
        embeddings (array): an array of an email's embeddings consisting of all the embeddings in an email
    
    '''
    oneEmbedding = []

    for email in embeddings:
        if len(email) == 0:
            oneEmbedding.append(np.zeros(300))  # Return a zero vector if no embeddings
        else:
            oneEmbedding.append(np.mean(email, axis=0))
    return oneEmbedding

def embedEmail(embeddings):
    '''
    Converts all word embeddings in an array of  emails to one vector

    Args:
        embeddings (array): an array of embeddings consisting of all the embeddings in an email
    
    '''
    if len(embeddings) == 0:
        return np.zeros(300)  # Return a zero vector if no embeddings
    else:
        return np.mean(embeddings, axis=0)

def getTrainingData():
    '''
    Generates training data and converts to word embeddings for 
    the email AI to train on using the dataset 
    '''
    embeddings_path = 'backend/DataHandle/word_embeddings.txt'
    word_to_vec = load_custom_embeddings(embeddings_path)
    emailsShuffled,emailsOriginal = getAllEmails()
    embeddings, labels = text_to_embedding(word_to_vec, emailsShuffled)
    email_embeddingsShuffled = embedEmails(embeddings)
    embeddings, labelsNotUse = text_to_embedding(word_to_vec, emailsOriginal)
    email_embeddings = embedEmails(embeddings)
    return email_embeddingsShuffled,email_embeddings,labels

def createEmbeddingFromEmail(email):
    '''
    Creates one embedding for the email you want to classify i.e email you want to test for spam
    '''
    
    embeddings_path = pr.resolve("word_embeddings.txt")
    word_to_vec = load_custom_embeddings(embeddings_path)
    embedding = embedEmail(inputToEmbedding(word_to_vec,email))
    return embedding