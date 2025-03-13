import pandas as pd
import re

'''
This file will separate the CSV data into 3 arrays: subject, body, label
subject[0], body[0], label[0] will correspond to the first email
This file will then have all the data for the Word2Vec to train from
'''
def specGone(sentence):
    return re.sub(r'[^a-zA-Z.!?@\s]', '', sentence).replace("\n", " ")




def removeSpace(sentence):
    return re.sub(r'\s+', ' ', sentence)


df = pd.read_csv("Enron.csv")
print(df.get("body"))
subjects = df.get("subject")
bodies = df.get("body")

bodyArray = []
subjectArray = []

for i in range(0, len(bodies)):
    bodyArray.append(removeSpace(specGone(bodies[i])))
   # bodyArray.append(bodies[i])

for i in range(0, len(subjects)):
    subjectArray.append(removeSpace(specGone(str(subjects[i]))))

bodyAsSentences = [] # List of Email Lists of Sentence Strings

for body in bodyArray:
    bodyAsSentences.append(re.split(r'[.!?]', body))

bodyAsWords = [] # List of Email Lists of Sentence Lists of Word Strings

for i, email in enumerate(bodyAsSentences):  # for email in list
    bodyAsWords.append([]) # Append an empty email
    for j, sentence in enumerate(email):  # for sentence in email
        bodyAsWords[i].append(sentence.split())  # Breaks up sentence string into list of words
