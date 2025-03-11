import pandas as pd
import re

'''
This file will separate the CSV data into 3 arrays: subject, body, label
subject[0], body[0], label[0] will correspond to the first email
This file will then have all the data for the Word2Vec to train from
'''
def specGone(sentence):
    return re.sub(r'[^a-zA-Z\s]', '', sentence).replace("\n", " ")

def removeSpace(sentence):
    return re.sub(r'\s+', ' ', sentence)


df = pd.read_csv("Enron.csv")
print(df.get("body"))
subjects = [[subject] for subject in df.get("subject")]
bodies = [[body] for body in df["body"]]

print(bodies)
cleanBodies = []

for body in bodies:
    emailBody = []
    for word in body:
        #word = specGone(word)
        word = removeSpace(word)
        emailBody.append(word)

    cleanBodies.append(emailBody)



print("CLEANED")
print(subjects[2519])
print(cleanBodies[2519])

