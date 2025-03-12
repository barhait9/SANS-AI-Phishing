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

# NEED TO REMOVE EMPTY SENTENCES
# OR NEXT PART WON'T WORK

# Trying to join email terminal words to previous sentence
grabWordOfNextList = False # To keep track of if sentence ends in an email over sentence iterations

for i, email in enumerate(bodyAsWords):  # for email in list
    for j, sentence in enumerate(email):  # for sentence in email
        if grabWordOfNextList:
            #print("PREVIOUS SENTENCE: ", bodyAsWords[i][j-1])
            #print("AFTER SENTENCE: ",bodyAsWords[i][j])
            bodyAsWords[i][j-1].append(bodyAsWords[i][j].pop(0)) # moves first word in current sentence to previous sentence

            if len(bodyAsWords[i][j]) == 0:
                pass
                #del bodyAsWords[i][j] # delete current sentence array, (there was only one email terminal in it)
                #instead of deleting here, save indexes to delete after all for loops have finished
            else:
                grabWordOfNextList = False # start of new sentence so stop moving email terminal words

        if len(bodyAsWords[i][j]) > 1:
            if bodyAsWords[i][j][-2] == "@":
                grabWordOfNextList = True


print(bodyAsWords[1][1][-2])

#print(bodyAsSentences[1][0])
