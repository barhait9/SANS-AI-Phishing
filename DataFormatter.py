import pandas as pd
import re

'''
This file will separate the CSV data into 3 arrays: subject, body, label
subject[0], body[0], label[0] will correspond to the first email
This file will then have all the data for the Word2Vec to train from
'''
def specGone(sentence):
    return re.sub(r'[^a-zA-Z.!?@\s]', '', sentence).replace("\n", " ").replace("�", "")

def removeSpace(sentence):
    return re.sub(r'\s+', ' ', sentence)


df = pd.read_csv("dataset.csv")
subjects = df.get("Subject")
bodies = df.get("Body")

bodyArray = []
subjectArray = []

for i in range(0, len(bodies)):
    bodyArray.append(removeSpace(specGone(str(bodies[i]))))
   # bodyArray.append(bodies[i])

for i in range(0, len(subjects)):
    subjectArray.append(removeSpace(specGone(str(subjects[i]))))

bodyAsSentences = [] # List of Email Lists of Sentence Strings

for i, body in enumerate(bodyArray):

    bodyAsSentences.append(re.split(r'[.!?]', body))

    # INCLUDE SUBJECTS ?
    # bodyAsSentences[i].insert(0, subjectArray[i]) # This line would insert subject data as the first sentence of the body



bodyAsWords = [] # List of Email Lists of Sentence Lists of Word Strings

for i, email in enumerate(bodyAsSentences):  # for email in list
    bodyAsWords.append([]) # Append an empty email
    for j, sentence in enumerate(email):  # for sentence in email
        bodyAsWords[i].append(sentence.split())  # Breaks up sentence string into list of words

# NEED TO REMOVE EMPTY SENTENCES
for i in range(len(bodyAsWords)-1, -1, -1): # iterating through list starting from end
    for j in range(len(bodyAsWords[i])-1, -1, -1):
        if not bodyAsWords[i][j]: # if sentence is empty
            del bodyAsWords[i][j]


# Joining sentences where (simple) singular full stop email domains were split up
for email in range(len(bodyAsWords)-1, -1, -1):  # for email index in list starting from end
    for sentence in range(len(bodyAsWords[email])-1, -1, -1):  # for sentence index in this email starting from end
        if len(bodyAsWords[email][sentence]) > 1: # If sentence has at least 2 words in it
            if bodyAsWords[email][sentence][-2] == "@" or "@" in bodyAsWords[email][sentence][-1]: # If sentence contains @ symbol at end
                if sentence != len(bodyAsWords[email])-1: # If there is a sentence after this sentence in the email

                    bodyAsWords[email][sentence].extend(bodyAsWords[email][sentence+1]) # Combines sentence containing @ with following sentence

                    del bodyAsWords[email][sentence+1] # Delete (now) duplicate following sentence

# NEED TO REMOVE 1 WORD SENTENCES
for i in range(len(bodyAsWords)-1, -1, -1): # iterating through list starting from end
    for j in range(len(bodyAsWords[i])-1, -1, -1):
        if len(bodyAsWords[i][j]) <= 1: # if sentence only has 1 word
            del bodyAsWords[i][j]

"""
 Sentence joiner code above doesn't look for multi domain endings like .co.uk, or fullstops in email names like bar.hait@gmail.com
 The only fullstops it detects (and fixes) are fullstops in simple email domains of the form name@mail.domain -> AKA singular full stop email domains
 Other than in these cases, any full stops that don't signify sentence endings will mess up the sentence splits.
 TBF this code does fix a lot of wrongly split sentence cases but also leaves many still unfixed...
"""

# Printing all sentences

def get_cleaned_dataset():
    return bodyAsWords


# bodyAsWords fully cleaned?
