import random
import pandas as pd

df = pd.read_csv("dataset.csv")

subjects = df.get("Subject")
bodies = df.get("Body")
labels = df.get("Label")

# List of numbers from 0 to (length of emails -1)
indexes = [i for i in range(0, len(bodies))]
shuffledIndexes = []

# Shuffle list of indexes
while len(indexes) > 0:
    random_index = random.randint(0,len(indexes)-1)
    shuffledIndexes.append(indexes.pop(random_index))

# Reorder subjects, bodies and labels to follow shuffled index order

shuffledSubjects = []
shuffledBodies = []
shuffledLabels = []

for i in shuffledIndexes:
    shuffledSubjects.append(subjects[i])
    shuffledBodies.append(bodies[i])
    shuffledLabels.append(int(labels[i]))

# Create new "DataFrame"? with shuffled data
shuffled_df = pd.DataFrame({
    'Subject': shuffledSubjects,
    'Body': shuffledBodies,
    'Label': shuffledLabels
})

# Write to shuffled CSV file
shuffled_df.to_csv('shuffled_dataset.csv', index=False)

