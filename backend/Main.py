from starlette.middleware.cors import CORSMiddleware
import uvicorn
from EmailAI.EmailNetwork import EmailNetwork
import numpy as np
from LoadingEmbeddings.loadEmbeddings import createEmbeddingFromEmail
import pickle
from fastapi import *
import logging
from PathResolver import path_resolver as pr

if __name__ == "__main__":

    #Original Code used to train the Email AI
    '''data,dataOriginal, labels = le.getTrainingData()
    trainingData = data[:4300]
    trainingLabels = labels[:4300]
    testingData = dataOriginal[4300:4423]
    email_net = EmailNetwork()
    X_train = np.random.randn(100, 300)  # 100 fake email embeddings
    y_train = np.array(trainingLabels, dtype=np.float64)


    print("Training Email Spam Classifier...")
    #email_net.train(trainingData, y_train, epochs=200, learning_rate=0.01)

    #save_network(email_net,"email_model.pkl")
    #email_net.save_weights("email_network1")
    email_net.load_weights("email_network1")
    #print("Spam" if email_net.classify(test_embedding) else "Not Spam")
    #print(test_embedding)
    for i in range(0,len(testingData)):
        print("Classify " + str(email_net.classify(testingData[i])))

'''


app = FastAPI()
@app.get("/")
def root():
    return {"Hello": "World"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify(request: Request):
    try:
        body = await request.body()
        email = body.decode("utf-8")
        print(f"Received email: {email}")
        email_embedding = createEmbeddingFromEmail(email)
        # Load model
        email_net = EmailNetwork()
        load_path = pr.resolve("email_network1")
        email_net.load_weights(load_path)

        #Classify Email
        result = email_net.classify(email_embedding)
        return {"classified": result}
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
