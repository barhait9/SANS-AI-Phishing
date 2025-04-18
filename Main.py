from EmailNetwork import EmailNetwork
import numpy as np


if __name__ == "__main__":
    email_net = EmailNetwork()
    X_train = np.random.randn(100, 300)  # 100 fake email embeddings
    y_train = np.random.randint(0, 2, 100)  # Random labels (0 or 1)

    print("Training Email Spam Classifier...")
    email_net.train(X_train, y_train, epochs=30, learning_rate=0.01)

    test_embedding = np.random.randn(300)  # Simulated email embedding
    #print("Spam" if email_net.classify(test_embedding) else "Not Spam")
    print(email_net.classify(test_embedding))


