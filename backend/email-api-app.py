from flask import Flask, request, jsonify
from flask_cors import CORS
from LoadingEmbeddings.loadEmbeddings import createEmbeddingFromEmail 
from EmailAI.EmailNetwork import EmailNetwork
import os
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CORS(app, resources={r"/*": {"origins": ["https://mail.google.com"]}})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"Hello": "World"})

@app.route("/health", methods=["GET"])
def health():
    return "Healthy", 200

@app.route("/verify", methods=["POST", "OPTIONS"])
def verify():

    logger.info(f"Request method: {request.method}")
    logger.info("Request headers for /verify endpoint:")
    for header, value in request.headers.items():
        logger.info(f"{header}: {value}")

    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')

        logger.info("Preflight response headers:")
        for header, value in response.headers.items():
            logger.info(f"{header}: {value}")
        return response
        
    try:
        email = request.data.decode("utf-8")
        print(f"Received email: {email}")
        email_embedding = createEmbeddingFromEmail(email)

        # Load model
        email_net = EmailNetwork()
        load_path = "/app/DataHandle/email_network1"
        email_net.load_weights(load_path)

        # Classify Email
        result = email_net.classify(email_embedding)
        return jsonify({"classified": result})
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port) 