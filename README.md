# SANS-AI-Phishing

SANS-AI-Phishing is a Google Chrome extension designed to detect potentially spam emails using AI and machine learning. This project leverages a Flask API deployed in a Docker container on Google Cloud Platform (GCP) via Cloud Run. It implements Word2Vec and AI models built entirely without external libraries, making it a lightweight and unique solution.

## Features
- **Spam Detection**: Analyze emails to detect potential spam.
- **Google Chrome Extension**: Easily integrates with Gmail in your browser.
- **Flask API**: A simple and efficient API for communication.
- **Custom Word2Vec Implementation**: Built from scratch without external libraries.
- **AI Model Without Libraries**: A fully custom AI model for email analysis.
- **GCP Cloud Run**: Deployed using Docker and Cloud Run for seamless scalability.

## Installation
1. Download the contents of the `extension` folder from this repository.
2. Open your browser's **Manage Extensions** page.
3. Enable **Developer Mode**.
4. Drag and drop the downloaded `extension` folder onto the Extensions page.
5. The extension should be installed successfully.

## Usage
1. Open Gmail in the current tab of your browser.
2. Interact with the extension by pressing the designated button.
3. The AI will analyze the emails in the Gmail tab and provide insights on potential spam risks.

## Dependencies
There are no dependencies required for running this project locally since it operates entirely off of GCP Cloud Run. Anyone with access to the deployed API can utilize it without additional setup.

## Contributions
We welcome contributions to improve the project! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description of your contributions.

## License
This project is licensed under the [MIT License](LICENSE).

## Support
If you encounter any issues or have questions, feel free to open an issue in the repository.

---

Happy emailing, and stay safe from spam!