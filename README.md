# Flask AI Application

This project is a Flask-based web application designed to serve image embeddings and filenames for AI-related tasks. It utilizes transfer learning with a pre-trained ResNet50 model to extract features from images and provides endpoints to access these features.

## Project Structure

```
flask-ai-app
├── app
│   ├── __init__.py          # Initializes the Flask application
│   ├── routes.py            # Contains route definitions for the app
│   ├── static               # Directory for static files (CSS, JS, images)
│   └── templates            # Directory for HTML templates
├── data
│   ├── features.pkl         # Serialized embeddings extracted from images
│   └── filenames.pkl        # List of filenames corresponding to the embeddings
├── requirements.txt         # Lists dependencies required for the app
└── README.md                # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd flask-ai-app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   flask run
   ```

## Usage

- The application will be accessible at `http://127.0.0.1:5000`.
- Endpoints are available to retrieve the `features.pkl` and `filenames.pkl` files.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.