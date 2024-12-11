import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from main import extract_articles, generate_results  # Import functions from your main.py

app = Flask(__name__)

# Folder for file uploads
UPLOAD_FOLDER = 'static/uploads'
ARTICLES_FOLDER = 'Articles'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTICLES_FOLDER, exist_ok=True)

# Allowed extensions for the uploaded file
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload and trigger processing
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the file part exists
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If no file selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # If the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Step 1: Extract articles (if not already done)
        extract_articles(filepath)

        # Step 2: Process the extracted data and generate results
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
        generate_results(filepath, output_file)

        # Send back the output file
        return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path='output.xlsx', as_attachment=True)

    # If file format is invalid
    return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
