from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'tmp/uploads'  # Change this to the directory where you want to save images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    raise Exception(f'The specified upload folder does not exist: {UPLOAD_FOLDER}')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the 'image' part
    if 'image' not in request.files:
        return 'No image part in the request', 400

    file = request.files['image']

    # If no file is selected
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save the file to the specified folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'File successfully uploaded', 200
    else:
        return 'Invalid file type', 400

if __name__ == '__main__':
    app.run(debug=True)
