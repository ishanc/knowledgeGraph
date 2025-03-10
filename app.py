from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from knowledgeGraph import process_file

# Create templates directory if it doesn't exist
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

app = Flask(__name__, 
    template_folder=TEMPLATE_DIR,
    static_url_path='',
    static_folder='static'
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error rendering template: {str(e)}")
        return f"""
        Error loading template. Please ensure:
        1. The templates directory exists at {TEMPLATE_DIR}
        2. index.html is present in the templates directory
        Error: {str(e)}
        """

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    results = []
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file using knowledgeGraph
            result = process_file(filepath)
            results.append({
                'filename': filename,
                'status': result
            })
            
            # Clean up uploaded file
            os.remove(filepath)
    
    return jsonify(results)

if __name__ == '__main__':
    print(f"Template directory: {TEMPLATE_DIR}")
    print(f"Template exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html'))}")
    app.run(debug=True, port=5000)
