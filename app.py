from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from knowledgeGraph import process_file
import json
from datetime import datetime
import logging
from json_validator import validate_json_file, fix_json_content
from knowledge_graph_builder import KnowledgeGraphBuilder
from chat import ChatManager  # Updated import

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Verify HuggingFace token
if not os.getenv('HUGGINGFACE_TOKEN'):
    logger.warning("HUGGINGFACE_TOKEN not found in environment variables")

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

# Add OUTPUT_FOLDER configuration
OUTPUT_FOLDER = 'processed_files'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Add configuration for knowledge graph
GRAPH_OUTPUT_DIR = 'static/graph'
if not os.path.exists(GRAPH_OUTPUT_DIR):
    os.makedirs(GRAPH_OUTPUT_DIR)

# Initialize knowledge graph builder
kg_builder = KnowledgeGraphBuilder(
    processed_files_dir=OUTPUT_FOLDER,
    output_dir=GRAPH_OUTPUT_DIR
)

# Initialize chat manager with knowledge graph
chat_manager = ChatManager(kg_builder)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return f"""Error loading template. Please ensure:
        1. The templates directory exists at {TEMPLATE_DIR}
        2. index.html is present in the templates directory
        Error: {str(e)}"""

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
            
            # Add timestamp to be included in processing
            timestamp = datetime.now().isoformat()
            # Process the file using knowledgeGraph with timestamp
            result = process_file(filepath, OUTPUT_FOLDER, timestamp)
            results.append({
                'filename': filename,
                'status': result,
                'timestamp': timestamp
            })
            # Clean up uploaded file
            os.remove(filepath)
    
    try:    
        # Rebuild knowledge graph after new files are processed
        kg_builder.build_graph()
        kg_builder.visualize_graph()
    except Exception as e:
        logger.error(f"Error updating knowledge graph: {str(e)}")
    
    return jsonify(results)

@app.route('/files', methods=['GET'])
def list_files():
    try:
        if not os.path.exists(OUTPUT_FOLDER):
            logger.debug(f"Output folder does not exist: {OUTPUT_FOLDER}")
            return jsonify([])
            
        json_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.json')]
        logger.debug(f"Found JSON files: {json_files}")
        
        files_data = []
        for filename in json_files:
            try:
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                logger.debug(f"Reading file: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                is_valid, _ = validate_json_file(file_path)
                files_data.append({
                    'filename': data.get('filename'),
                    'file_type': data.get('file_type'),
                    'json_path': filename,
                    'timestamp': data.get('timestamp'),
                    'is_valid': is_valid,
                })
                logger.debug(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error reading {filename}: {str(e)}")
        
        logger.debug(f"Returning files_data: {files_data}")
        return jsonify(files_data)
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        return jsonify([])

@app.route('/processed/<path:filename>')
def serve_json(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Try to fix any JSON issues before serving
            try:
                json_data = json.loads(content)
                fixed_content = fix_json_content(json_data)
                return app.response_class(
                    response=fixed_content,
                    status=200,
                    mimetype='application/json'
                )
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw content
                return jsonify({'error': 'Invalid JSON content'}), 400
    except Exception as e:
        logger.error(f"Error serving JSON file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/files/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'status': 'success', 'message': f'File {filename} deleted'})
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/validate/<path:filename>')
def validate_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
            
        is_valid, error = validate_json_file(file_path)
        if is_valid:
            return jsonify({'status': 'success', 'message': 'File structure is valid'})
        else:
            return jsonify({'status': 'error', 'message': error}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/graph/build', methods=['POST'])
def build_graph():
    try:
        kg_builder.build_graph()
        kg_builder.visualize_graph()
        return jsonify({'status': 'success', 'message': 'Knowledge graph built successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/graph/query', methods=['POST'])
def query_graph():
    request_data = request.get_json()
    if not request_data:
        return jsonify({'error': 'No request data provided'}), 400
        
    query = request_data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        results = kg_builder.query_graph(query)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/graph/view')
def view_graph():
    return send_from_directory('static/graph', 'knowledge_graph.html')

@app.route('/chat')
def chat_interface():
    return render_template('chat.html')

@app.route('/chat/session', methods=['POST'])
def create_chat_session():
    session_id = chat_manager.create_session()
    return jsonify({'session_id': session_id})

@app.route('/chat/message', methods=['POST'])
def chat_message():
    request_data = request.get_json()
    if not request_data:
        return jsonify({'error': 'No request data provided'}), 400
        
    session_id = request_data.get('session_id')
    message = request_data.get('message')
    
    if not session_id or not message:
        return jsonify({'error': 'Missing session_id or message'}), 400
    
    session = chat_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Invalid or expired session'}), 404
    
    # Add user message to history
    session.add_message("user", message)
    
    # Generate response
    response = session.generate_response(message)
    
    # Add assistant response to history
    session.add_message("assistant", response)
    
    return jsonify({
        'response': response,
        'session_id': session_id
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)