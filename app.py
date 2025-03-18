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
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from llm_utils import retry_llm_call, FORMAT_INSTRUCTIONS, EXAMPLE_RESPONSE

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Verify HuggingFace token
if not os.getenv('HUGGINGFACE_TOKEN'):
    logger.warning("HUGGINGFACE_TOKEN not found in environment variables")

# Configure retry strategy for HuggingFace API calls
retry_strategy = Retry(
    total=5,  # Increased from 3
    backoff_factor=0.5,  # Reduced from 1
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)

# Configure connection pooling
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # Increased connection pool
    pool_maxsize=10,
    pool_block=False
)

# Configure session with keep-alive
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)
http.headers.update({
    'Connection': 'keep-alive',
    'Keep-Alive': 'timeout=60, max=1000'
})

def call_huggingface_api(endpoint, data):
    def _make_request():
        response = http.post(
            endpoint, 
            json={
                **data,
                "format_instructions": FORMAT_INSTRUCTIONS,
                "example_response": EXAMPLE_RESPONSE
            },
            headers={
                "Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}",
                'Connection': 'keep-alive'
            },
            timeout=30
        )
        response.raise_for_status()
        return response.text
    
    return retry_llm_call(_make_request)

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
    
    try:
        # Add user message to history
        session.add_message("user", message)
        
        # Generate response with error handling
        response = session.generate_response(message)
        if not response:
            raise ValueError("Failed to generate response")
            
        # Add assistant response to history
        session.add_message("assistant", response)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error in chat message: {str(e)}")
        return jsonify({'error': 'Failed to process message'}), 500

# Add new routes for graph editing
@app.route('/graph/editor')
def graph_editor():
    return render_template('graph_editor.html')

@app.route('/graph/data')
def get_graph_data():
    nodes_data = []
    edges_data = []
    
    for node, data in kg_builder.nx_graph.nodes(data=True):
        nodes_data.append({
            'id': node,
            'label': node,
            'color': data.get('color', '#1f77b4'),
            'size': data.get('size', 20)
        })
    
    for source, target, data in kg_builder.nx_graph.edges(data=True):
        edges_data.append({
            'id': f"{source}-{target}",
            'from': source,
            'to': target,
            'label': data.get('type', 'related'),
            'width': data.get('weight', 1) * 2
        })
    
    return jsonify({
        'nodes': nodes_data,
        'edges': edges_data
    })

@app.route('/graph/node', methods=['POST'])
def add_node():
    data = request.get_json()
    try:
        node_data = {
            'type': data.get('type', 'concept'),
            'label': data['label'],
            'timestamp': datetime.now().isoformat(),
            'source': 'user_edit',
            'confidence': 1.0
        }
        kg_builder.nx_graph.add_node(data['label'], **kg_builder._prepare_node_data(node_data))
        kg_builder.visualize_graph()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/graph/node/<node_id>', methods=['PUT', 'DELETE'])
def modify_node(node_id):
    try:
        if request.method == 'DELETE':
            kg_builder.nx_graph.remove_node(node_id)
        else:
            data = request.get_json()
            kg_builder.nx_graph.nodes[node_id].update(kg_builder._prepare_node_data(data))
        
        kg_builder.visualize_graph()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/graph/relationship', methods=['POST'])
def add_relationship():
    data = request.get_json()
    try:
        edge_data = {
            'type': data['type'],
            'weight': 1.0,
            'timestamp': datetime.now().isoformat(),
            'source': 'user_edit',
            'confidence': 1.0
        }
        kg_builder.nx_graph.add_edge(
            data['source'], 
            data['target'], 
            **kg_builder._prepare_node_data(edge_data)
        )
        kg_builder.visualize_graph()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/graph/stats')
def get_graph_stats():
    try:
        nodes = list(kg_builder.nx_graph.nodes(data=True))  # Convert to list for accurate counting
        edges = kg_builder.nx_graph.edges(data=True)
        
        # Debug logging to inspect node types
        for node, data in nodes:
            logger.debug(f"Node {node} has type: {data.get('type', 'unknown')}")
            
        stats = {
            'nodes': len(nodes),
            'edges': len(edges),
            'topics': sum(1 for _, data in nodes if str(data.get('type', '')).lower() == 'topic'),
            'concepts': sum(1 for _, data in nodes if str(data.get('type', '')).lower() in ['concept', 'entity'])
        }
        
        logger.debug(f"Graph stats: {stats}")
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/huggingface', methods=['POST'])
def huggingface_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    endpoint = "https://api-inference.huggingface.co/models/your-model"
    result = call_huggingface_api(endpoint, data)
    
    if result is None:
        return jsonify({'error': 'Failed to call HuggingFace API'}), 500
    
    return jsonify(result)

# Update the prompt endpoint with more detailed examples
@app.route('/huggingface/prompt', methods=['GET'])
def huggingface_prompt():
    return jsonify({
        "format_instructions": FORMAT_INSTRUCTIONS,
        "example_response": EXAMPLE_RESPONSE
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)