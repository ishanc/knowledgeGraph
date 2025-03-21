import os
import sys
import time
import signal
import threading
import platform
import psutil
from datetime import datetime, timedelta
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from knowledgeGraph import process_file
import json
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
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
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

# Add CORS and security settings
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['CORS_HEADERS'] = 'Content-Type'

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

# Add production management related variables
HEALTH_CHECK_FILE = '.health_status'
SERVER_START_TIME = datetime.now()

# Track active requests for graceful shutdown
active_requests = 0
active_requests_lock = threading.Lock()
terminating = False

# Request tracking middleware
@app.before_request
def before_request():
    global active_requests
    if terminating:
        # If shutting down, don't accept new requests
        return jsonify({'error': 'Server is shutting down'}), 503
    
    with active_requests_lock:
        active_requests += 1

@app.teardown_request
def teardown_request(exception=None):
    global active_requests
    with active_requests_lock:
        active_requests -= 1

# Signal handler for graceful shutdown
def shutdown_handler(signum, frame):
    global terminating
    terminating = True
    
    logger.info("Shutdown signal received. Starting graceful shutdown...")
    logger.info(f"Waiting for {active_requests} active requests to complete...")
    
    # Set health check to unhealthy
    with open(HEALTH_CHECK_FILE, 'w') as f:
        f.write('UNHEALTHY')
    
    # Wait for active requests to finish (with timeout)
    shutdown_start = time.time()
    while active_requests > 0 and time.time() - shutdown_start < 30:  # 30 second timeout
        time.sleep(0.1)
    
    logger.info("Graceful shutdown completed")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# Create health check file on startup
with open(HEALTH_CHECK_FILE, 'w') as f:
    f.write('HEALTHY')

@app.route('/')
def index():
    print("Home route accessed!")  # Add debug print
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        # Return a basic HTML response if template fails
        return """
        <html>
            <head><title>Knowledge Graph</title></head>
            <body>
                <h1>Welcome to the Knowledge Graph!</h1>
                <p><a href="/graph/editor">Go to Graph Editor</a></p>
            </body>
        </html>
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

def clean_node_label(label):
    """Clean and normalize a node label."""
    if not label:
        return None
        
    # Remove extra whitespace
    label = ' '.join(label.split())
    
    # Fix broken words (remove random splits)
    label = label.replace(' o ', ' ')
    label = label.replace(' s ', 's ')
    label = label.replace(' ed ', 'ed ')
    label = label.replace(' ing ', 'ing ')
    label = label.replace(' ment', 'ment')
    label = label.replace(' tion', 'tion')
    
    # Remove single letters except 'a' and 'i'
    words = label.split()
    words = [w for w in words if len(w) > 1 or w.lower() in ['a', 'i']]
    
    return ' '.join(words).strip()

def is_valid_node_label(label):
    """Validate if a node label is meaningful and properly formatted."""
    if not label:
        return False
    
    # Basic validation rules
    if (len(label) < 3 or  # Too short
        len(label) > 100 or  # Too long
        len([c for c in label if c.isalpha()]) < 3):  # Not enough letters
        return False
        
    words = label.split()
    
    # Word-level validation
    for word in words:
        # Check for broken words or meaningless sequences
        if (len(word) == 1 and word.lower() not in ['a', 'i']) or \
           (len(word) < 3 and word.lower() not in ['an', 'of', 'to', 'in', 'on', 'at', 'by', 'up']):
            return False
            
    # Additional checks
    if (len(words) > 10 or  # Too many words
        sum(1 for c in label if c.isspace()) / len(label) > 0.3 or  # Too many spaces
        sum(1 for c in label if c.isalpha()) / len(label) < 0.5):  # Not enough letters
        return False
    
    return True

@app.route('/graph/data')
def get_graph_data():
    """Get the current graph data."""
    try:
        # Read from the existing knowledge graph HTML file
        graph_file = os.path.join(GRAPH_OUTPUT_DIR, 'knowledge_graph.html')
        if not os.path.exists(graph_file):
            logger.error("Knowledge graph visualization file not found")
            return jsonify({'error': 'Knowledge graph visualization not found'}), 404

        with open(graph_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract nodes and edges data using regex
        import re
        nodes_pattern = r'nodes\s*=\s*new\s*vis\.DataSet\((.*?)\);'
        edges_pattern = r'edges\s*=\s*new\s*vis\.DataSet\((.*?)\);'
        
        nodes_match = re.search(nodes_pattern, content, re.DOTALL)
        edges_match = re.search(edges_pattern, content, re.DOTALL)
        
        if not nodes_match or not edges_match:
            logger.error("Could not find graph data in visualization file")
            return jsonify({'error': 'Failed to extract graph data'}), 500

        try:
            nodes_data = json.loads(nodes_match.group(1))
            edges_data = json.loads(edges_match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing graph data: {e}")
            return jsonify({'error': 'Invalid graph data format'}), 500

        return jsonify({
            'nodes': nodes_data,
            'edges': edges_data
        })

    except Exception as e:
        logger.error(f"Error in get_graph_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/graph/node', methods=['POST'])
def add_node():
    data = request.get_json()
    try:
        # Clean and validate the label
        label = clean_node_label(data['label'])
        if not label or not is_valid_node_label(label):
            return jsonify({'error': 'Invalid node label. Label must be meaningful and properly formatted.'}), 400
            
        node_data = {
            'type': data.get('type', 'concept'),
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'source': 'user_edit',
            'confidence': 1.0
        }
        
        # Check for duplicate nodes (case insensitive)
        if any(n for n in kg_builder.nx_graph.nodes if n.lower() == label.lower()):
            return jsonify({'error': 'A node with this label already exists'}), 400
            
        kg_builder.nx_graph.add_node(label, **kg_builder._prepare_node_data(node_data))
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

@app.route('/graph/relationship/<source>/<target>', methods=['PUT', 'DELETE'])
def modify_relationship(source, target):
    try:
        if request.method == 'DELETE':
            kg_builder.nx_graph.remove_edge(source, target)
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Validate relationship data
            if 'type' not in data:
                return jsonify({'error': 'Relationship type is required'}), 400

            edge_data = {
                'type': data['type'],
                'weight': data.get('weight', 1.0),
                'timestamp': datetime.now().isoformat(),
                'source': 'user_edit',
                'confidence': data.get('confidence', 1.0)
            }
            
            # Update edge data
            kg_builder.nx_graph[source][target].update(
                kg_builder._prepare_node_data(edge_data)
            )
            
        kg_builder.visualize_graph()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error modifying relationship: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/graph/history/<node_id>')
def get_node_history(node_id):
    try:
        history = kg_builder.get_node_history(node_id)
        return jsonify({'history': history})
    except Exception as e:
        logger.error(f"Error getting node history: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Enhance the stats endpoint
@app.route('/graph/stats')
def get_graph_stats():
    try:
        nodes = list(kg_builder.nx_graph.nodes(data=True))
        edges = list(kg_builder.nx_graph.edges(data=True))
        
        # Get edge types
        edge_types = {}
        for _, _, data in edges:
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        stats = {
            'nodes': len(nodes),
            'edges': len(edges),
            'topics': sum(1 for _, data in nodes if str(data.get('type', '')).lower() == 'topic'),
            'concepts': sum(1 for _, data in nodes if str(data.get('type', '')).lower() in ['concept', 'entity']),
            'edge_types': edge_types,
            'last_modified': datetime.now().isoformat()
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

@app.route('/restart', methods=['POST'])
def restart_server():
    """
    Restart the Flask server and return stats.
    
    For production use, this endpoint can be configured to:
    1. Signal a process manager (systemd, supervisor, etc.) to restart the service
    2. Or perform a graceful in-process restart if running standalone
    """
    try:
        # Collect stats before restart
        stats = {
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - SERVER_START_TIME),
            'uptime_seconds': (datetime.now() - SERVER_START_TIME).total_seconds(),
            'nodes': len(list(kg_builder.nx_graph.nodes())) if hasattr(kg_builder, 'nx_graph') else 0,
            'edges': len(list(kg_builder.nx_graph.edges())) if hasattr(kg_builder, 'nx_graph') else 0,
            'processed_files': len(os.listdir(OUTPUT_FOLDER)) if os.path.exists(OUTPUT_FOLDER) else 0,
            'active_requests': active_requests
        }
        
        # Write health check as unhealthy to prevent new traffic
        with open(HEALTH_CHECK_FILE, 'w') as f:
            f.write('RESTARTING')
        
        # Check if we're running in a production environment with a process manager
        # You can customize this check for your deployment environment
        if os.environ.get('USE_PROCESS_MANAGER') == 'true':
            # Signal process manager to restart the service
            # This is a safer approach for production
            
            # This example uses touch to modify a file that a process manager 
            # like supervisor can watch for changes
            restart_trigger_file = os.environ.get('RESTART_TRIGGER_FILE', '.restart_trigger')
            with open(restart_trigger_file, 'w') as f:
                f.write(datetime.now().isoformat())
            
            return jsonify({
                'status': 'success', 
                'message': 'Restart signal sent to process manager', 
                'stats': stats,
                'restart_mode': 'process_manager'
            })
        else:
            # Development mode in-process restart
            # Set environment variable for clean restart
            os.environ['WERKZEUG_RUN_MAIN'] = 'true'
            
            # Schedule a restart after responding to the client
            def restart():
                # Wait for active requests to finish (max 5 seconds)
                wait_start = time.time()
                while active_requests > 1 and time.time() - wait_start < 5:  # >1 because this request is active
                    time.sleep(0.1)
                
                # Perform restart
                time.sleep(1)
                os.execv(sys.executable, [sys.executable] + sys.argv)
            
            # Start the restart process in a separate thread
            restart_thread = threading.Thread(target=restart)
            restart_thread.daemon = True
            restart_thread.start()
            
            # Return success with stats
            return jsonify({
                'status': 'success', 
                'message': 'Server restarting', 
                'stats': stats,
                'restart_mode': 'in_process'
            })
    except Exception as e:
        logger.error(f"Error restarting server: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Check if kg_builder is functional
        if not hasattr(kg_builder, 'nx_graph'):
            return jsonify({
                'status': 'warning',
                'message': 'Knowledge graph not initialized'
            }), 200  # Return 200 with warning status
        
        # Perform basic DB/file system check
        if not os.path.exists(OUTPUT_FOLDER):
            return jsonify({
                'status': 'warning',
                'message': 'Output folder not found'
            }), 200
        
        # All checks passed
        return jsonify({
            'status': 'healthy',
            'message': 'Service is healthy',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 503  # Service Unavailable

@app.route('/status')
def system_status():
    """System status endpoint with detailed metrics for monitoring."""
    uptime = datetime.now() - SERVER_START_TIME
    
    # Get system metrics
    process = psutil.Process(os.getpid())
    
    try:
        status = {
            'server': {
                'start_time': SERVER_START_TIME.isoformat(),
                'uptime': str(uptime),
                'uptime_seconds': uptime.total_seconds(),
                'environment': os.environ.get('FLASK_ENV', 'production'),
                'python_version': platform.python_version(),
                'platform': platform.platform()
            },
            'process': {
                'cpu_percent': process.cpu_percent(),
                'memory_usage_mb': process.memory_info().rss / (1024 * 1024),
                'thread_count': process.num_threads(),
                'active_requests': active_requests
            },
            'graph': {
                'nodes': len(list(kg_builder.nx_graph.nodes())) if hasattr(kg_builder, 'nx_graph') else 0,
                'edges': len(list(kg_builder.nx_graph.edges())) if hasattr(kg_builder, 'nx_graph') else 0
            },
            'storage': {
                'processed_files': len(os.listdir(OUTPUT_FOLDER)) if os.path.exists(OUTPUT_FOLDER) else 0
            }
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add server start time tracking
if not os.getenv('SERVER_START_TIME'):
    os.environ['SERVER_START_TIME'] = datetime.now().isoformat()

if __name__ == '__main__':
    # Add startup message with environment info
    logger.info(f"Starting application server on port 8080")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    app.run(host='0.0.0.0', port=8080, debug=True)