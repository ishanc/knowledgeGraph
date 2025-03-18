import networkx as nx
from rdflib import Graph, Namespace
import json
import os
import spacy
import pyvis.network as net
from typing import Dict, List, Any, Tuple
import logging
from mistral_wrapper import MistralWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path
import sys  # Import sys directly
from scipy import sparse  # Add this import
from datetime import datetime, date  # Add this import

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Knowledge Graph Builder for processing and visualizing document relationships."""
    
    def __init__(self, processed_files_dir: str, output_dir: str) -> None:
        self.processed_files_dir = processed_files_dir
        self.output_dir = output_dir
        self.graph = Graph()
        self.nx_graph = nx.DiGraph()
        self.max_text_length = 5000000  # Increased to 5M characters
        
        # Try to load standard model first
        try:
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_lg")
            self.nlp.max_length = self.max_text_length
            logger.info("Using en_core_web_lg model")
        except OSError:
            logger.warning("Large model not found, falling back to standard model")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = self.max_text_length  # Set max length for small model too
                logger.info("Using en_core_web_sm model")
            except OSError:
                logger.info("Downloading standard spaCy model...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])  # Use sys directly
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = self.max_text_length  # Set max length for downloaded model
                
        self.mistral = MistralWrapper()
        
        # Create namespace for our knowledge graph
        self.KG = Namespace("http://knowledge.graph/")
        self.graph.bind("kg", self.KG)
        
        # Use a consistent embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Change back to consistent model
        
        # Reset cache to avoid dimension mismatch with previous embeddings
        self.cache_dir = Path(output_dir) / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        if (self.cache_dir / 'embeddings.joblib').exists():
            os.remove(self.cache_dir / 'embeddings.joblib')
        
        # Initialize empty cache
        self.embedding_cache = {}
        self._load_cache()

        self.relationship_hierarchy = {
            'is_a': 1,  # Highest priority
            'has': 0.9,
            'part_of': 0.8,
            'belongs_to': 0.7,
            'related_to': 0.6,
            'similar_to': 0.5,
            'referenced_by': 0.4,
            'mentioned_in': 0.3
        }

        # Add conversion utilities
        self.serializable_types = (int, float, str, bool, list, dict)
        
        # Define numpy numeric types
        self.numpy_int_types = (np.integer,)  # Base class for numpy integer types
        self.numpy_float_types = (np.floating,)  # Base class for numpy float types

    def _load_cache(self):
        """Load cached embeddings if they exist"""
        cache_file = self.cache_dir / 'embeddings.joblib'
        if cache_file.exists():
            self.embedding_cache = joblib.load(cache_file)

    def _save_cache(self):
        """Save embeddings cache to disk"""
        cache_file = self.cache_dir / 'embeddings.joblib'
        joblib.dump(self.embedding_cache, cache_file)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get or compute embedding for text with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embedding_model.encode(text)
        embedding_array = np.array(embedding)  # Convert to numpy array
        self.embedding_cache[text] = embedding_array
        return embedding_array

    def _chunk_text(self, text: str, chunk_size: int = 100000) -> List[str]:
        """Split text into manageable chunks."""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            if current_length + len(para) > chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def _safe_chunk_text(self, text: str) -> List[str]:
        """Safely chunk very large texts with intelligent boundary detection"""
        if len(text) <= self.max_text_length:
            return [text]
            
        chunk_size = self.max_text_length // 2  # Use half max length for safety
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # Find the last complete sentence or paragraph boundary
            possible_boundaries = [
                text.rfind('\n\n', start, end),  # Paragraph
                text.rfind('. ', start, end),     # Sentence
                text.rfind('? ', start, end),     # Question
                text.rfind('! ', start, end),     # Exclamation
                text.rfind('\n', start, end),     # Line break
            ]
            
            # Use the latest boundary found, or fall back to word boundary
            boundary = max(b for b in possible_boundaries if b != -1)
            if boundary == -1:
                boundary = text.rfind(' ', start, end)
            if boundary == -1:  # If no good boundary found
                boundary = end - 1
                
            chunk = text[start:boundary + 1].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = boundary + 1
            
        return chunks

    def extract_entities_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities and relations with enhanced granularity and chunking"""
        # First break text into safe chunks
        safe_chunks = self._safe_chunk_text(text)
        all_entities = []
        seen_entities = set()
        
        for chunk in safe_chunks:
            try:
                # Further break down into processing chunks with overlap
                process_size = 90000  # Smaller chunks for processing
                overlap = 5000
                
                start = 0
                while start < len(chunk):
                    end = min(start + process_size, len(chunk))
                    
                    # Find appropriate boundary
                    if end < len(chunk):
                        # Look for sentence or paragraph boundary
                        for boundary in ['\n\n', '. ', '? ', '! ', '\n', ' ']:
                            pos = chunk.rfind(boundary, start, end)
                            if pos != -1:
                                end = pos + len(boundary)
                                break
                    
                    process_chunk = chunk[start:end].strip()
                    
                    if process_chunk:
                        messages = [{
                            "role": "user",
                            "content": f"""Extract entities and relationships from this text. Return only a JSON object with this exact structure:
{{
    "entities": [
        {{
            "text": "example entity",
            "category": "CONCEPT|PERSON|ORGANIZATION|TECHNOLOGY|PROCESS|FEATURE",
            "weight": 0.8,
            "relationships": [
                {{
                    "type": "is_a|has|part_of|belongs_to|related_to",
                    "target": "other entity",
                    "weight": 0.7
                }}
            ]
        }}
    ]
}}

Example valid response:
{{
    "entities": [
        {{
            "text": "SAP S/4HANA",
            "category": "TECHNOLOGY",
            "weight": 0.9,
            "relationships": [
                {{
                    "type": "has",
                    "target": "Enterprise Management",
                    "weight": 0.8
                }}
            ]
        }},
        {{
            "text": "Enterprise Management",
            "category": "CONCEPT",
            "weight": 0.7,
            "relationships": []
        }}
    ]
}}

Text to analyze: {process_chunk}"""
                        }]
                        
                        try:
                            response = self.mistral.generate_with_context(
                                messages=messages,
                                max_tokens=1000,
                                temperature=0.1
                            )
                            
                            # Initialize response_str and enhanced_analysis
                            response_str = ""
                            enhanced_analysis = {"entities": []}
                            
                            if response:
                                # Attempt to extract just the JSON part if there's any extra text
                                json_start = response.find('{')
                                json_end = response.rfind('}') + 1
                                if json_start >= 0 and json_end > json_start:
                                    response_str = response[json_start:json_end]
                                
                                if response_str:
                                    try:
                                        enhanced_analysis = json.loads(response_str)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse JSON response for chunk. Response: {response_str[:100]}...")
                            
                            # Add entities from Mistral's analysis
                            for entity in enhanced_analysis.get("entities", []):
                                if entity["text"] not in seen_entities:
                                    all_entities.append({
                                        'text': entity["text"],
                                        'label': entity.get("category", "CONCEPT"),
                                        'weight': entity.get("weight", 0.5),
                                        'relationships': entity.get("relationships", [])
                                    })
                                    seen_entities.add(entity["text"])
                            
                            # Get spaCy entities if needed
                            if not enhanced_analysis.get("entities"):
                                for ent in self.nlp(process_chunk).ents:
                                    if ent.text not in seen_entities:
                                        all_entities.append({
                                            'text': ent.text,
                                            'label': ent.label_,
                                            'weight': 0.5,
                                            'relationships': []
                                        })
                                        seen_entities.add(ent.text)
                                        
                        except Exception as e:
                            logger.warning(f"Error processing response: {str(e)}")
                    
                    # Move start position with overlap
                    start = end - overlap
                    if start < end - overlap // 2:  # Prevent tiny chunks at the end
                        start = end
                        
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        return all_entities

    def extract_hierarchical_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract hierarchical relationships between entities"""
        messages = [{
            "role": "user",
            "content": f"""Analyze this text and identify hierarchical relationships. Return only a JSON object with this exact structure:
{{
    "concepts": [
        {{
            "text": "example concept",
            "parents": ["parent1", "parent2"],
            "children": ["child1", "child2"],
            "related": ["related1", "related2"]
        }}
    ]
}}

Example valid response:
{{
    "concepts": [
        {{
            "text": "Enterprise Management",
            "parents": ["SAP S/4HANA"],
            "children": ["Finance", "Logistics"],
            "related": ["Business Process"]
        }}
    ]
}}

Text to analyze: {text}"""
        }]
        
        try:
            response = self.mistral.generate_with_context(
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            
            # Initialize response_str and concepts
            response_str = ""
            concepts_data = {"concepts": []}
            
            if response:
                # Attempt to extract just the JSON part if there's any extra text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_str = response[json_start:json_end]
            
                if response_str:
                    try:
                        concepts_data = json.loads(response_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON response: {response_str[:100]}...")
                        return []
            
            if not isinstance(concepts_data, dict) or "concepts" not in concepts_data:
                logger.warning("Invalid JSON structure received")
                return []
            
            return concepts_data["concepts"]
            
        except Exception as e:
            logger.error(f"Error extracting hierarchical relations: {e}")
            return []

    def identify_topics(self, texts: List[str], n_topics: int = 15) -> Tuple[Dict[str, List[str]], np.ndarray]:
        """Identify topics using hierarchical clustering and improved chunking"""
        if not texts:
            return {"General": ["no_documents"]}, np.array([])
        
        # Split texts into smaller chunks for more granular topic identification
        processed_texts = []
        chunk_sources = {}  # Track which document each chunk came from
        
        for idx, text in enumerate(texts):
            chunks = self._split_into_chunks(text, max_length=500)
            processed_texts.extend(chunks)
            for chunk in chunks:
                chunk_sources[chunk] = idx
        
        # Get embeddings for all chunks
        embeddings = np.vstack([self.get_embedding(text) for text in processed_texts])
        
        # Calculate optimal number of topics with higher minimum
        n_topics = self._calculate_optimal_topics(embeddings, min_topics=8, max_topics=25)
        
        # Use DBSCAN for initial clustering
        eps = self._calculate_optimal_eps(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=2)
        cluster_labels = clustering.fit_predict(embeddings)
        
        # If DBSCAN didn't find enough clusters, use hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        if len(set(cluster_labels)) < n_topics:
            clustering = AgglomerativeClustering(n_clusters=n_topics)
            cluster_labels = clustering.fit_predict(embeddings)
        
        # Enhanced topic extraction with TF-IDF and ngrams
        vectorizer = TfidfVectorizer(
            max_features=3000,  # Increased for more terms
            stop_words='english',
            ngram_range=(1, 4),  # Include up to 4-grams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        terms = vectorizer.get_feature_names_out()
        
        # Process each cluster to identify topics
        topics = {}
        for label in set(cluster_labels):
            if label == -1:
                continue
            
            # Get chunks in this cluster
            cluster_mask = cluster_labels == label
            cluster_texts = [text for i, text in enumerate(processed_texts) if cluster_mask[i]]
            
            # Get TF-IDF scores for this cluster
            cluster_mask_matrix = sparse.csr_matrix(tfidf_matrix)[cluster_mask].toarray()  # Modified line
            significance_scores = np.mean(cluster_mask_matrix, axis=0)
            
            # Get more terms for better topic characterization
            top_term_indices = significance_scores.argsort()[-20:][::-1]
            top_terms = [str(terms[i]) for i in top_term_indices]
            
            # Generate specific topic name using context
            topic_name = self._generate_specific_topic_name(top_terms, cluster_texts)
            topics[topic_name] = top_terms
        
        return topics, embeddings

    def _split_into_chunks(self, text: str, max_length: int = 300) -> List[str]:
        """Split text into smaller chunks for more granular analysis"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            # Try to keep related sentences together
            if current_length + len(sent.text) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # For very long sentences, split by tokens
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                words = chunk.split()
                temp_chunk = []
                temp_length = 0
                for word in words:
                    if temp_length + len(word) > max_length and temp_chunk:
                        final_chunks.append(' '.join(temp_chunk))
                        temp_chunk = []
                        temp_length = 0
                    temp_chunk.append(word)
                    temp_length += len(word) + 1
                if temp_chunk:
                    final_chunks.append(' '.join(temp_chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def _calculate_optimal_topics(self, embeddings: np.ndarray, min_topics: int = 15, max_topics: int = 50) -> int:
        """Calculate optimal number of topics with higher granularity"""
        from sklearn.metrics import silhouette_score
        
        n_samples = embeddings.shape[0]
        if n_samples < min_topics:
            return max(5, n_samples)
        
        scores = []
        for n in range(min_topics, min(max_topics + 1, n_samples)):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append((n, score))
        
        # Pick the number of topics that gives the best silhouette score
        return max(scores, key=lambda x: x[1])[0]

    def _calculate_optimal_eps(self, embeddings: np.ndarray) -> float:
        """Calculate optimal eps parameter for DBSCAN using nearest neighbors"""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        distances = np.sort(distances[:, 1])
        
        # Find the elbow point
        diffs = np.diff(distances)
        elbow_point = np.argmax(diffs) + 1
        return distances[elbow_point]

    def _generate_specific_topic_name(self, top_terms: List[str], cluster_texts: List[str]) -> str:
        """Generate more specific and meaningful topic names"""
        sample_content = "\n".join(cluster_texts[:2])
        
        messages = [{
            "role": "user",
            "content": f"""Generate a specific topic name (2-4 words) based on these terms and content:

Terms: {', '.join(top_terms[:10])}
Content: {sample_content[:500]}"""
        }]
        
        try:
            topic_name = self.mistral.generate_with_context(
                messages=messages,
                max_tokens=20,
                temperature=0.3
            )
            if topic_name is None:
                return f"Topic ({', '.join(top_terms[:3])})"
            
            cleaned_name = topic_name.strip().replace('\n', ' ').title()
            return cleaned_name if cleaned_name else f"Topic ({', '.join(top_terms[:3])})"
        except Exception as e:
            logger.warning(f"Error generating specific topic name: {e}")
            return f"Topic ({', '.join(top_terms[:3])})"

    def create_dynamic_schema(self, json_data: Dict) -> Dict[str, Any]:
        """Generate dynamic schema based on JSON content"""
        schema = {'classes': set(), 'properties': set(), 'relationships': set()}
        
        def analyze_value(value, parent_key=None):
            if isinstance(value, dict):
                for k, v in value.items():
                    schema['classes'].add(k.title())
                    if parent_key:
                        schema['relationships'].add((parent_key.title(), 'has' + k.title(), k.title()))
                    analyze_value(v, k)
            elif isinstance(value, list):
                for item in value:
                    analyze_value(item, parent_key)
            else:
                if parent_key:
                    schema['properties'].add((parent_key.title(), str(type(value).__name__)))

        analyze_value(json_data)
        return schema

    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, self.numpy_int_types):
            return int(obj)
        if isinstance(obj, self.numpy_float_types):
            return float(obj)
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(i) for i in obj]
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        return obj

    def _prepare_node_data(self, data):
        """Prepare node data for storage"""
        serializable_data = {}
        for key, value in data.items():
            if key == 'embedding':
                # Store embeddings separately or convert to list
                serializable_data[key] = self._make_json_serializable(value)
            else:
                # Ensure type field is always lowercase for consistent comparison
                if key == 'type':
                    value = str(value).lower()
                serializable_data[key] = self._make_json_serializable(value)
        
        # Ensure type is set if missing
        if 'type' not in serializable_data:
            serializable_data['type'] = 'concept'  # Default type
            
        return serializable_data

    def build_graph(self) -> None:
        """Build the knowledge graph from processed files."""
        # Clear existing graph data
        self.graph = Graph()
        self.nx_graph = nx.DiGraph()
        self.embedding_cache = {}  # Clear cache to ensure consistent embeddings
        
        # Track all entities for relationship building
        all_entities = {}  # Dict to store all entities by text
        
        # Collect all texts first for topic identification
        all_texts = []
        text_sources = {}
        
        for filename in os.listdir(self.processed_files_dir):
            if not filename.endswith('.json'):
                continue
                
            with open(os.path.join(self.processed_files_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    content = str(data['data']['content'])
                    all_texts.append(content)
                    text_sources[content] = filename
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {str(e)}")
                    continue
        
        # First identify topics
        topics, _ = self.identify_topics(all_texts)
        
        # Add topic nodes
        for topic_name, terms in topics.items():
            node_data = {
                'type': 'topic',
                'terms': terms,
                'label': topic_name,
                'weight': 1.0,
                'color': '#ff7f0e',  # Orange for topics
                'size': 30,
                'embedding': self.get_embedding(topic_name)
            }
            self.nx_graph.add_node(topic_name, **self._prepare_node_data(node_data))
            logger.debug(f"Added topic node: {topic_name}")
            all_entities[topic_name] = node_data
        
        # First pass: Create all entity nodes
        for content, filename in text_sources.items():
            try:
                entities = self.extract_entities_relations(content)
                
                # Add concept nodes first
                for entity in entities:
                    node_data = {
                        'type': 'concept',
                        'label': entity['label'],
                        'weight': entity.get('weight', 0.5),
                        'color': '#1f77b4',  # Blue for concepts
                        'size': 25,
                        'embedding': self.get_embedding(entity['text']),
                        'source_file': filename
                    }
                    
                    self.nx_graph.add_node(
                        entity['text'],
                        **self._prepare_node_data(node_data)
                    )
                    all_entities[entity['text']] = node_data
                    logger.debug(f"Added concept node: {entity['text']}")
            except Exception as e:
                logger.error(f"Error in first pass for {filename}: {str(e)}")
                continue
        
        # Second pass: Create relationships
        for content, filename in text_sources.items():
            try:
                entities = self.extract_entities_relations(content)
                
                for entity in entities:
                    source_text = entity['text']
                    
                    # Add relationships from entity analysis
                    for rel in entity.get('relationships', []):
                        target_text = rel['target']
                        
                        # Only create edge if both nodes exist
                        if source_text in all_entities and target_text in all_entities:
                            edge_data = {
                                'type': rel['type'],
                                'weight': rel.get('weight', 0.5),
                                'source_file': filename,
                                'color': '#2ecc71'  # Green for entity relationships
                            }
                            self.nx_graph.add_edge(
                                source_text,
                                target_text,
                                **self._prepare_node_data(edge_data)
                            )
                            logger.debug(f"Added relationship: {source_text} -{rel['type']}-> {target_text}")
                    
                    # Connect to relevant topics with higher threshold
                    entity_embedding = self.get_embedding(source_text)
                    for topic_name, topic_data in all_entities.items():
                        if topic_data.get('type') == 'topic':
                            topic_embedding = topic_data['embedding']
                            similarity = float(self.cosine_similarity(
                                entity_embedding.reshape(1, -1),
                                topic_embedding.reshape(1, -1)
                            ))
                            
                            if similarity > 0.4:  # Increased threshold for better precision
                                edge_data = {
                                    'type': 'belongs_to',
                                    'weight': similarity,
                                    'source_file': filename,
                                    'color': '#e74c3c'  # Red for topic relationships
                                }
                                self.nx_graph.add_edge(
                                    source_text,
                                    topic_name,
                                    **self._prepare_node_data(edge_data)
                                )
                                logger.debug(f"Connected to topic: {source_text} -> {topic_name}")
                            
            except Exception as e:
                logger.error(f"Error in second pass for {filename}: {str(e)}")
                continue
        
        # Save embeddings cache
        self._save_cache()
        
        # Log final counts
        nodes = list(self.nx_graph.nodes(data=True))
        edges = list(self.nx_graph.edges(data=True))
        logger.info(f"Graph built with {len(nodes)} nodes and {len(edges)} edges")
        logger.info(f"Topics: {sum(1 for _, data in nodes if data.get('type') == 'topic')}")
        logger.info(f"Concepts: {sum(1 for _, data in nodes if data.get('type') == 'concept')}")
        logger.info(f"Relationships: {len(edges)}")

    def visualize_graph(self, output_file: str = "knowledge_graph.html") -> None:
        """Generate and save an interactive visualization of the knowledge graph."""
        # Initialize network
        nt = net.Network(height="900px", width="100%", bgcolor="#ffffff")
        nt.toggle_physics(True)

        # Add nodes and edges as before
        for node, data in self.nx_graph.nodes(data=True):
            size = data.get('size', 20)
            color = data.get('color', '#1f77b4')
            
            # Enhanced node metadata for better semantic understanding
            if data.get('type') == 'topic':
                title = (f"Topic: {node}\n"
                        f"Terms: {', '.join(data['terms'])}\n"
                        f"Weight: {data.get('weight', 0.5)}\n"
                        f"Connected Topics: {len(self.nx_graph[node])}")
                node_type = "topic"
            else:
                title = (f"Entity: {node}\n"
                        f"Type: {data.get('label', 'Unknown')}\n"
                        f"Weight: {data.get('weight', 0.5)}\n"
                        f"Properties: {', '.join(str(k) + ': ' + str(v) for k,v in data.items() if k not in ['size', 'color', 'label', 'weight'])}")
                node_type = data.get('label', 'entity')
                
            nt.add_node(node,
                       label=node,
                       title=title,
                       size=size,
                       color=color,
                       node_type=node_type,
                       weight=data.get('weight', 0.5),
                       metadata=data)

        for source, target, data in self.nx_graph.edges(data=True):
            width = data.get('weight', 1) * 5
            relation_type = data.get('type', 'related')
            title = (f"Relationship: {relation_type}\n"
                    f"Source: {source}\n"
                    f"Target: {target}\n"
                    f"Weight: {data.get('weight', 1)}\n"
                    f"Properties: {', '.join(str(k) + ': ' + str(v) for k,v in data.items() if k not in ['weight', 'type'])}")
                    
            nt.add_edge(source,
                       target,
                       title=title,
                       width=width,
                       relation_type=relation_type,
                       weight=data.get('weight', 1),
                       metadata=data)

        # Set physics options as before
        nt.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          },
          "nodes": {
            "font": {
              "size": 12,
              "strokeWidth": 2
            }
          },
          "edges": {
            "font": {
              "size": 10,
              "align": "middle"
            },
            "smooth": {
              "type": "continuous",
              "roundness": 0.5
            }
          }
        }
        """)
        
        # Generate the graph HTML content
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Knowledge Graph Visualization</title>
                <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.css" rel="stylesheet" type="text/css">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
                <style type="text/css">
                    #mynetwork {{
                        width: 100%;
                        height: 900px;
                        background-color: #ffffff;
                        border: 1px solid lightgray;
                    }}
                </style>
            </head>
            <body>
                <div id="mynetwork"></div>
                <script type="text/javascript">
                    function drawGraph() {{
                        var container = document.getElementById('mynetwork');
                        var nodes = new vis.DataSet({json.dumps(list(nt.nodes))});
                        var edges = new vis.DataSet({json.dumps(list(nt.edges))});
                        var data = {{nodes: nodes, edges: edges}};
                        var options = {nt.options};
                        var network = new vis.Network(container, data, options);
                    }}
                    window.addEventListener('load', function() {{ drawGraph(); }});
                </script>
            </body>
            </html>
        """
        
        # Save the HTML file
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Enhanced graph visualization saved to {output_path}")

    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language"""
        # Execute query and return results
        results = []
        for subj, pred, obj in self.graph:
            if query.lower() in str(subj).lower() or query.lower() in str(obj).lower():
                results.append({
                    'subject': str(subj),
                    'relation': str(pred),
                    'object': str(obj)
                })
        return results

    def get_node_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get the history of changes for a node"""
        if node_id not in self.nx_graph:
            return []
            
        node_data = self.nx_graph.nodes[node_id]
        history = [{
            'timestamp': node_data.get('timestamp'),
            'source': node_data.get('source'),
            'confidence': node_data.get('confidence'),
            'version': node_data.get('version'),
            'type': 'node_created'
        }]
        
        # Add relationship changes
        for _, neighbor, edge_data in self.nx_graph.edges(node_id, data=True):
            history.append({
                'timestamp': edge_data.get('timestamp'),
                'source': edge_data.get('source'),
                'confidence': edge_data.get('confidence'),
                'type': 'relationship_added',
                'relationship': edge_data.get('type'),
                'target': neighbor
            })
            
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        return history

    def get_relationship_confidence(self, source: str, target: str) -> float:
        """Get the confidence score for a relationship between two nodes"""
        if not self.nx_graph.has_edge(source, target):
            return 0.0
            
        edge_data = self.nx_graph.get_edge_data(source, target)
        return edge_data.get('confidence', 0.0)

    def validate_relationship(self, source: str, target: str, relationship_type: str) -> Dict[str, Any]:
        """Validate a relationship between nodes"""
        if not self.nx_graph.has_edge(source, target):
            return {'valid': False, 'reason': 'No relationship exists'}
            
        edge_data = self.nx_graph.get_edge_data(source, target)
        
        return {
            'valid': edge_data.get('type') == relationship_type,
            'confidence': edge_data.get('confidence', 0.0),
            'source': edge_data.get('source'),
            'timestamp': edge_data.get('timestamp'),
            'actual_type': edge_data.get('type')
        }

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between two vectors or sets of vectors"""
        # Convert inputs to numpy arrays if they aren't already
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
            
        # Ensure we're working with numpy arrays
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        
        # Ensure inputs are 2D
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
            
        # Avoid division by zero
        norm_a = np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
        norm_b = np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)
        
        # Calculate cosine similarity
        return np.dot(a / norm_a, (b / norm_b).T)
