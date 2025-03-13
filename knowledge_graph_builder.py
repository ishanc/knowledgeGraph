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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from pathlib import Path
import sys  # Import sys directly
from scipy import sparse  # Add this import

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    def __init__(self, processed_files_dir: str, output_dir: str):
        self.processed_files_dir = processed_files_dir
        self.output_dir = output_dir
        self.graph = Graph()
        self.nx_graph = nx.DiGraph()
        
        # Try to load standard model first
        try:
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_lg")  # Use large model instead of transformer
            logger.info("Using en_core_web_lg model")
        except OSError:
            logger.warning("Large model not found, falling back to standard model")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Using en_core_web_sm model")
            except OSError:
                logger.info("Downloading standard spaCy model...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])  # Use sys directly
                self.nlp = spacy.load("en_core_web_sm")
                
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

    def extract_entities_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities and relations with enhanced granularity"""
        doc = self.nlp(text)
        
        system_prompt = """You are an expert at extracting detailed entities and relationships from technical documents.
        Analyze the text with extreme detail and identify:
        1. Main concepts and their subconcepts (up to 3 levels deep)
        2. Technical terms and their definitions
        3. Process steps and their sequence
        4. Conditions and requirements
        5. Cross-references and dependencies
        6. Attributes and parameters
        7. Usage examples and contexts
        8. Related technical standards
        9. Implementation details
        10. Dependencies and prerequisites

        Format the response as JSON with the following detailed structure:
        {
            "entities": [
                {
                    "text": "entity_name",
                    "category": "detailed_category",
                    "weight": 0.8,
                    "level": 1,
                    "technical_details": {
                        "definition": "technical definition",
                        "parameters": ["param1", "param2"],
                        "requirements": ["req1", "req2"]
                    },
                    "implementation": {
                        "steps": ["step1", "step2"],
                        "conditions": ["condition1", "condition2"]
                    },
                    "subtopics": [{
                        "name": "subtopic1",
                        "level": 2,
                        "details": ["detail1", "detail2"]
                    }],
                    "relationships": [
                        {
                            "target": "other_entity",
                            "type": "relationship_type",
                            "strength": 0.7,
                            "dependency_type": "prerequisite|corequisite|postrequisite",
                            "context": "usage context"
                        }
                    ]
                }
            ]
        }"""
        
        response = self.mistral.generate_with_context(
            system_prompt=system_prompt,
            user_prompt=text,
            max_tokens=1000,
            temperature=0.1
        )
        
        try:
            enhanced_analysis = json.loads(response)
        except json.JSONDecodeError:
            enhanced_analysis = {"entities": []}
        
        # Combine Mistral's analysis with spaCy's extraction
        entities_relations = []
        seen_entities = set()
        
        # Add entities from Mistral's analysis
        for entity in enhanced_analysis.get("entities", []):
            if entity["text"] not in seen_entities:
                entities_relations.append({
                    'text': entity["text"],
                    'label': entity.get("category", "CONCEPT"),
                    'weight': entity.get("weight", 0.5),
                    'subtopics': entity.get("subtopics", []),
                    'relationships': entity.get("relationships", [])
                })
                seen_entities.add(entity["text"])
        
        # Add additional entities from spaCy
        for ent in doc.ents:
            if ent.text not in seen_entities:
                entities_relations.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'weight': 0.5,  # Default weight
                    'subtopics': [],
                    'relationships': []
                })
                seen_entities.add(ent.text)
        
        return entities_relations

    def extract_hierarchical_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract hierarchical relationships between entities"""
        system_prompt = """Analyze the text and extract hierarchical relationships between concepts.
        For each concept identify:
        1. Parent concepts (broader categories)
        2. Child concepts (more specific instances)
        3. Related concepts (semantically similar)
        4. Properties and attributes
        5. Actions or capabilities

        Format as JSON:
        {
            "concepts": [
                {
                    "name": "concept_name",
                    "type": "concept_type",
                    "parents": ["parent1", "parent2"],
                    "children": ["child1", "child2"],
                    "related": ["related1", "related2"],
                    "properties": ["prop1", "prop2"],
                    "actions": ["action1", "action2"],
                    "importance": 0.8
                }
            ]
        }"""
        
        response = self.mistral.generate_with_context(
            system_prompt=system_prompt,
            user_prompt=text,
            max_tokens=1500,
            temperature=0.1
        )
        
        try:
            concepts = json.loads(response)
            return concepts.get("concepts", [])  # Return the list directly
        except json.JSONDecodeError:
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
        system_prompt = """Given these key terms and example texts from a topic cluster, 
        generate a specific, meaningful topic name (2-4 words) that captures the main theme.
        Consider technical terms and domain-specific concepts.
        Focus on being precise rather than general.
        
        Example formats:
        - "Enterprise Configuration Management"
        - "Sales Document Processing"
        - "Tax Code Implementation"
        
        Terms: {terms}
        
        Example content: {content}
        
        Response should be just the topic name, nothing else."""
        
        sample_content = "\n".join(cluster_texts[:2])  # Use first two texts as examples
        prompt = system_prompt.format(
            terms=", ".join(top_terms[:10]),
            content=sample_content[:500]  # Limit content length
        )
        
        try:
            topic_name = self.mistral.generate_response(prompt, max_tokens=20, temperature=0.3)
            topic_name = topic_name.strip().replace('\n', ' ').title()
            return topic_name if topic_name else f"Topic ({', '.join(top_terms[:3])})"
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

    def build_graph(self):
        """Build enhanced knowledge graph with more granular relationships"""
        # Clear existing graph data
        self.graph = Graph()
        self.nx_graph = nx.DiGraph()
        self.embedding_cache = {}  # Clear cache to ensure consistent embeddings
        
        # Collect and process all texts
        all_texts = []
        text_sources = {}  # Map text to source file
        
        for filename in os.listdir(self.processed_files_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.processed_files_dir, filename), 'r') as f:
                    data = json.load(f)
                    content = str(data['data']['content'])
                    all_texts.append(content)
                    text_sources[content] = filename
        
        # Get topics and embeddings
        topics, doc_embeddings = self.identify_topics(all_texts)
        
        # Add topic nodes with enhanced properties
        for topic, terms in topics.items():
            self.nx_graph.add_node(topic,
                                 type='topic',
                                 terms=terms,
                                 size=30,
                                 color='#ff7f0e',
                                 embedding=self.get_embedding(topic))
        
        # Process each document
        similarity_threshold = 0.3  # Changed from 0.5
        for idx, content in enumerate(all_texts):
            # Extract hierarchical relationships with enhanced detail
            hierarchy = self.extract_hierarchical_relations(content)
            
            # Process each concept
            for concept in hierarchy:
                # Add concept node
                self.nx_graph.add_node(concept['name'],
                                     type='concept',
                                     concept_type=concept['type'],
                                     properties=concept['properties'],
                                     actions=concept['actions'],
                                     importance=concept.get('importance', 0.5),
                                     size=25 * concept.get('importance', 0.5),
                                     color='#1f77b4',
                                     embedding=self.get_embedding(concept['name']))
                
                # Add hierarchical relationships
                for parent in concept['parents']:
                    self.nx_graph.add_edge(concept['name'], parent,
                                         type='is_a',
                                         weight=0.8)
                
                for child in concept['children']:
                    self.nx_graph.add_edge(concept['name'], child,
                                         type='has',
                                         weight=0.8)
                
                for related in concept['related']:
                    self.nx_graph.add_edge(concept['name'], related,
                                         type='related_to',
                                         weight=0.6)
            
            # Connect to relevant topics using semantic similarity
            doc_embedding = np.array(doc_embeddings[idx]).reshape(1, -1)  # Convert to 2D array
            for topic in topics:
                topic_embedding = np.array(self.nx_graph.nodes[topic]['embedding']).reshape(1, -1)
                similarity = float(cosine_similarity(doc_embedding, topic_embedding)[0, 0])
                if similarity > similarity_threshold:  # Lower threshold for more connections
                    self.nx_graph.add_edge(text_sources[content],
                                         topic,
                                         type='belongs_to',
                                         weight=similarity)
        
        # Save embeddings cache
        self._save_cache()

    def visualize_graph(self, output_file: str = "knowledge_graph.html"):
        """Generate enhanced interactive visualization of the knowledge graph"""
        nt = net.Network(height="900px", width="100%", bgcolor="#ffffff")
        nt.toggle_physics(True)
        
        # Add nodes with enhanced visualization
        for node, data in self.nx_graph.nodes(data=True):
            size = data.get('size', 20)
            color = data.get('color', '#1f77b4')  # Default color for entities
            
            if data.get('type') == 'topic':
                title = f"Topic: {node}\nTerms: {', '.join(data['terms'])}"
            else:
                title = f"Entity: {node}\nType: {data.get('label', 'Unknown')}\nWeight: {data.get('weight', 0.5)}"
            
            nt.add_node(node,
                       label=node,
                       title=title,
                       size=size,
                       color=color)
        
        # Add edges with enhanced visualization
        for edge in self.nx_graph.edges(data=True):
            width = edge[2].get('weight', 0.5) * 5  # Edge width based on relationship strength
            nt.add_edge(edge[0],
                       edge[1],
                       title=edge[2].get('type', 'related'),
                       width=width)
        
        # Configure physics for better layout
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
          }
        }
        """)
        
        output_path = os.path.join(self.output_dir, output_file)
        nt.save_graph(output_path)
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
