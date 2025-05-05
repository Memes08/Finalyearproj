import logging
import csv
import json
import os

class Neo4jGraphManager:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        self.graph = None
        self.triples_storage = {}  # Simple in-memory storage
        
        logging.warning("Neo4j package not installed. Using in-memory storage instead.")
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        raise NotImplementedError("Neo4j connection is not available. Required packages not installed.")
    
    def get_neo4j_graph(self, graph_id):
        """Get Neo4j graph for a specific knowledge graph ID"""
        raise NotImplementedError("Neo4j graph retrieval is not available. Required packages not installed.")
    
    def import_triples(self, triples, graph_id):
        """Import triples into storage with graph_id as a property"""
        try:
            # Clear existing graph data for this graph_id to avoid duplicates
            if str(graph_id) in self.triples_storage:
                del self.triples_storage[str(graph_id)]
            
            # Create directory for CSV storage
            os.makedirs('data/triples', exist_ok=True)
            file_path = f'data/triples/graph_{graph_id}.csv'
            
            # Store in memory
            self.triples_storage[str(graph_id)] = triples
            
            # Save to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['entity1', 'relation', 'entity2'])
                for triple in triples:
                    writer.writerow(triple)
            
            logging.info(f"Successfully stored {len(triples)} triples for graph {graph_id}")
            return True
        
        except Exception as e:
            logging.error(f"Error storing triples: {e}")
            raise RuntimeError(f"Failed to store triples: {e}")
    
    def get_graph_data(self, graph_id):
        """Get nodes and relationships for visualization"""
        try:
            graph_id = str(graph_id)
            if graph_id not in self.triples_storage:
                # Try to load from file
                file_path = f'data/triples/graph_{graph_id}.csv'
                if os.path.exists(file_path):
                    triples = []
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)  # Skip header
                        for row in reader:
                            if len(row) == 3:
                                triples.append(tuple(row))
                    self.triples_storage[graph_id] = triples
                else:
                    return [], []  # No data found
            
            # Get triples from memory
            triples = self.triples_storage[graph_id]
            
            # Extract unique entities
            entities = set()
            for source, _, target in triples:
                entities.add(source)
                entities.add(target)
            
            # Create entity to ID mapping
            entity_to_id = {entity: i for i, entity in enumerate(entities)}
            
            # Create nodes
            nodes = [{"id": i, "label": entity} for entity, i in entity_to_id.items()]
            
            # Create relationships
            relationships = [
                {
                    "source": entity_to_id[source],
                    "target": entity_to_id[target],
                    "type": relation
                }
                for source, relation, target in triples
            ]
            
            return nodes, relationships
        
        except Exception as e:
            logging.error(f"Error retrieving graph data: {e}")
            raise RuntimeError(f"Failed to retrieve graph data: {e}")
    
    def delete_graph(self, graph_id):
        """Delete all nodes and relationships for a knowledge graph"""
        try:
            graph_id = str(graph_id)
            if graph_id in self.triples_storage:
                del self.triples_storage[graph_id]
            
            # Delete file if exists
            file_path = f'data/triples/graph_{graph_id}.csv'
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logging.info(f"Successfully deleted graph {graph_id}")
            return True
        
        except Exception as e:
            logging.error(f"Error deleting graph: {e}")
            raise RuntimeError(f"Failed to delete graph: {e}")
            
    def predict_relationships(self, graph_id, limit=5):
        """Generate predictions for potential missing relationships in the graph
        
        Args:
            graph_id (int): The knowledge graph ID
            limit (int): Maximum number of predictions to return
            
        Returns:
            list: List of dictionaries containing prediction information
        """
        try:
            # Get graph data
            nodes, relationships = self.get_graph_data(graph_id)
            
            # In a real Neo4j implementation, we would use graph algorithms and pattern matching
            # For this implementation, we'll use a pattern-based approach
            
            # Get node categories and relationship types for the graph
            node_categories = {}
            for node in nodes:
                node_id = node['id']
                # Extract category from label (simple heuristic)
                label = node['label']
                if "Person" in label or "Actor" in label or "Director" in label:
                    category = "Person"
                elif "Movie" in label or "Film" in label:
                    category = "Movie"
                elif "Company" in label or "Studio" in label:
                    category = "Organization"
                elif "Genre" in label:
                    category = "Genre"
                else:
                    category = "Other"
                
                # Add category to node
                node['category'] = category
                
                if category not in node_categories:
                    node_categories[category] = []
                node_categories[category].append(node_id)
            
            # Track existing relationships to avoid duplicates
            existing_relationships = set()
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                rel_type = rel['type']
                existing_relationships.add((source, target, rel_type))
            
            # Find common patterns (e.g., Person-DIRECTED->Movie)
            relationship_patterns = {}
            
            # Count relationship types between categories
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                rel_type = rel['type']
                
                # Find source and target node categories
                source_category = None
                target_category = None
                
                for node in nodes:
                    if node['id'] == source:
                        source_category = node.get('category', 'Unknown')
                    if node['id'] == target:
                        target_category = node.get('category', 'Unknown')
                
                if source_category and target_category:
                    pattern_key = f"{source_category}-{rel_type}->{target_category}"
                    if pattern_key not in relationship_patterns:
                        relationship_patterns[pattern_key] = {
                            'count': 0,
                            'source_category': source_category,
                            'target_category': target_category,
                            'relationship_type': rel_type,
                            'examples': []
                        }
                    
                    relationship_patterns[pattern_key]['count'] += 1
                    if len(relationship_patterns[pattern_key]['examples']) < 5:
                        relationship_patterns[pattern_key]['examples'].append((source, target))
            
            # Sort patterns by frequency (most common first)
            sorted_patterns = sorted(
                relationship_patterns.items(), 
                key=lambda x: x[1]['count'], 
                reverse=True
            )
            
            # Generate predictions based on common patterns
            predictions = []
            
            for _, pattern in sorted_patterns:
                if len(predictions) >= limit:
                    break
                
                source_category = pattern['source_category']
                target_category = pattern['target_category']
                rel_type = pattern['relationship_type']
                
                # Skip if we don't have nodes in these categories
                if (source_category not in node_categories or 
                    target_category not in node_categories):
                    continue
                
                # Get nodes of the relevant categories
                source_nodes = node_categories[source_category]
                target_nodes = node_categories[target_category]
                
                # Find nodes that don't have this relationship pattern yet
                for source_id in source_nodes:
                    if len(predictions) >= limit:
                        break
                        
                    source_node = next((n for n in nodes if n['id'] == source_id), None)
                    if not source_node:
                        continue
                    
                    # Track how many connections of this type the source node already has
                    existing_count = 0
                    for rel in relationships:
                        if rel['source'] == source_id and rel['type'] == rel_type:
                            existing_count += 1
                    
                    # Skip over-connected nodes
                    if existing_count >= 5:  # arbitrary threshold
                        continue
                    
                    # Find potential targets
                    for target_id in target_nodes:
                        # Skip self-connections
                        if source_id == target_id:
                            continue
                            
                        # Skip existing relationships
                        if (source_id, target_id, rel_type) in existing_relationships:
                            continue
                        
                        target_node = next((n for n in nodes if n['id'] == target_id), None)
                        if not target_node:
                            continue
                        
                        # Calculate confidence based on pattern frequency and node relevance
                        # In a real implementation, this would use graph embeddings and ML
                        confidence = min(0.95, (pattern['count'] / len(relationships)) * 2)
                        confidence = round(confidence * 100)
                        
                        # Skip low-confidence predictions
                        if confidence < 60:
                            continue
                        
                        predictions.append({
                            'source': {
                                'id': source_id,
                                'label': source_node['label'],
                                'category': source_category
                            },
                            'target': {
                                'id': target_id,
                                'label': target_node['label'],
                                'category': target_category
                            },
                            'relationship': rel_type,
                            'confidence': confidence
                        })
                        
                        # Limit predictions per pattern
                        if len(predictions) >= limit:
                            break
            
            # Sort predictions by confidence
            predictions = sorted(predictions, key=lambda p: p['confidence'], reverse=True)
            return predictions[:limit]
            
        except Exception as e:
            logging.error(f"Error predicting relationships: {str(e)}")
            return []
