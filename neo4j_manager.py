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
