import os
import logging
import urllib.request
import tempfile
import pandas as pd

from flask import flash
from models import InputSource, db


def process_github_url(url, graph_id, kg_processor, neo4j_manager, domain="movie"):
    """
    Process a GitHub URL to a CSV file and create a knowledge graph
    
    Args:
        url: The URL to the raw CSV file on GitHub
        graph_id: The ID of the knowledge graph to add data to
        kg_processor: The KnowledgeGraphProcessor instance
        neo4j_manager: The Neo4jGraphManager instance
        domain: The domain of the knowledge graph
        
    Returns:
        Tuple of (success, entity_count, relationship_count)
    """
    # Create a temporary directory for downloaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Get the filename from the URL
            filename = url.split('/')[-1]
            if not filename.endswith('.csv'):
                flash("The URL must point to a CSV file", 'danger')
                return False, 0, 0
                
            # Download the file
            file_path = os.path.join(temp_dir, filename)
            logging.info(f"Downloading file from URL: {url} to {file_path}")
            urllib.request.urlretrieve(url, file_path)
            logging.info(f"Successfully downloaded file from GitHub URL")
            
            # Process the CSV file
            logging.info(f"Processing CSV file with domain: {domain}")
            triples = kg_processor.process_csv(file_path, domain)
            
            if not triples:
                flash("No entities or relationships could be extracted from the CSV file", 'warning')
                return False, 0, 0
                
            logging.info(f"Extracted {len(triples)} triples from CSV")
            
            # Save the triples to the graph database
            neo4j_manager.import_triples(triples, graph_id)
            
            # Create a record of the input source
            input_source = InputSource(
                source_type='url',
                filename=filename,
                url=url,
                processed=True,
                entity_count=len(set([t[0] for t in triples] + [t[2] for t in triples])),
                relationship_count=len(triples),
                graph_id=graph_id
            )
            
            db.session.add(input_source)
            db.session.commit()
            
            return True, input_source.entity_count, input_source.relationship_count
            
        except Exception as e:
            logging.error(f"Error processing GitHub URL: {e}")
            flash(f"Error processing file: {str(e)}", 'danger')
            return False, 0, 0