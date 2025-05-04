import os
import logging
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import urllib.request
import tempfile

from models import KnowledgeGraph, InputSource, db
from github_url_processor import process_github_url
from forms import DataInputForm
from app import app
from kg_processor import KnowledgeGraphProcessor

# Create a Blueprint for GitHub URL processing routes
github_url_bp = Blueprint('github_url', __name__)

# Get Neo4j and KG processor instances
try:
    from neo4j_manager import Neo4jGraphManager
    neo4j_manager = Neo4jGraphManager(None, None, None)  # In-memory fallback
except ImportError:
    logging.warning("Neo4j package not installed. Using in-memory storage instead.")
    class MockNeo4jManager:
        def import_triples(self, triples, graph_id):
            logging.info(f"Importing {len(triples)} triples to in-memory storage for graph {graph_id}")
            return True
            
        def get_graph_data(self, graph_id):
            logging.info(f"Getting mock graph data for graph {graph_id}")
            return [], []
            
        def delete_graph(self, graph_id):
            logging.info(f"Deleting mock graph {graph_id}")
            return True
    
    neo4j_manager = MockNeo4jManager()

# Initialize KG processor
kg_processor = KnowledgeGraphProcessor(neo4j_manager)

@github_url_bp.route('/graph/<int:graph_id>/github_url', methods=['GET', 'POST'])
@login_required
def process_github_url_route(graph_id):
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return render_template('db_unavailable.html')
    
    form = DataInputForm()
    
    # Handle form submission
    if request.method == 'POST':
        github_url = request.form.get('github_url')
        
        if not github_url:
            flash("Please enter a GitHub URL", 'danger')
            return render_template('github_url_process.html', form=form, graph=graph)
        
        # Process the GitHub URL
        success, entity_count, relationship_count = process_github_url(
            github_url, graph_id, kg_processor, neo4j_manager, domain=graph.domain)
        
        if success:
            flash(f"Successfully processed GitHub URL and added {entity_count} entities and {relationship_count} relationships to the knowledge graph", 'success')
            return redirect(url_for('visualization', graph_id=graph_id))
        else:
            return render_template('github_url_process.html', form=form, graph=graph)
    
    # GET request - display the form
    return render_template('github_url_process.html', form=form, graph=graph)

# Register the blueprint with the app
app.register_blueprint(github_url_bp)