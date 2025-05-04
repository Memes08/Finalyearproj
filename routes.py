import os
import json
import uuid
import logging
import urllib.request
from datetime import datetime

from flask import render_template, flash, redirect, url_for, request, session, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.utils import secure_filename

from app import app, db
from forms import LoginForm, RegistrationForm, NewKnowledgeGraphForm, DataInputForm, QueryForm
from models import User, KnowledgeGraph, InputSource
from whisper_transcriber import WhisperTranscriber
from kg_processor import KnowledgeGraphProcessor
from neo4j_manager import Neo4jGraphManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get global Neo4j manager instance for all routes
neo4j_manager = None
try:
    from neo4j import GraphDatabase
    # Connect to Neo4j (should be configured in environment variables)
    neo4j_manager = Neo4jGraphManager(
        uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        username=os.environ.get('NEO4J_USER', 'neo4j'),
        password=os.environ.get('NEO4J_PASSWORD', 'password')
    )
    logging.info("Neo4j database connected successfully")
except ImportError:
    logging.warning("Neo4j package not installed. Using in-memory storage instead.")
    
    # Simple in-memory implementation for Neo4j when not available
    class InMemoryNeo4jManager:
        def __init__(self):
            self.graphs = {}
            logging.info("Initialized in-memory graph storage")
            
        def import_triples(self, triples, graph_id):
            if graph_id not in self.graphs:
                self.graphs[graph_id] = []
            self.graphs[graph_id].extend(triples)
            logging.info(f"Added {len(triples)} triples to graph {graph_id}")
            return len(triples)
        
        def get_graph_data(self, graph_id):
            if graph_id not in self.graphs:
                return [], []
                
            triples = self.graphs.get(graph_id, [])
            
            nodes = {}
            links = []
            
            # Process triples into nodes and links
            for triple in triples:
                source, relation, target = triple
                
                # Skip if any part is not a string
                if not all(isinstance(x, str) for x in [source, relation]):
                    continue
                    
                # Add source node if not exists
                if source not in nodes:
                    source_type = next((t[2] for t in triples if t[0] == source and t[1] == 'type'), 'Entity')
                    nodes[source] = {
                        'id': source,
                        'label': source,
                        'type': source_type
                    }
                    
                # Add target node if string and not exists
                if isinstance(target, str) and target not in nodes:
                    target_type = next((t[2] for t in triples if t[0] == target and t[1] == 'type'), 'Entity')
                    nodes[target] = {
                        'id': target,
                        'label': target,
                        'type': target_type
                    }
                    
                # Add relationship (only if target is string)
                if isinstance(target, str):
                    links.append({
                        'source': source,
                        'target': target,
                        'type': relation
                    })
                else:
                    # For non-string targets, add as property to source node
                    if 'properties' not in nodes[source]:
                        nodes[source]['properties'] = {}
                    nodes[source]['properties'][relation] = target
            
            return list(nodes.values()), links
            
        def delete_graph(self, graph_id):
            if graph_id in self.graphs:
                del self.graphs[graph_id]
                return True
            return False
            
    # Use in-memory implementation instead
    neo4j_manager = InMemoryNeo4jManager()
except Exception as e:
    logging.error(f"Error initializing Neo4j: {e}")
    flash("Unable to connect to graph database. Some features may be limited.", "warning")
    neo4j_manager = InMemoryNeo4jManager()

# Create the upload folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Utility for filename security
def secure_filename(filename):
    """Make a filename secure by removing path components and strange characters"""
    # Get rid of path components and special characters
    filename = os.path.basename(filename)
    # Only allow alphanumeric, dash, underscore, and dot
    return ''.join(c for c in filename if c.isalnum() or c in '-_.')

@app.route('/')
def index():
    """Landing page showing application overview"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.check_password(form.password.data):
                login_user(user)
                next_page = request.args.get('next')
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(next_page or url_for('dashboard'))
            else:
                flash('Login unsuccessful. Please check your email and password.', 'danger')
        except Exception as e:
            logging.error(f"Database error during login: {e}")
            flash('Unable to access account due to database issues. Please try again later.', 'danger')
            return render_template('db_unavailable.html')
    
    return render_template('login.html', form=form, title='Login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Database error during registration: {e}")
            flash('Unable to register due to database issues. Please try again later.', 'danger')
            return render_template('db_unavailable.html')
    
    return render_template('register.html', form=form, title='Register')

@app.route('/logout')
@login_required
def logout():
    """Log out the current user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing knowledge graphs and options"""
    try:
        knowledge_graphs = KnowledgeGraph.query.filter_by(user_id=current_user.id).all()
        return render_template('dashboard.html', knowledge_graphs=knowledge_graphs)
    except Exception as e:
        logging.error(f"Database error accessing dashboard: {e}")
        flash("Unable to load your dashboard due to database connection issues.", "danger")
        return render_template('db_unavailable.html')

@app.route('/new-graph', methods=['GET', 'POST'])
@login_required
def new_graph():
    """Create a new knowledge graph"""
    form = NewKnowledgeGraphForm()
    
    if form.validate_on_submit():
        try:
            graph = KnowledgeGraph(
                name=form.name.data,
                description=form.description.data,
                domain=form.domain.data,
                user_id=current_user.id
            )
            
            db.session.add(graph)
            db.session.commit()
            
            flash(f'Knowledge graph "{form.name.data}" created successfully!', 'success')
            return redirect(url_for('process_data', graph_id=graph.id))
        except Exception as e:
            logging.error(f"Database error creating graph: {e}")
            flash("Unable to create knowledge graph due to database connection issues.", "danger")
            return render_template('db_unavailable.html')
    
    return render_template('new_graph.html', form=form, title='New Knowledge Graph')

@app.route('/graph/<int:graph_id>/process', methods=['GET', 'POST'])
@login_required
def process_data(graph_id):
    """Process data for a knowledge graph"""
    # For both GET and POST requests, use the github_url_page function
    # which provides a simplified interface focused on GitHub URLs
    return github_url_page(graph_id)
    
@app.route('/graph/<int:graph_id>/github', methods=['GET', 'POST'])
@login_required
def github_url_page(graph_id):
    """Process GitHub URL for knowledge graph"""
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
    
    # Handle GET request - just show the form
    if request.method == 'GET':
        return render_template('github_url_only.html', form=form, graph=graph)
    
    # Handle POST request - process the GitHub URL
    github_url = request.form.get('github_url')
    
    if not github_url:
        flash("Please enter a GitHub URL", 'danger')
        return render_template('github_url_only.html', form=form, graph=graph)
    
    # Create a unique directory for this input
    process_id = str(uuid.uuid4())
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
    os.makedirs(input_dir, exist_ok=True)
    
    # Create progress tracking entry
    process_key = f"{graph.id}_{process_id}"
    progress_key = f"progress_{process_key}"
    session[progress_key] = {
        'status': 'Starting data processing...',
        'percent': 5,
        'step': 'init'
    }
    session.modified = True
    
    # Store the process key for use in templates
    session['current_process_id'] = process_key
    
    # Create an input source
    input_source = InputSource(
        source_type='url',
        url=github_url,
        knowledge_graph=graph
    )
    
    try:
        # Download file from URL
        try:
            filename = github_url.split('/')[-1]
            file_path = os.path.join(input_dir, filename)
            
            # Update progress
            session[progress_key] = {
                'status': f"Downloading file from URL: {github_url}",
                'percent': 20,
                'step': 'upload'
            }
            session.modified = True
            
            logging.info(f"Downloading file from URL: {github_url} to {file_path}")
            urllib.request.urlretrieve(github_url, file_path)
            logging.info(f"Successfully downloaded file from URL")
            
            # Update progress
            session[progress_key] = {
                'status': 'Processing file...',
                'percent': 40,
                'step': 'extraction'
            }
            session.modified = True
        except Exception as e:
            logging.error(f"Error downloading file from URL: {e}")
            flash(f"Error downloading file: {str(e)}", 'danger')
            return render_template('github_url_only.html', form=form, graph=graph)
        
        # Process the file as a CSV
        if filename.lower().endswith('.csv'):
            # Process the CSV file
            processor = KnowledgeGraphProcessor(neo4j_manager)
            
            # Update progress
            session[progress_key] = {
                'status': 'Extracting entities from CSV...',
                'percent': 60,
                'step': 'extraction'
            }
            session.modified = True
            
            triples = processor.process_csv(file_path, domain=graph.domain)
            
            if not triples:
                flash("No entities or relationships could be extracted from the CSV file", 'warning')
                return render_template('github_url_only.html', form=form, graph=graph)
            
            # Update progress
            session[progress_key] = {
                'status': 'Importing to graph database...',
                'percent': 80,
                'step': 'database'
            }
            session.modified = True
            
            # Add triples to Neo4j
            neo4j_manager.import_triples(triples, graph.id)
            
            # Update input source details
            input_source.filename = filename
            input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples if isinstance(t[2], str)]))
            input_source.relationship_count = len(triples)
            input_source.processed = True
            
            # Save the input source
            db.session.add(input_source)
            db.session.commit()
            
            # Update progress
            session[progress_key] = {
                'status': 'Processing complete!',
                'percent': 100,
                'step': 'complete'
            }
            session.modified = True
            
            # Redirect to visualization
            flash(f'Successfully processed CSV file from GitHub URL and added {input_source.entity_count} entities with {input_source.relationship_count} relationships to your knowledge graph.', 'success')
            return redirect(url_for('visualization', graph_id=graph.id))
        
        else:
            flash("The file must be a CSV file", 'danger')
            return render_template('github_url_only.html', form=form, graph=graph)
    
    except Exception as e:
        logging.error(f"Error processing GitHub URL: {e}")
        flash(f"Error processing file: {str(e)}", 'danger')
        
        # Update progress with error
        session[progress_key] = {
            'status': f'Error: {str(e)}',
            'percent': 0,
            'step': 'error'
        }
        session.modified = True
        
        return render_template('github_url_only.html', form=form, graph=graph)

@app.route('/video_summary/<int:graph_id>/<process_id>', methods=['GET'])
@login_required
def video_summary(graph_id, process_id):
    """Show video transcription summary and allow user to approve/reject"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))
        
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
    
    # Check if the directory exists
    if not os.path.isdir(input_dir):
        flash('Process data not found. Please try again.', 'danger')
        return redirect(url_for('process_data', graph_id=graph_id))
        
    # Load transcription or text content to display
    transcription = ""
    transcription_path = os.path.join(input_dir, 'transcription.txt')
    if os.path.exists(transcription_path):
        with open(transcription_path, 'r') as f:
            transcription = f.read()
    
    # Load preview triples
    triples = []
    preview_path = os.path.join(input_dir, 'preview_triples.json')
    if os.path.exists(preview_path):
        with open(preview_path, 'r') as f:
            triples = json.load(f)
    
    # Load metadata for summary stats
    metadata = {}
    metadata_path = os.path.join(input_dir, 'summary_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Extract YouTube URL if available
    youtube_url = ""
    url_path = os.path.join(input_dir, 'youtube_url.txt')
    if os.path.exists(url_path):
        with open(url_path, 'r') as f:
            youtube_url = f.read().strip()
    
    return render_template(
        'video_summary.html',
        graph=graph,
        process_id=process_id,
        transcription=transcription,
        triples=triples[:100],  # Limit to 100 for display
        triple_count=len(triples),
        metadata=metadata,
        youtube_url=youtube_url
    )

@app.route('/approve_video/<int:graph_id>/<process_id>', methods=['POST'])
@login_required
def approve_video(graph_id, process_id):
    """Approve video processing and create knowledge graph"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))
    
    try:
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
        
        # Load preview triples
        preview_path = os.path.join(input_dir, 'preview_triples.json')
        if not os.path.exists(preview_path):
            flash('Process data not found. Please try again.', 'danger')
            return redirect(url_for('process_data', graph_id=graph_id))
            
        with open(preview_path, 'r') as f:
            triples = json.load(f)
        
        # Find the input source record
        input_source = InputSource.query.filter_by(
            graph_id=graph_id, 
            processed=False
        ).order_by(InputSource.created_at.desc()).first()
        
        if not input_source:
            # Create a new one if we can't find the original
            input_source = InputSource(
                source_type='video',
                knowledge_graph=graph
            )
        
        # Save CSV for reference
        csv_path = os.path.join(input_dir, 'extracted_triples.csv')
        processor = KnowledgeGraphProcessor(neo4j_manager)
        processor.save_triples_to_csv(triples, csv_path)
        
        # Import triples to Neo4j
        neo4j_manager.import_triples(triples, graph.id)
        
        # Update input source as processed
        input_source.processed = True
        input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples if isinstance(t[2], str)]))
        input_source.relationship_count = len(triples)
        
        # Save the input source
        db.session.add(input_source)
        db.session.commit()
        
        flash('Knowledge graph created successfully!', 'success')
        return redirect(url_for('visualization', graph_id=graph.id))
    except Exception as e:
        logging.error(f"Error approving video: {e}")
        flash(f'Error approving video: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/reject_video/<int:graph_id>/<process_id>', methods=['POST'])
@login_required
def reject_video(graph_id, process_id):
    """Reject video processing and return to data input"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))
        
    # Find the input source record and delete it
    input_source = InputSource.query.filter_by(
        graph_id=graph_id, 
        processed=False
    ).order_by(InputSource.created_at.desc()).first()
    
    if input_source:
        db.session.delete(input_source)
        db.session.commit()
    
    flash('Video processing has been rejected. You can try again with a different source.', 'info')
    return redirect(url_for('process_data', graph_id=graph.id))

@app.route('/graph/<int:graph_id>/visualization')
@login_required
def visualization(graph_id):
    """Visualize a knowledge graph"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
            
        # Get the input sources for this graph
        input_sources = InputSource.query.filter_by(graph_id=graph.id).all()
        
        return render_template('visualization.html', graph=graph, input_sources=input_sources)
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/get_processing_progress/<process_id>')
def get_processing_progress(process_id):
    """Get the current progress of a data processing job"""
    # Process ID format: {graph_id}_{uuid}
    progress_key = f"progress_{process_id}"
    
    if progress_key in session:
        progress_data = session[progress_key]
        return jsonify(progress_data)
    
    # Default response if no progress data found
    return jsonify({
        'status': 'Progress data not found',
        'percent': 0,
        'step': 'unknown'
    })

@app.route('/graph/<int:graph_id>/query', methods=['GET', 'POST'])
@login_required
def query_graph(graph_id):
    """Query a knowledge graph using natural language"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))
    
    form = QueryForm()
    results = []
    
    if form.validate_on_submit():
        processor = KnowledgeGraphProcessor(neo4j_manager)
        results = processor.query_knowledge_graph(form.query.data, graph.id)
        
        if not results:
            flash("No results found for your query. Try rephrasing or using different keywords.", "warning")
        else:
            flash(f"Found {len(results)} results for your query.", "success")
    
    return render_template('query.html', form=form, graph=graph, results=results)

@app.route('/api/graph/<int:graph_id>/data')
@login_required
def graph_data(graph_id):
    """Get graph data for visualization"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            return jsonify({'error': 'Access denied'}), 403
    
        # Get graph data from Neo4j
        nodes, links = neo4j_manager.get_graph_data(graph.id)
        return jsonify({
            'nodes': nodes,
            'links': links
        })
    except Exception as e:
        logging.error(f"Error getting graph data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    return render_template('profile.html', user=current_user, knowledge_graphs=KnowledgeGraph.query.filter_by(user_id=current_user.id).all())

@app.route('/graph/<int:graph_id>/delete', methods=['POST'])
@login_required
def delete_graph(graph_id):
    """Delete a knowledge graph"""
    try:
        graph = KnowledgeGraph.query.get_or_404(graph_id)
        
        # Check if graph belongs to current user
        if graph.user_id != current_user.id:
            flash('Access denied. You do not own this knowledge graph.', 'danger')
            return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Database error accessing graph {graph_id}: {e}")
        flash("Unable to access knowledge graph due to database connection issues.", "danger")
        return redirect(url_for('dashboard'))
        
    try:
        # Delete from Neo4j
        neo4j_manager.delete_graph(graph.id)
        
        # Delete all input sources
        InputSource.query.filter_by(graph_id=graph.id).delete()
        
        # Delete the graph
        db.session.delete(graph)
        db.session.commit()
        
        flash('Knowledge graph deleted successfully.', 'success')
    except Exception as e:
        logging.error(f"Error deleting graph {graph_id}: {e}")
        flash('Error deleting knowledge graph. Please try again.', 'danger')
    
    return redirect(url_for('dashboard'))