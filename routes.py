import os
import uuid
import logging
import csv
from datetime import datetime
import urllib.request

# Import werkzeug safely
try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename):
        # Simple fallback implementation to remove dangerous characters
        return ''.join(c for c in filename if c.isalnum() or c in '._-')
    logging.warning("Werkzeug not installed. Using simplified secure_filename.")

# Import Flask modules
try:
    from flask import render_template, redirect, url_for, flash, request, jsonify, send_file
    from flask_login import login_user, logout_user, current_user, login_required
    HAS_FLASK = True
except ImportError:
    logging.error("Flask not installed. Application will not run.")
    HAS_FLASK = False

from app import app, db
from models import User, KnowledgeGraph, InputSource
from forms import LoginForm, RegistrationForm, NewKnowledgeGraphForm, DataInputForm, QueryForm
from neo4j_manager import Neo4jGraphManager
from kg_processor import KnowledgeGraphProcessor

# Import pandas safely
try:
    import pandas as pd
    app.config['HAS_PANDAS'] = True
except ImportError:
    app.config['HAS_PANDAS'] = False
    logging.warning("Pandas not installed. CSV processing will be limited.")

# Initialize Neo4j manager
neo4j_manager = Neo4jGraphManager(
    uri=app.config['NEO4J_URI'],
    username=app.config['NEO4J_USERNAME'],
    password=app.config['NEO4J_PASSWORD']
)

# Initialize knowledge graph processor
kg_processor = KnowledgeGraphProcessor(
    neo4j_manager=neo4j_manager,
    groq_api_key=app.config['GROQ_API_KEY']
)


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login failed. Please check your email and password.', 'danger')
    
    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    knowledge_graphs = KnowledgeGraph.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', knowledge_graphs=knowledge_graphs)


@app.route('/graph/new', methods=['GET', 'POST'])
@login_required
def new_graph():
    form = NewKnowledgeGraphForm()
    if form.validate_on_submit():
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
    
    return render_template('new_graph.html', form=form)


@app.route('/graph/<int:graph_id>/process', methods=['GET', 'POST'])
@login_required
def process_data(graph_id):
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        flash('Access denied. You do not own this knowledge graph.', 'danger')
        return redirect(url_for('dashboard'))
    
    form = DataInputForm()
    if form.validate_on_submit():
        # Create a unique directory for this input
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
        os.makedirs(input_dir, exist_ok=True)
        
        input_source = InputSource(
            source_type=form.input_type.data,
            knowledge_graph=graph
        )
        
        try:
            if form.input_type.data == 'csv':
                # Check if pandas is available
                if not app.config.get('HAS_PANDAS', False):
                    flash('CSV processing requires pandas, which is currently unavailable.', 'warning')
                    return render_template('process.html', form=form, graph=graph)
                
                # Save CSV file
                csv_file = form.csv_file.data
                filename = secure_filename(csv_file.filename)
                
                # Check if this file already exists for this graph
                existing_source = InputSource.query.filter_by(
                    graph_id=graph.id,
                    filename=filename,
                    source_type='csv'
                ).first()
                
                if existing_source:
                    flash(f'This file ({filename}) has already been processed for this knowledge graph.', 'warning')
                    return render_template('process.html', form=form, graph=graph)
                
                csv_path = os.path.join(input_dir, filename)
                csv_file.save(csv_path)
                
                # Process CSV file
                triples = kg_processor.process_csv(csv_path, domain=graph.domain)
                
                # Update input source details
                input_source.filename = filename
                input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples]))
                input_source.relationship_count = len(triples)
                
                # Insert into Neo4j
                neo4j_manager.import_triples(triples, graph.id)
                
            elif form.input_type.data == 'url':
                # Check if pandas is available
                if not app.config.get('HAS_PANDAS', False):
                    flash('URL processing requires pandas, which is currently unavailable.', 'warning')
                    return render_template('process.html', form=form, graph=graph)
                
                # Download CSV from URL
                url = form.github_url.data
                if url:
                    try:
                        filename = url.split('/')[-1]
                        
                        # Check if this URL already exists for this graph
                        existing_source = InputSource.query.filter_by(
                            graph_id=graph.id,
                            url=url,
                            source_type='url'
                        ).first()
                        
                        if existing_source:
                            flash(f'This URL has already been processed for this knowledge graph.', 'warning')
                            return render_template('process.html', form=form, graph=graph)
                            
                        csv_path = os.path.join(input_dir, filename)
                        urllib.request.urlretrieve(url, csv_path)
                    except Exception as e:
                        logging.error(f"Error downloading CSV from URL: {e}")
                        flash(f"Error downloading CSV: {str(e)}", 'danger')
                        return render_template('process.html', form=form, graph=graph)
                else:
                    flash("No GitHub URL provided", 'danger')
                    return render_template('process.html', form=form, graph=graph)
                
                # Process CSV file
                triples = kg_processor.process_csv(csv_path, domain=graph.domain)
                
                # Update input source details
                input_source.url = url
                input_source.filename = filename
                input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples]))
                input_source.relationship_count = len(triples)
                
                # Insert into Neo4j
                neo4j_manager.import_triples(triples, graph.id)
            
            input_source.processed = True
            db.session.add(input_source)
            db.session.commit()
            
            flash('Data processed and added to knowledge graph successfully!', 'success')
            return redirect(url_for('visualization', graph_id=graph.id))
            
        except NotImplementedError as e:
            flash(f'Feature not available: {str(e)}', 'warning')
            logging.warning(f"Not implemented: {e}")
        except Exception as e:
            flash(f'Error processing data: {str(e)}', 'danger')
            logging.error(f"Error processing data: {e}", exc_info=True)
    
    return render_template('process.html', form=form, graph=graph)


@app.route('/graph/<int:graph_id>/visualize')
@login_required
def visualization(graph_id):
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        flash('Access denied. You do not own this knowledge graph.', 'danger')
        return redirect(url_for('dashboard'))
    
    input_sources = InputSource.query.filter_by(graph_id=graph.id).all()
    
    return render_template('visualization.html', graph=graph, input_sources=input_sources)


@app.route('/graph/<int:graph_id>/query', methods=['GET', 'POST'])
@login_required
def query_graph(graph_id):
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        flash('Access denied. You do not own this knowledge graph.', 'danger')
        return redirect(url_for('dashboard'))
    
    form = QueryForm()
    result = None
    has_result = False  # Add a flag for templates to check
    
    if form.validate_on_submit():
        query_text = form.query.data
        try:
            result = kg_processor.query_knowledge_graph(query_text, graph.id)
            has_result = result is not None  # Set flag when there's a result
        except NotImplementedError as e:
            flash(f'Feature not available: {str(e)}', 'warning')
        except Exception as e:
            flash(f'Error querying knowledge graph: {str(e)}', 'danger')
    
    return render_template('query.html', form=form, graph=graph, result=result, has_result=has_result)


@app.route('/graph/<int:graph_id>/data')
@login_required
def graph_data(graph_id):
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        return jsonify({"error": "Access denied"}), 403
    
    try:
        nodes, relationships = neo4j_manager.get_graph_data(graph.id)
        return jsonify({
            "nodes": nodes,
            "relationships": relationships
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/graph/<int:graph_id>/analytics')
@login_required
def graph_analytics(graph_id):
    """Analytics dashboard for a knowledge graph"""
    # Fetch the graph
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        flash('You do not have permission to view this graph.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Get graph data for analytics
        nodes, relationships = neo4j_manager.get_graph_data(graph.id)
        data = {"nodes": nodes, "relationships": relationships}
        
        # Calculate centrality for top nodes
        centrality_scores = calculate_centrality(data)
        top_nodes = []
        
        # Sort nodes by centrality
        node_centrality_pairs = list(centrality_scores.items())
        node_centrality_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10 or fewer nodes
        for node_id, score in node_centrality_pairs[:10]:
            for node in nodes:
                if node['id'] == node_id:
                    node_info = {
                        'id': node_id,
                        'label': node.get('label', 'Unknown'),
                        'category': node.get('category', 'Other'),
                        'centrality': score
                    }
                    top_nodes.append(node_info)
                    break
        
        # Count categories
        category_counts = {}
        for node in nodes:
            category = node.get('category', 'Other')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count relationship types
        relationship_counts = {}
        for rel in relationships:
            rel_type = rel.get('type', 'Other')
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        # Prepare graph data for the template and export functionality
        graph_data = {
            'node_count': len(nodes),
            'relationship_count': len(relationships),
            'density': calculate_graph_density(data),
            'avg_degree': sum(centrality_scores.values()) / max(1, len(centrality_scores)),
            'avg_path_length': 0,  # Would require more complex computation
            'isolated_nodes': sum(1 for score in centrality_scores.values() if score == 0),
            'clustering_coefficient': 0,  # Would require more complex computation
            'central_nodes': top_nodes,
            'category_counts': category_counts,
            'relationship_counts': relationship_counts,
            'communities': detect_communities(data),
            'domain': graph.domain,
        }
        
        return render_template('analytics_dark.html', graph=graph, graph_data=graph_data)
    except Exception as e:
        flash(f'Error generating analytics: {str(e)}', 'danger')
        return redirect(url_for('visualization', graph_id=graph_id))

@app.route('/graph/<int:graph_id>/predictions')
@login_required
def graph_predictions(graph_id):
    """Get predictions for missing relationships in a knowledge graph"""
    # Fetch the graph
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        return jsonify({"error": "Access denied"}), 403
    
    try:
        # Generate predictions
        limit = request.args.get('limit', 5, type=int)
        predictions = neo4j_manager.predict_relationships(graph.id, limit=limit)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def calculate_graph_density(data):
    """Calculate graph density (ratio of actual connections to potential connections)"""
    nodes = len(data['nodes'])
    edges = len(data['relationships'])
    
    if nodes <= 1:
        return 0
    
    # Maximum possible edges in a directed graph: n(n-1)
    max_edges = nodes * (nodes - 1)
    if max_edges == 0:
        return 0
        
    return round((edges / max_edges) * 100, 2)

def calculate_centrality(data):
    """Calculate degree centrality for nodes (simplified version)"""
    # Count connections for each node
    centrality = {}
    for node in data['nodes']:
        centrality[node['id']] = 0
    
    # Count actual connections
    for rel in data['relationships']:
        source_id = rel['source']
        target_id = rel['target']
        
        if source_id in centrality:
            centrality[source_id] += 1
        
        if target_id in centrality:
            centrality[target_id] += 1
    
    return centrality

def detect_communities(data):
    """Simplified community detection based on node categories"""
    communities = {}
    
    for node in data['nodes']:
        category = node.get('category', 'Unknown')
        if category not in communities:
            communities[category] = 0
        communities[category] += 1
    
    # Convert to sorted list
    community_list = [
        {'name': k, 'count': v} 
        for k, v in communities.items()
    ]
    return sorted(community_list, key=lambda x: x['count'], reverse=True)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    from forms import ProfileUpdateForm
    
    form = ProfileUpdateForm(
        original_username=current_user.username,
        original_email=current_user.email
    )
    
    # Pre-populate form with current user data
    if request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    
    if form.validate_on_submit():
        # Update username and email
        current_user.username = form.username.data
        current_user.email = form.email.data
        
        # Update password if provided
        if form.current_password.data and form.new_password.data:
            # Verify current password
            if current_user.check_password(form.current_password.data):
                current_user.set_password(form.new_password.data)
                flash('Your password has been updated.', 'success')
            else:
                flash('Current password is incorrect.', 'danger')
                return redirect(url_for('profile'))
        
        # Save changes
        db.session.commit()
        flash('Your profile has been updated.', 'success')
        return redirect(url_for('profile'))
    
    # Get user's knowledge graphs
    user_graphs = KnowledgeGraph.query.filter_by(user_id=current_user.id).all()
    
    # Get account statistics
    stats = {
        'total_graphs': len(user_graphs),
        'total_entities': sum([sum([s.entity_count for s in graph.input_sources]) for graph in user_graphs]),
        'total_relationships': sum([sum([s.relationship_count for s in graph.input_sources]) for graph in user_graphs]),
        'member_since': current_user.created_at,
        'last_login': current_user.last_login
    }
    
    return render_template('profile.html', form=form, stats=stats, graphs=user_graphs)


@app.route('/graph/<int:graph_id>/delete', methods=['POST'])
@login_required
def delete_graph(graph_id):
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Check if graph belongs to current user
    if graph.user_id != current_user.id:
        flash('Access denied. You do not own this knowledge graph.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Delete from Neo4j
        neo4j_manager.delete_graph(graph.id)
        
        # Delete related input sources
        InputSource.query.filter_by(graph_id=graph.id).delete()
        
        # Delete graph record
        db.session.delete(graph)
        db.session.commit()
        
        flash('Knowledge graph deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting knowledge graph: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))
