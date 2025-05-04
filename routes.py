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
from whisper_transcriber import WhisperTranscriber

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

# Initialize Whisper transcriber
whisper_transcriber = WhisperTranscriber()


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
    
    # Check if we're getting a custom selected_input_type field from our form
    selected_input_type = request.form.get('selected_input_type')
    if selected_input_type and selected_input_type in ['video', 'csv', 'url']:
        form.input_type.data = selected_input_type
    
    if form.validate_on_submit():
        # Create a unique directory for this input
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
        os.makedirs(input_dir, exist_ok=True)
        
        # Log the form data to debug
        logging.info(f"Form data - input_type: {form.input_type.data}")
        if form.input_type.data == 'url':
            logging.info(f"GitHub URL: {form.github_url.data}")
        
        input_source = InputSource(
            source_type=form.input_type.data,
            knowledge_graph=graph
        )
        
        try:
            if form.input_type.data == 'video':
                # Check if video processing is available
                if not app.config.get('HAS_VIDEO_PROCESSING', False):
                    flash('Video processing is currently unavailable. Please install the required packages.', 'warning')
                    return render_template('process.html', form=form, graph=graph)
                
                # Save video file
                video_file = form.video_file.data
                filename = secure_filename(video_file.filename)
                video_path = os.path.join(input_dir, filename)
                video_file.save(video_path)
                
                # Transcribe video
                audio_path = os.path.join(input_dir, 'audio.wav')
                transcription = whisper_transcriber.transcribe_video(video_path, audio_path)
                
                # Save output as CSV for Neo4j
                csv_path = os.path.join(input_dir, 'transcription_triples.csv')
                triples = kg_processor.extract_triples_from_text(transcription, domain=graph.domain)
                kg_processor.save_triples_to_csv(triples, csv_path)
                
                # Update input source details
                input_source.filename = filename
                input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples]))
                input_source.relationship_count = len(triples)
                
                # Insert into Neo4j
                neo4j_manager.import_triples(triples, graph.id)
                
            elif form.input_type.data == 'csv':
                # Check if pandas is available
                if not app.config.get('HAS_PANDAS', False):
                    flash('CSV processing requires pandas, which is currently unavailable.', 'warning')
                    return render_template('process.html', form=form, graph=graph)
                
                # Save CSV file
                csv_file = form.csv_file.data
                filename = secure_filename(csv_file.filename)
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
    
    if form.validate_on_submit():
        query_text = form.query.data
        try:
            result = kg_processor.query_knowledge_graph(query_text, graph.id)
        except NotImplementedError as e:
            flash(f'Feature not available: {str(e)}', 'warning')
        except Exception as e:
            flash(f'Error querying knowledge graph: {str(e)}', 'danger')
    
    return render_template('query.html', form=form, graph=graph, result=result)


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


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', KnowledgeGraph=KnowledgeGraph)


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
