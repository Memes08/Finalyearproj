import os
import uuid
import logging
import csv
import re
from datetime import datetime
import urllib.request
import json

# Import werkzeug safely
try:
    from werkzeug.utils import secure_filename
except ImportError:
    def secure_filename(filename):
        # Simple fallback implementation to remove dangerous characters
        return ''.join(c for c in filename if c.isalnum() or c in '._-')
    logging.warning("Werkzeug not installed. Using simplified secure_filename.")
    
# Utility function to save files in chunks to prevent timeouts
def save_file_chunked(file_storage, destination_path, chunk_size=4*1024*1024, progress_callback=None):
    """
    Save a file in chunks to prevent timeouts with large files
    
    Args:
        file_storage: The FileStorage object from the form
        destination_path: Where to save the file
        chunk_size: Size of chunks in bytes (default 4MB)
        progress_callback: Optional callback function to report progress
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logging.info(f"Starting chunked save to: {destination_path}")
        bytes_written = 0
        
        with open(destination_path, 'wb') as f:
            # Reset file pointer to beginning
            file_storage.stream.seek(0)
            
            # Read and write in chunks
            chunk = file_storage.read(chunk_size)
            while chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                
                # Report progress if callback provided
                if progress_callback:
                    progress_callback(bytes_written)
                    
                # Get next chunk
                chunk = file_storage.read(chunk_size)
                
        logging.info(f"Successfully saved file: {os.path.basename(destination_path)} ({bytes_written/1024/1024:.2f}MB)")
        return True
        
    except Exception as e:
        logging.error(f"Error in chunked file save: {str(e)}")
        return False

# Import Flask modules
try:
    from flask import render_template, redirect, url_for, flash, request, jsonify, send_file, session, abort
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

# Set video processing flag based on Whisper availability
app.config['HAS_VIDEO_PROCESSING'] = whisper_transcriber.has_whisper
logging.info(f"Video processing with Whisper is {'available' if app.config['HAS_VIDEO_PROCESSING'] else 'unavailable'}")


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
    if selected_input_type and selected_input_type in ['video', 'youtube', 'csv', 'url']:
        form.input_type.data = selected_input_type
    
    if form.validate_on_submit():
        # Create a unique directory for this input
        process_id = str(uuid.uuid4())
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
        os.makedirs(input_dir, exist_ok=True)
        
        # Log the form data to debug
        logging.info(f"Form data - input_type: {form.input_type.data}")
        if form.input_type.data == 'url':
            logging.info(f"GitHub URL: {form.github_url.data}")
        elif form.input_type.data == 'youtube':
            logging.info(f"YouTube URL: {form.youtube_url.data}")
        
        # Create progress tracking entry with a consistent ID format
        # This must match the format used in the JavaScript to fetch progress correctly
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
        
        input_source = InputSource(
            source_type=form.input_type.data,
            knowledge_graph=graph
        )
        
        try:
            if form.input_type.data == 'youtube':
                # Process YouTube URL
                youtube_url = form.youtube_url.data
                
                # Update progress
                session[progress_key] = {
                    'status': 'Processing YouTube video link...',
                    'percent': 15,
                    'step': 'download'
                }
                session.modified = True
                
                # Store the URL in the input source
                input_source.url = youtube_url
                
                # Create Whisper transcriber for processing
                whisper_transcriber = WhisperTranscriber()
                
                if not whisper_transcriber.has_pytube:
                    flash('YouTube processing is not available. PyTube is not installed.', 'danger')
                    return render_template('process.html', form=form, graph=graph, current_process_id=session.get('current_process_id'))
                
                # Update progress
                session[progress_key] = {
                    'status': 'Fetching YouTube video transcript...',
                    'percent': 30,
                    'step': 'transcription'
                }
                session.modified = True
                
                try:
                    # Get transcript from YouTube (will try captions first, then download and transcribe)
                    youtube_transcript = whisper_transcriber.get_youtube_transcript(youtube_url, input_dir)
                    
                    # Update progress
                    session[progress_key] = {
                        'status': 'Transcript obtained, processing knowledge graph...',
                        'percent': 60,
                        'step': 'processing'
                    }
                    session.modified = True
                    
                    # Save transcript to file for the summary page
                    transcription_path = os.path.join(input_dir, 'transcription.txt')
                    with open(transcription_path, 'w') as f:
                        f.write(youtube_transcript)
                    
                    # Use the processor to extract entities and relationships
                    processor = KnowledgeGraphProcessor(neo4j_manager)
                    triples = processor.extract_triples_from_text(youtube_transcript, domain=graph.domain)
                    
                    # If we got empty results, try a fallback approach
                    if not triples or len(triples) < 3:
                        logging.warning("Minimal triples detected, using fallback extraction")
                        
                        # Try extracting with patterns
                        triples = processor.extract_entities_with_patterns(youtube_transcript, domain=graph.domain)
                        
                        # If still empty, create some basic triples from the URL
                        if not triples or len(triples) < 3:
                            # Extract video ID from URL
                            video_id = None
                            if 'youtube.com/watch?v=' in youtube_url:
                                video_id = youtube_url.split('youtube.com/watch?v=')[1].split('&')[0]
                            elif 'youtu.be/' in youtube_url:
                                video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
                            
                            if video_id:
                                triples = [
                                    (f"youtube_{video_id}", "type", "Video"),
                                    (f"youtube_{video_id}", "source", "YouTube"),
                                    (f"youtube_{video_id}", "url", youtube_url),
                                    (f"youtube_{video_id}", "youtube_id", video_id),
                                    (f"youtube_{video_id}", "processed_date", datetime.now().strftime('%Y-%m-%d')),
                                    (f"youtube_{video_id}", "knowledge_graph", f"{graph.name}")
                                ]
                    
                    # Save preview triples for later user approval
                    preview_path = os.path.join(input_dir, 'preview_triples.json')
                    with open(preview_path, 'w') as f:
                        json.dump(triples, f)
                    
                    # Save the URL for the summary page
                    url_path = os.path.join(input_dir, 'youtube_url.txt')
                    with open(url_path, 'w') as f:
                        f.write(youtube_url)
                    
                    # Save metadata for summary page
                    summary_metadata = {
                        "detected_domain": graph.domain,
                        "content_type": "YouTube Video",
                        "entity_count": len(set([t[0] for t in triples] + [t[2] for t in triples if not isinstance(t[2], (int, float))])),
                        "relation_count": len(triples),
                        "processing_method": "YouTube API + Whisper Transcription"
                    }
                    metadata_path = os.path.join(input_dir, 'summary_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(summary_metadata, f)
                    
                    # Update progress - ready for approval
                    session[progress_key] = {
                        'status': 'Ready for approval...',
                        'percent': 90,
                        'step': 'approval'
                    }
                    session.modified = True
                    
                    # Update input source details
                    input_source.entity_count = summary_metadata["entity_count"]
                    input_source.relationship_count = summary_metadata["relation_count"]
                    input_source.processed = False
                    
                    # Save input source
                    db.session.add(input_source)
                    db.session.commit()
                    
                    # Redirect to the summary approval page like the main flow
                    return redirect(url_for('video_summary', graph_id=graph.id, process_id=process_id))
                
                except Exception as e:
                    logging.error(f"Error processing YouTube video: {str(e)}")
                    session[progress_key] = {
                        'status': f'Error processing YouTube video: {str(e)}',
                        'percent': 0,
                        'step': 'error'
                    }
                    session.modified = True
                    flash(f'Error processing YouTube video: {str(e)}', 'danger')
                    return render_template('process.html', form=form, graph=graph, current_process_id=session.get('current_process_id'))
                
            elif form.input_type.data == 'video':
                # Check if video processing is available
                if not app.config.get('HAS_VIDEO_PROCESSING', False):
                    # Instead of rejecting the upload, provide a fallback
                    session[progress_key] = {
                        'status': 'Using fallback mode for video processing...',
                        'percent': 15,
                        'step': 'fallback'
                    }
                    session.modified = True
                    
                    # Save video file with chunked method to prevent timeouts
                    video_file = form.video_file.data
                    filename = secure_filename(video_file.filename)
                    video_path = os.path.join(input_dir, filename)
                    
                    # Create a progress callback to update session
                    def update_progress(bytes_written):
                        mb_written = bytes_written / (1024 * 1024)
                        session[progress_key]['status'] = f'Saving video file ({mb_written:.1f}MB)...'
                        session.modified = True
                    
                    # Use chunked file save to prevent timeouts
                    save_success = save_file_chunked(
                        video_file, 
                        video_path,
                        progress_callback=update_progress
                    )
                    
                    if not save_success:
                        logging.warning("Chunked file save failed, attempting simple save")
                        try:
                            video_file.stream.seek(0)  # Reset file pointer
                            video_file.save(video_path)
                            save_success = os.path.exists(video_path) and os.path.getsize(video_path) > 0
                        except Exception as e:
                            logging.error(f"Simple file save also failed: {str(e)}")
                            save_success = False
                    
                    if not save_success:
                        flash('Error saving video file. Please try a smaller file or different format.', 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=session.get('current_process_id'))
                    
                    # Update progress
                    session[progress_key] = {
                        'status': 'Video saved. Extracting basic metadata...',
                        'percent': 30,
                        'step': 'extraction'
                    }
                    session.modified = True
                    
                    # Create basic metadata triples from filename
                    # This is a minimal fallback when transcription isn't available
                    try:
                        # Get basic info from filename (remove extension)
                        base_filename = os.path.splitext(filename)[0]
                        # Replace underscores and dashes with spaces
                        title = base_filename.replace('_', ' ').replace('-', ' ')
                        
                        # Create some basic triples about the video
                        triples = [
                            (f"video_{graph.id}_{process_id}", "title", title),
                            (f"video_{graph.id}_{process_id}", "type", "Video"),
                            (f"video_{graph.id}_{process_id}", "filename", filename),
                            (f"video_{graph.id}_{process_id}", "upload_date", datetime.now().strftime('%Y-%m-%d')),
                            (f"video_{graph.id}_{process_id}", "knowledge_graph", f"{graph.name}")
                        ]
                        
                        # Add additional context based on domain if available
                        if graph.domain == "movie":
                            triples.append((f"video_{graph.id}_{process_id}", "category", "Movie"))
                        elif graph.domain == "music":
                            triples.append((f"video_{graph.id}_{process_id}", "category", "Music Video"))
                        
                        # Update progress
                        session[progress_key] = {
                            'status': 'Created basic metadata from video file...',
                            'percent': 60,
                            'step': 'extraction'
                        }
                        session.modified = True
                        
                        # Update input source details
                        input_source.filename = filename
                        input_source.entity_count = 1  # Just the video entity
                        input_source.relationship_count = len(triples)
                        
                        # Save preview triples for later user approval
                        preview_path = os.path.join(input_dir, 'preview_triples.json')
                        with open(preview_path, 'w') as f:
                            json.dump(triples, f)
                        
                        # Save the filename for the summary page
                        filename_path = os.path.join(input_dir, 'filename.txt')
                        with open(filename_path, 'w') as f:
                            f.write(filename)
                        
                        # Create a basic transcription file
                        transcription_path = os.path.join(input_dir, 'transcription.txt')
                        with open(transcription_path, 'w') as f:
                            f.write(f"Fallback mode: Transcription not available for '{filename}'.\n\nBasic metadata extracted from filename.")
                        
                        # Save metadata for summary page
                        summary_metadata = {
                            "detected_domain": graph.domain,
                            "content_type": "Video (using fallback processing)",
                            "improvement_tips": [
                                "Install whisper for better transcription",
                                "Use more descriptive filenames for better auto-detection"
                            ]
                        }
                        metadata_path = os.path.join(input_dir, 'summary_metadata.json')
                        with open(metadata_path, 'w') as f:
                            json.dump(summary_metadata, f)
                            
                        # Update progress - ready for approval
                        session[progress_key] = {
                            'status': 'Ready for approval...',
                            'percent': 80,
                            'step': 'approval'
                        }
                        session.modified = True
                        
                        # Update input source with file info but mark as not processed yet
                        input_source.filename = filename
                        input_source.processed = False
                        
                        # Save input source
                        db.session.add(input_source)
                        db.session.commit()
                        
                        # Redirect to the summary approval page like the main flow
                        return redirect(url_for('video_summary', graph_id=graph.id, process_id=process_id))
                    except Exception as e:
                        logging.error(f"Error in fallback video processing: {e}")
                        session[progress_key] = {
                            'status': f'Error processing video: {str(e)}',
                            'percent': 0,
                            'step': 'error'
                        }
                        session.modified = True
                        flash(f'Error processing video: {str(e)}', 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)
                else:
                    # Update progress - 15%
                    session[progress_key] = {
                        'status': 'Saving video file...',
                        'percent': 15,
                        'step': 'upload'
                    }
                    session.modified = True
                    
                    # Save video file with chunked method to prevent timeouts
                    video_file = form.video_file.data
                    filename = secure_filename(video_file.filename)
                    video_path = os.path.join(input_dir, filename)
                    
                    # Create a progress callback to update session
                    def update_progress(bytes_written):
                        mb_written = bytes_written / (1024 * 1024)
                        session[progress_key]['status'] = f'Saving video file ({mb_written:.1f}MB)...'
                        session.modified = True
                    
                    # Use chunked file save to prevent timeouts
                    save_success = save_file_chunked(
                        video_file, 
                        video_path,
                        progress_callback=update_progress
                    )
                    
                    if not save_success:
                        logging.warning("Chunked file save failed, attempting simple save")
                        try:
                            video_file.stream.seek(0)  # Reset file pointer
                            video_file.save(video_path)
                            save_success = os.path.exists(video_path) and os.path.getsize(video_path) > 0
                        except Exception as e:
                            logging.error(f"Simple file save also failed: {str(e)}")
                            save_success = False
                    
                    if not save_success:
                        flash('Error saving video file. Please try a smaller file or different format.', 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=session.get('current_process_id'))
                    
                    # Update progress - 30%
                    session[progress_key] = {
                        'status': 'Transcribing video to text...',
                        'percent': 30,
                        'step': 'transcription'
                    }
                    session.modified = True
                    
                    # Transcribe video with comprehensive error handling
                    audio_path = os.path.join(input_dir, 'audio.wav')
                    try:
                        logging.info(f"Starting video transcription for {filename}")
                        
                        # Update progress for user
                        session[progress_key] = {
                            'status': 'Extracting audio from video...',
                            'percent': 35,
                            'step': 'audio_extraction'
                        }
                        session.modified = True
                        
                        # First try with the enhanced error handling in whisper_transcriber
                        transcription = whisper_transcriber.transcribe_video(video_path, audio_path)
                        logging.info(f"Successfully completed transcription of {filename}")
                        
                    except Exception as e:
                        logging.error(f"Error during video transcription: {str(e)}")
                        
                        # Update progress to show fallback
                        session[progress_key] = {
                            'status': 'Transcription error, using fallback extraction...',
                            'percent': 40,
                            'step': 'fallback'
                        }
                        session.modified = True
                        
                        # Multi-tiered fallback system - try multiple approaches
                        try:
                            # First try: Basic metadata extraction directly with ffprobe
                            logging.info("Attempting metadata extraction as primary fallback")
                            transcription = whisper_transcriber._extract_video_metadata(video_path)
                            logging.info("Used fallback metadata extraction for transcription")
                            
                        except Exception as metadata_e:
                            logging.error(f"Metadata extraction fallback failed: {str(metadata_e)}")
                            
                            # Update progress to show second fallback
                            session[progress_key] = {
                                'status': 'Using secondary fallback...',
                                'percent': 45,
                                'step': 'fallback2'
                            }
                            session.modified = True
                            
                            try:
                                # Second try: Check if we can extract some audio features
                                logging.info("Attempting audio feature extraction as secondary fallback")
                                
                                # Create a simple audio path for extraction attempt
                                audio_dir = os.path.dirname(video_path)
                                audio_path = os.path.join(audio_dir, f"fallback_audio_{uuid.uuid4().hex[:8]}.wav")
                                
                                # Try super simple audio extraction
                                extraction_cmd = [
                                    "ffmpeg", "-y", "-i", video_path, 
                                    "-vn", "-acodec", "pcm_s16le", 
                                    "-ar", "16000", "-ac", "1", 
                                    "-t", "30",  # Just first 30 seconds
                                    audio_path
                                ]
                                
                                # Run with high timeout and capture output
                                subprocess.run(extraction_cmd, timeout=90, capture_output=True, text=True)
                                
                                # Check if we got an audio file
                                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                    # Try to get audio duration with ffprobe
                                    duration_cmd = [
                                        "ffprobe", "-v", "error", "-show_entries", 
                                        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", 
                                        audio_path
                                    ]
                                    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                                    duration = float(duration_result.stdout.strip()) if duration_result.stdout.strip() else 0
                                    
                                    # Use audio info to create better fallback
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    base_filename = os.path.splitext(filename)[0]
                                    title = base_filename.replace('_', ' ').replace('-', ' ')
                                    
                                    # Create basic transcript with audio info
                                    transcription = f"""
                                        Video Analysis with Audio Detection
                                        Timestamp: {timestamp}
                                        File: {filename}
                                        Title: {title}
                                        Audio Duration: {int(duration//60)} minutes {int(duration%60)} seconds
                                        
                                        The system detected audio in this video file but could not
                                        fully transcribe the content due to processing limitations.
                                        
                                        Basic information extracted:
                                        - Video contains audio track
                                        - Audio appears to be {'stereo' if 'stereo' in subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=channels', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path], capture_output=True, text=True).stdout else 'mono'}
                                        - Audio sample rate: {subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path], capture_output=True, text=True).stdout.strip()} Hz
                                    """
                                    
                                    # Clean up temp file
                                    try:
                                        os.remove(audio_path)
                                    except:
                                        pass
                                        
                                    logging.info("Created improved fallback using audio analysis")
                                else:
                                    # If audio extraction failed, raise to go to next fallback
                                    raise Exception("Failed to extract audio for fallback")
                                    
                            except Exception as audio_e:
                                logging.error(f"Audio-based fallback also failed: {str(audio_e)}")
                                
                                # Update progress to show final fallback
                                session[progress_key] = {
                                    'status': 'Using minimal filename-based fallback...',
                                    'percent': 48,
                                    'step': 'fallback3'
                                }
                                session.modified = True
                                
                                # Third try: Most basic fallback from filename
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                base_filename = os.path.splitext(filename)[0]
                                title = base_filename.replace('_', ' ').replace('-', ' ')
                                
                                # Extract words with 3+ chars for basic entities
                                words = []
                                for word in re.findall(r"\b\w{3,}\b", title):
                                    words.append(word.capitalize())
                                    
                                # Check for potential domain-specific content
                                domain_hints = []
                                if any(term in filename.lower() for term in ["veda", "upanishad", "sanskrit", "hindu"]):
                                    domain_hints.append("Likely contains Vedic or ancient text content")
                                elif any(term in filename.lower() for term in ["movie", "film", "cinema", "actor"]):
                                    domain_hints.append("Appears to be movie or film-related content")
                                elif any(term in filename.lower() for term in ["crime", "case", "investigation", "detective"]):
                                    domain_hints.append("May contain crime case or investigation content")
                                
                                # Build the transcription with any domain hints
                                transcription = f"""
                                    Basic Video Information (All Processing Failed)
                                    Timestamp: {timestamp}
                                    File: {filename}
                                    Suspected Title: {title}
                                    
                                    Unable to extract detailed content from this video.
                                    The system encountered errors during processing.
                                    
                                    Basic entities detected from filename: 
                                    {', '.join(words)}
                                    
                                    {domain_hints[0] if domain_hints else "No specific domain detected from filename"}
                                """
                                logging.warning("Using minimal fallback transcription from filename only")
                    
                    # Update progress - showing summary for approval
                    session[progress_key] = {
                        'status': 'Generating summary for approval...',
                        'percent': 60,
                        'step': 'summary'
                    }
                    session.modified = True
                    
                    # Save the transcription for the summary page
                    transcription_path = os.path.join(input_dir, 'transcription.txt')
                    with open(transcription_path, 'w') as f:
                        f.write(transcription)
                    
                    # Generate preview triples for user approval
                    preview_triples = kg_processor.extract_triples_from_text(transcription, domain=graph.domain)
                    
                    # If no meaningful triples were found, use at least the basic metadata
                    if len(preview_triples) < 3:
                        # Generate some basic triples about the video
                        base_filename = os.path.splitext(filename)[0]
                        # Replace underscores and dashes with spaces
                        title = base_filename.replace('_', ' ').replace('-', ' ')
                        preview_triples = [
                            (f"video_{process_id[:8]}", "title", title),
                            (f"video_{process_id[:8]}", "type", "Video"),
                            (f"video_{process_id[:8]}", "filename", filename),
                            (f"video_{process_id[:8]}", "upload_date", datetime.now().strftime('%Y-%m-%d')),
                            (f"video_{process_id[:8]}", "knowledge_graph", f"{graph.name}")
                        ]
                    
                    # Save preview triples for later use
                    preview_path = os.path.join(input_dir, 'preview_triples.json')
                    with open(preview_path, 'w') as f:
                        json.dump(preview_triples, f)
                    
                    # Auto-detect domain from content if possible
                    filename_lower = filename.lower()
                    detected_domain = graph.domain
                    
                    # Try to determine content type from filename
                    if any(term in filename_lower for term in ["murder", "crime", "case", "detective", "police", "forensic", "investigation"]):
                        content_type = "Crime investigation or murder case study"
                        improvement_tips = [
                            "Add specific location information in the file name",
                            "Include dates in the file name for better timeline extraction",
                            "Consider pre-processing the video to add named entities in description"
                        ]
                    elif any(term in filename_lower for term in ["veda", "vedic", "upanishad", "sanskrit", "hindu"]):
                        content_type = "Vedic or ancient text analysis"
                        improvement_tips = [
                            "Specify which Veda is being discussed in the filename",
                            "Add Sanskrit terms in description for better entity extraction"
                        ]
                    elif any(term in filename_lower for term in ["movie", "film", "cinema", "trailer"]):
                        content_type = "Movie or film content"
                        improvement_tips = [
                            "Include release year in the filename",
                            "Add director and main actors to filename or description"
                        ]
                    else:
                        content_type = "General video content"
                        improvement_tips = [
                            "Use more descriptive filenames for better auto-detection",
                            "Include key entities in the filename for improved extraction"
                        ]
                    
                    # Save metadata for summary page
                    summary_metadata = {
                        "detected_domain": detected_domain,
                        "content_type": content_type,
                        "improvement_tips": improvement_tips
                    }
                    metadata_path = os.path.join(input_dir, 'summary_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(summary_metadata, f)
                    
                    # Update input source with file info but mark as not processed yet
                    input_source.filename = filename
                    input_source.processed = False
                    
                    # Save input source to get its ID
                    db.session.add(input_source)
                    db.session.commit()
                    
                    # Redirect to the summary approval page
                    return redirect(url_for('video_summary', graph_id=graph.id, process_id=process_id))
                
            elif form.input_type.data == 'csv':
                # Update progress - 15%
                session[progress_key] = {
                    'status': 'Saving CSV file...',
                    'percent': 15,
                    'step': 'upload'
                }
                session.modified = True
                
                # Save CSV file with chunked method to prevent timeouts
                csv_file = form.csv_file.data
                filename = secure_filename(csv_file.filename)
                csv_path = os.path.join(input_dir, filename)
                
                # Create a progress callback to update session
                def update_progress(bytes_written):
                    mb_written = bytes_written / (1024 * 1024)
                    session[progress_key]['status'] = f'Saving CSV file ({mb_written:.1f}MB)...'
                    session.modified = True
                
                # Use chunked file save to prevent timeouts
                save_success = save_file_chunked(
                    csv_file, 
                    csv_path,
                    progress_callback=update_progress
                )
                
                if not save_success:
                    logging.warning("Chunked file save failed, attempting simple save")
                    try:
                        csv_file.stream.seek(0)  # Reset file pointer
                        csv_file.save(csv_path)
                        save_success = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
                    except Exception as e:
                        logging.error(f"Simple file save also failed: {str(e)}")
                        save_success = False
                
                if not save_success:
                    flash('Error saving CSV file. Please try a smaller file or different format.', 'danger')
                    return render_template('process.html', form=form, graph=graph, current_process_id=session.get('current_process_id'))
                
                # Update progress - 30%
                session[progress_key] = {
                    'status': 'Analyzing CSV structure...',
                    'percent': 30,
                    'step': 'analysis'
                }
                session.modified = True
                
                # Process CSV file - with fallback for missing pandas
                try:
                    # Update progress - 50%
                    session[progress_key] = {
                        'status': 'Extracting entities and relationships...',
                        'percent': 50,
                        'step': 'extraction'
                    }
                    session.modified = True
                    
                    # First try using the standard kg_processor
                    triples = kg_processor.process_csv(csv_path, domain=graph.domain)
                except Exception as e:
                    logging.warning(f"Error processing CSV with kg_processor: {e}")
                    logging.info("Attempting to process CSV with basic fallback processor")
                    
                    # Update progress - fallback
                    session[progress_key] = {
                        'status': 'Using fallback CSV processor...',
                        'percent': 45,
                        'step': 'fallback'
                    }
                    session.modified = True
                    
                    # Basic fallback CSV processor
                    try:
                        triples = []
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            # Try to read as CSV
                            import csv
                            reader = csv.reader(f)
                            headers = next(reader)  # Get headers
                            
                            # Process each row - create simple subject-predicate-object triples
                            for row in reader:
                                if len(row) < 2:
                                    continue  # Skip rows with insufficient columns
                                
                                # Create basic entity identifier
                                entity_id = row[0]
                                
                                # Create relationship triples from each column
                                for i, value in enumerate(row[1:], 1):
                                    if value and i < len(headers):
                                        predicate = headers[i]
                                        triples.append((entity_id, predicate, value))
                        
                        if not triples:
                            raise ValueError("No valid triples could be extracted from CSV")
                        
                        logging.info(f"Successfully extracted {len(triples)} triples using fallback processor")
                    except Exception as inner_e:
                        logging.error(f"Fallback CSV processing failed: {inner_e}")
                        flash(f"Error processing CSV data: {str(inner_e)}", 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)
                
                # Update input source details
                input_source.filename = filename
                input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples]))
                input_source.relationship_count = len(triples)
                
                # Insert into Neo4j
                neo4j_manager.import_triples(triples, graph.id)
                
            elif form.input_type.data == 'url':
                # Download CSV from URL
                url = form.github_url.data
                if url:
                    try:
                        filename = url.split('/')[-1]
                        csv_path = os.path.join(input_dir, filename)
                        logging.info(f"Downloading CSV from URL: {url} to {csv_path}")
                        urllib.request.urlretrieve(url, csv_path)
                        logging.info(f"Successfully downloaded CSV file from URL")
                    except Exception as e:
                        logging.error(f"Error downloading CSV from URL: {e}")
                        flash(f"Error downloading CSV: {str(e)}", 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)
                else:
                    flash("No GitHub URL provided", 'danger')
                    return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)
                
                # Process CSV file - with fallback for missing pandas
                try:
                    # First try using the standard kg_processor
                    triples = kg_processor.process_csv(csv_path, domain=graph.domain)
                except Exception as e:
                    logging.warning(f"Error processing CSV with kg_processor: {e}")
                    logging.info("Attempting to process CSV with basic fallback processor")
                    
                    # Basic fallback CSV processor
                    try:
                        triples = []
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            # Try to read as CSV
                            import csv
                            reader = csv.reader(f)
                            headers = next(reader)  # Get headers
                            
                            # Process each row - create simple subject-predicate-object triples
                            for row in reader:
                                if len(row) < 2:
                                    continue  # Skip rows with insufficient columns
                                
                                # Create basic entity identifier
                                entity_id = row[0]
                                
                                # Create relationship triples from each column
                                for i, value in enumerate(row[1:], 1):
                                    if value and i < len(headers):
                                        predicate = headers[i]
                                        triples.append((entity_id, predicate, value))
                        
                        if not triples:
                            raise ValueError("No valid triples could be extracted from CSV")
                        
                        logging.info(f"Successfully extracted {len(triples)} triples using fallback processor")
                    except Exception as inner_e:
                        logging.error(f"Fallback CSV processing failed: {inner_e}")
                        flash(f"Error processing CSV data: {str(inner_e)}", 'danger')
                        return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)
                
                # Update input source details
                input_source.url = url
                input_source.filename = filename
                input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples]))
                input_source.relationship_count = len(triples)
                
                # Insert into Neo4j
                neo4j_manager.import_triples(triples, graph.id)
            
            # Update progress - 90%
            session[progress_key] = {
                'status': 'Finalizing and updating records...',
                'percent': 90,
                'step': 'finalizing'
            }
            session.modified = True
            
            input_source.processed = True
            db.session.add(input_source)
            db.session.commit()
            
            # Update progress - 100%
            session[progress_key] = {
                'status': 'Processing complete! Redirecting to visualization...',
                'percent': 100,
                'step': 'complete'
            }
            session.modified = True
            
            flash(f'Data processed and added to knowledge graph successfully! {input_source.entity_count} entities and {input_source.relationship_count} relationships were created.', 'success')
            return redirect(url_for('visualization', graph_id=graph.id))
            
        except NotImplementedError as e:
            # Update progress - error
            session[progress_key] = {
                'status': f'Feature not available: {str(e)}',
                'percent': 0,
                'step': 'error'
            }
            session.modified = True
            flash(f'Feature not available: {str(e)}', 'warning')
            logging.warning(f"Not implemented: {e}")
        except Exception as e:
            # Update progress - error
            session[progress_key] = {
                'status': f'Error processing data: {str(e)}',
                'percent': 0,
                'step': 'error'
            }
            session.modified = True
            flash(f'Error processing data: {str(e)}', 'danger')
            logging.error(f"Error processing data: {e}", exc_info=True)
    
    # Pass the current process ID if available to the template
    current_process_id = session.get('current_process_id', '')
    return render_template('process.html', form=form, graph=graph, current_process_id=current_process_id)


@app.route('/graph/<int:graph_id>/video-summary/<string:process_id>', methods=['GET'])
@login_required
def video_summary(graph_id, process_id):
    """Show video transcription summary and allow user to approve/reject"""
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Security check - ensure graph belongs to current user
    if graph.user_id != current_user.id:
        flash('You do not have permission to access this graph', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the processing directory using the same structure as in process_data
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        flash('Process data not found', 'danger')
        return redirect(url_for('process_data', graph_id=graph_id))
    
    # Get the transcription
    transcription_path = os.path.join(input_dir, 'transcription.txt')
    if os.path.exists(transcription_path):
        with open(transcription_path, 'r') as f:
            transcription = f.read()
    else:
        transcription = "Transcription not available"
    
    # Get the preview triples
    preview_path = os.path.join(input_dir, 'preview_triples.json')
    if os.path.exists(preview_path):
        with open(preview_path, 'r') as f:
            preview_triples = json.load(f)
    else:
        preview_triples = []
    
    # Get the metadata
    metadata_path = os.path.join(input_dir, 'summary_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "detected_domain": graph.domain,
            "content_type": "Unknown content type",
            "improvement_tips": ["Add more descriptive filenames"]
        }
    
    # Get the filename
    filename_path = os.path.join(input_dir, 'filename.txt')
    if os.path.exists(filename_path):
        with open(filename_path, 'r') as f:
            filename = f.read()
    else:
        filename = "Unknown file"
    
    # Prepare data for display
    entity_types = {}
    relationship_types = {}
    
    # Count entity and relationship types
    for triple in preview_triples:
        # Subject entity
        entity_type = triple[0].split('_')[0] if '_' in triple[0] else 'Unknown'
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Object entity (if not a literal)
        if not triple[2].startswith('"') and not triple[2].isdigit():
            entity_type = triple[2].split('_')[0] if '_' in triple[2] else 'Unknown'
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Relationship type
        rel_type = triple[1]
        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
    
    return render_template(
        'video_summary.html', 
        graph=graph,
        process_id=process_id,
        transcription=transcription,
        preview_triples=preview_triples,
        entity_types=entity_types,
        relationship_types=relationship_types,
        metadata=metadata,
        filename=filename,
        total_entities=len(set([t[0] for t in preview_triples] + [t[2] for t in preview_triples if not t[2].startswith('"') and not t[2].isdigit()])),
        total_relationships=len(preview_triples)
    )

@app.route('/graph/<int:graph_id>/approve-video/<string:process_id>', methods=['POST'])
@login_required
def approve_video(graph_id, process_id):
    """Approve video processing and create knowledge graph"""
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Security check - ensure graph belongs to current user
    if graph.user_id != current_user.id:
        flash('You do not have permission to access this graph', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the processing directory using the same structure as in process_data
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
    
    # Check if directory exists
    if not os.path.exists(input_dir):
        flash('Process data not found', 'danger')
        return redirect(url_for('process_data', graph_id=graph_id))
    
    # Load the preview triples
    preview_path = os.path.join(input_dir, 'preview_triples.json')
    if not os.path.exists(preview_path):
        flash('Preview data not found', 'danger')
        return redirect(url_for('video_summary', graph_id=graph_id, process_id=process_id))
    
    with open(preview_path, 'r') as f:
        triples = json.load(f)
    
    # Initialize the progress key - match the format from process_data
    process_key = f"{graph_id}_{process_id}"
    progress_key = f"progress_{process_key}"
    
    # Update progress - 80%
    session[progress_key] = {
        'status': 'Importing approved data to graph database...',
        'percent': 80,
        'step': 'database'
    }
    session.modified = True
    
    # Get input source
    input_source = InputSource.query.filter_by(
        graph_id=graph_id, 
        source_type='video'
    ).order_by(InputSource.created_at.desc()).first()
    
    if input_source:
        # Update input source details
        input_source.entity_count = len(set([t[0] for t in triples] + [t[2] for t in triples if not t[2].startswith('"') and not t[2].isdigit()]))
        input_source.relationship_count = len(triples)
        input_source.processed = True
        db.session.commit()
    
    # Save as CSV for Neo4j
    csv_path = os.path.join(input_dir, 'approved_triples.csv')
    kg_processor.save_triples_to_csv(triples, csv_path)
    
    # Insert into Neo4j
    neo4j_manager.import_triples(triples, graph.id)
    
    # Update progress - 100%
    session[progress_key] = {
        'status': 'Processing complete!',
        'percent': 100,
        'step': 'complete'
    }
    session.modified = True
    
    flash('Video data has been approved and added to your knowledge graph', 'success')
    return redirect(url_for('visualization', graph_id=graph_id))

@app.route('/graph/<int:graph_id>/reject-video/<string:process_id>', methods=['POST'])
@login_required
def reject_video(graph_id, process_id):
    """Reject video processing and return to data input"""
    graph = KnowledgeGraph.query.get_or_404(graph_id)
    
    # Security check - ensure graph belongs to current user
    if graph.user_id != current_user.id:
        flash('You do not have permission to access this graph', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the processing directory using the same structure as in process_data
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], process_id)
    
    # Delete the input source record
    input_source = InputSource.query.filter_by(
        graph_id=graph_id, 
        source_type='video'
    ).order_by(InputSource.created_at.desc()).first()
    
    if input_source:
        db.session.delete(input_source)
        db.session.commit()
    
    # Reset progress - match the format from process_data
    process_key = f"{graph_id}_{process_id}"
    progress_key = f"progress_{process_key}"
    if progress_key in session:
        session.pop(progress_key)
    
    flash('Video processing has been rejected. Please try with a different input source or settings.', 'info')
    return redirect(url_for('process_data', graph_id=graph_id))

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


# API endpoint to get processing progress
@app.route('/api/progress/<string:process_id>', methods=['GET'])
@login_required
def get_processing_progress(process_id):
    """Get the current progress of a data processing job"""
    # The process_id should be in the format "{graph_id}_{uuid}"
    # This matches the format stored in session['current_process_id']
    
    # Define consistent progress key
    progress_key = f"progress_{process_id}"
    
    # Check if we have the current process ID in the session
    current_id = session.get('current_process_id')
    
    # If the requested ID matches the current process ID, use that progress
    if current_id and process_id == current_id:
        progress_data = session.get(progress_key, {
            'status': 'Processing data...',
            'percent': 30,
            'step': 'processing'
        })
    else:
        # Default to waiting if no match or no current process
        progress_data = session.get(progress_key, {
            'status': 'Waiting to start...',
            'percent': 0,
            'step': 'waiting'
        })
    
    return jsonify(progress_data)

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
