import os
import logging

from flask import Flask, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager


# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Create base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize database
db = SQLAlchemy(model_class=Base)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    # Handle Heroku-style PostgreSQL URLs
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///knowledge_graph.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "connect_args": {"connect_timeout": 15}  # Add a timeout to prevent long connection attempts
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Set upload folder and max content length
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create directory for triples data
os.makedirs('data/triples', exist_ok=True)

# Flag to indicate if optional dependencies are available
app.config['HAS_VIDEO_PROCESSING'] = False
app.config['HAS_NEO4J'] = False
app.config['HAS_LLM'] = False

# Neo4j configuration
app.config['NEO4J_URI'] = os.environ.get("NEO4J_URI")
app.config['NEO4J_USERNAME'] = os.environ.get("NEO4J_USERNAME")
app.config['NEO4J_PASSWORD'] = os.environ.get("NEO4J_PASSWORD")

# GROQ API key for LLM services
app.config['GROQ_API_KEY'] = os.environ.get("XAI_API_KEY")

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Database availability flag
app.config['HAS_DATABASE'] = False

# Global error handlers
@app.errorhandler(OperationalError)
def handle_db_operational_error(e):
    logging.error(f"Database operational error: {e}")
    flash("Database connection error. Please try again later.", "danger")
    return render_template('db_unavailable.html'), 503

@app.errorhandler(SQLAlchemyError)
def handle_db_error(e):
    logging.error(f"Database error: {e}")
    flash("Database error occurred. Please try again later.", "danger")
    return render_template('db_unavailable.html'), 503

@app.errorhandler(500)
def handle_server_error(e):
    logging.error(f"Internal server error: {e}")
    return render_template('db_unavailable.html', 
                          error_title="Server Error", 
                          error_message="An unexpected error occurred. Our team has been notified."), 500

@app.errorhandler(404)
def handle_not_found(e):
    return render_template('db_unavailable.html', 
                          error_title="Page Not Found", 
                          error_message="The requested page does not exist."), 404

@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    logging.error(f"Unhandled exception: {e}")
    return render_template('db_unavailable.html', 
                          error_title="Unexpected Error", 
                          error_message="Something went wrong. Please try again later."), 500

try:
    with app.app_context():
        # Import models to ensure they're registered with SQLAlchemy
        import models
        
        # Create all database tables
        db.create_all()
        
        # Set database availability flag
        app.config['HAS_DATABASE'] = True
        
        # Import routes after models to avoid circular imports
        import routes
        
        logging.info("Database successfully initialized")
except Exception as e:
    logging.error(f"Error initializing database: {e}")
    logging.warning("Application will continue without database support. Some features may be unavailable.")
    
    # Continue by importing routes anyway
    import routes
