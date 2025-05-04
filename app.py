import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
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
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///knowledge_graph.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
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
app.config['GROQ_API_KEY'] = os.environ.get("GROQ_API_KEY")

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

with app.app_context():
    # Import models to ensure they're registered with SQLAlchemy
    import models
    
    # Create all database tables
    db.create_all()
    
    # Import routes after models to avoid circular imports
    import routes
