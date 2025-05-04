from datetime import datetime
from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Define relationship to knowledge graphs
    knowledge_graphs = db.relationship('KnowledgeGraph', backref='owner', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class KnowledgeGraph(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    domain = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Define relationship to input sources
    input_sources = db.relationship('InputSource', backref='knowledge_graph', lazy='dynamic')


class InputSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_type = db.Column(db.String(20), nullable=False)  # 'video', 'csv', 'url'
    filename = db.Column(db.String(255))
    url = db.Column(db.String(255))
    processed = db.Column(db.Boolean, default=False)
    entity_count = db.Column(db.Integer, default=0)
    relationship_count = db.Column(db.Integer, default=0)
    graph_id = db.Column(db.Integer, db.ForeignKey('knowledge_graph.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_source_display(self):
        if self.source_type == 'video':
            return f"Video: {self.filename}"
        elif self.source_type == 'csv':
            return f"CSV: {self.filename}"
        elif self.source_type == 'url':
            return f"URL: {self.url}"
        return "Unknown source"
