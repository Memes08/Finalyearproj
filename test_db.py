import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Create app
app = Flask(__name__)

# Get database URL from environment
database_url = os.environ.get('DATABASE_URL')
logging.info(f"Original DATABASE_URL: {database_url}")

# Fix potential Heroku-style URL
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    logging.info(f"Fixed DATABASE_URL: {database_url}")

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Simple User model for testing
class TestUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

# Try to create tables and add a test user
with app.app_context():
    try:
        db.create_all()
        logging.info("Tables created successfully")
        
        # Try to add a test user
        test_user = TestUser(name="Test User")
        db.session.add(test_user)
        db.session.commit()
        logging.info("Test user added successfully")
        
        # Try to query
        users = TestUser.query.all()
        logging.info(f"Found {len(users)} users")
        for user in users:
            logging.info(f"User: {user.id} - {user.name}")
            
    except SQLAlchemyError as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"General error: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)