import os
import logging
from flask import Flask, render_template, g

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Create a simple class to emulate current_user for templates
class AnonymousUser:
    is_authenticated = False
    is_active = False
    is_anonymous = True

# Context processor to add current_user to all templates
@app.context_processor
def inject_user():
    # Return anonymous user for all templates
    return {'current_user': AnonymousUser()}

# Basic route that works without database
@app.route('/')
def index():
    return render_template('index.html')

# Login page (simplified)
@app.route('/login')
def login():
    return render_template('db_unavailable.html',
                         error_title="Login Unavailable",
                         error_message="User login is temporarily unavailable due to database maintenance.")

# Register page (simplified)
@app.route('/register')
def register():
    return render_template('db_unavailable.html',
                         error_title="Registration Unavailable",
                         error_message="User registration is temporarily unavailable due to database maintenance.")

# Dashboard page (simplified)
@app.route('/dashboard')
def dashboard():
    return render_template('db_unavailable.html',
                         error_title="Dashboard Unavailable",
                         error_message="The dashboard is temporarily unavailable due to database maintenance.")

# Basic error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {e}")
    return render_template('db_unavailable.html', 
                          error_title="Application Error", 
                          error_message="The application experienced an error. Please try again later."), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
