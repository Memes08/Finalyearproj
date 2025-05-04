import os
import logging
from flask import Flask, render_template

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Basic route that works without database
@app.route('/')
def index():
    return render_template('base.html')

# Basic error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {e}")
    return render_template('db_unavailable.html', 
                          error_title="Application Error", 
                          error_message="The application experienced an error. Please try again later."), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
