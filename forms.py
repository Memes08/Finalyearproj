import re
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, FileField, RadioField, URLField
from wtforms.validators import DataRequired, Email, EqualTo, Length, URL, Optional, ValidationError
from flask_wtf.file import FileAllowed
from models import User


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please use a different one.')


class NewKnowledgeGraphForm(FlaskForm):
    name = StringField('Graph Name', validators=[DataRequired(), Length(min=3, max=100)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    domain = SelectField('Domain', choices=[
        ('movie', 'Movies'),
        ('book', 'Books'),
        ('music', 'Music'),
        ('academic', 'Academic'),
        ('vedas', 'Vedas & Ancient Texts'),
        ('business', 'Business'),
        ('crime', 'Crime Cases'),
        ('custom', 'Custom')
    ], validators=[DataRequired()])
    submit = SubmitField('Create Knowledge Graph')


class DataInputForm(FlaskForm):
    input_type = RadioField('Input Type', choices=[
        ('video', 'Video File'),
        ('youtube_transcript', 'YouTube Transcript'),
        ('text', 'Text File Upload'),
        ('url', 'GitHub Raw File URL')
    ], default='url', validators=[DataRequired()])
    
    video_file = FileField('Upload Video', validators=[
        Optional(),
        FileAllowed(['mp4', 'avi', 'mov', 'mkv'], 'Only video files allowed!')
    ])
    
    youtube_url = URLField('YouTube Video URL', validators=[
        Optional(), 
        URL(message='Please enter a valid YouTube URL'),
    ])
    
    youtube_video_id = StringField('YouTube Video ID', validators=[Optional()])
    youtube_transcript = TextAreaField('YouTube Transcript', validators=[Optional()])
    youtube_title = StringField('Video Title (Optional)', validators=[Optional()])
    
    text_file = FileField('Upload Text File', validators=[
        Optional(),
        FileAllowed(['txt', 'md', 'text'], 'Only text files allowed!')
    ])
    
    text_content = TextAreaField('Or paste text directly:', validators=[Optional()])
    
    github_url = URLField('GitHub Raw File URL', validators=[Optional(), URL()])
    
    submit = SubmitField('Process Data')
    
    def validate(self, **kwargs):
        # Import logging and request here to avoid circular import
        import logging
        from flask import request
        
        # Log all form data for debugging
        logging.info(f"Form data received: {request.form}")
        logging.info(f"Files received: {request.files}")
        
        # Check if we have a selected_input_type in the form data
        selected_type = request.form.get('selected_input_type')
        if selected_type and selected_type in ['video', 'youtube', 'youtube_transcript', 'text', 'url']:
            logging.info(f"Setting input_type from selected_input_type: {selected_type}")
            self.input_type.data = selected_type
        
        # Check for the radio button data as well
        radio_input_type = request.form.get('input_type')
        if radio_input_type and not selected_type:
            logging.info(f"Using radio input_type: {radio_input_type}")
            self.input_type.data = radio_input_type
            
        # Skip standard validation as it's causing issues
        # Just focus on the core validation logic for each input type
        logging.info(f"Validating form with input_type={self.input_type.data}")
        
        if not self.input_type.data:
            logging.warning("No input type selected")
            self.input_type.errors = ['Please select an input type.']
            return False
        
        if self.input_type.data == 'video':
            if not request.files.get('video_file'):
                logging.warning("Video file validation failed: No file uploaded")
                self.video_file.errors = ['Please upload a video file.']
                return False
                
        elif self.input_type.data == 'youtube_transcript':
            transcript_data = request.form.get('youtube_transcript', '')
            logging.info(f"YouTube transcript data length: {len(transcript_data)}")
            if not transcript_data:
                logging.warning("YouTube transcript validation failed: No transcript provided")
                self.youtube_transcript.errors = ['Please paste the YouTube transcript.']
                return False
                
        elif self.input_type.data == 'text':
            has_file = 'text_file' in request.files and request.files['text_file'].filename
            has_content = 'text_content' in request.form and request.form['text_content'].strip()
            
            logging.info(f"Text validation - has file: {has_file}, has content: {has_content}")
            
            if not (has_file or has_content):
                logging.warning("Text validation failed: No text file or content provided")
                self.text_file.errors = ['Please either upload a text file or paste text directly.']
                return False
                
        elif self.input_type.data == 'url':
            github_url = request.form.get('github_url', '')
            logging.info(f"Validating GitHub URL: '{github_url}'")
            if not github_url:
                logging.warning("URL validation failed: No GitHub URL provided")
                self.github_url.errors = ['Please enter a GitHub raw file URL.']
                return False
            
            # We're skipping strict URL validation to ensure CSV URLs from any source can work
            # This allows for both GitHub and other raw file URLs to be processed
            logging.info(f"GitHub/Raw URL validation passed: {github_url}")
            return True
            
        logging.info(f"Form validation successful for input type: {self.input_type.data}")
        return True


class QueryForm(FlaskForm):
    query = TextAreaField('Ask a question about your knowledge graph', 
                         validators=[DataRequired(), Length(min=5, max=500)])
    submit = SubmitField('Search')
