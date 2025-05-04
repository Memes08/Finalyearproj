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
    ], validators=[DataRequired()])
    
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
        if not super().validate():
            return False
            
        if self.input_type.data == 'video' and not self.video_file.data:
            self.video_file.errors.append('Please upload a video file.')
            return False
        elif self.input_type.data == 'youtube_transcript' and not self.youtube_transcript.data:
            self.youtube_transcript.errors = ['Please paste the YouTube transcript.']
            return False
        elif self.input_type.data == 'text' and not (self.text_file.data or self.text_content.data):
            self.text_file.errors.append('Please either upload a text file or paste text directly.')
            return False
        elif self.input_type.data == 'url' and not self.github_url.data:
            self.github_url.errors.append('Please enter a GitHub raw file URL.')
            return False
            
        return True


class QueryForm(FlaskForm):
    query = TextAreaField('Ask a question about your knowledge graph', 
                         validators=[DataRequired(), Length(min=5, max=500)])
    submit = SubmitField('Search')
