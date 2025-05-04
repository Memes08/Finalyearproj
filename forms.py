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
        ('custom', 'Custom')
    ], validators=[DataRequired()])
    submit = SubmitField('Create Knowledge Graph')


class DataInputForm(FlaskForm):
    input_type = RadioField('Input Type', choices=[
        ('video', 'Video File'),
        ('csv', 'CSV Upload'),
        ('url', 'GitHub CSV URL')
    ], validators=[DataRequired()])
    
    video_file = FileField('Upload Video', validators=[
        Optional(),
        FileAllowed(['mp4', 'avi', 'mov', 'mkv'], 'Only video files allowed!')
    ])
    
    csv_file = FileField('Upload CSV', validators=[
        Optional(),
        FileAllowed(['csv'], 'Only CSV files allowed!')
    ])
    
    github_url = URLField('GitHub Raw CSV URL', validators=[Optional(), URL()])
    
    submit = SubmitField('Process Data')
    
    def validate(self, **kwargs):
        if not super().validate():
            return False
            
        if self.input_type.data == 'video' and not self.video_file.data:
            self.video_file.errors.append('Please upload a video file.')
            return False
        elif self.input_type.data == 'csv' and not self.csv_file.data:
            self.csv_file.errors.append('Please upload a CSV file.')
            return False
        elif self.input_type.data == 'url' and not self.github_url.data:
            self.github_url.errors.append('Please enter a GitHub CSV URL.')
            return False
            
        return True


class QueryForm(FlaskForm):
    query = TextAreaField('Ask a question about your knowledge graph', 
                         validators=[DataRequired(), Length(min=5, max=500)])
    submit = SubmitField('Search')
