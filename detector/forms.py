from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class TrainForm(FlaskForm):
    user_input = StringField('Enter only values of 1 or 0',
                             validators=[DataRequired(), Length(min=32, max = 32)])
    submit = SubmitField('Submit')

class TestForm(FlaskForm):
    user_input = StringField('Enter only values of 1 or 0',
                             validators=[DataRequired(), Length(min=32, max = 32)])
    submit = SubmitField('Submit')