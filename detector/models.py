from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sample = db.Column(db.String(32), nullable=False)

    def __repr__(self):
        return f'{self.sample}'

class Ann_Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    weight_path = db.Column(db.String, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    trained_on = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'ANN model: {self.weight_path}, Trained: {self.date}, Human samples trained on: {self.trained_on}'