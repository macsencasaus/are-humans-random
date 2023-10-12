from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from detector.models import db 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deez_nutz'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db.init_app(app)

from detector import routes