import numpy as np

from flask import render_template, url_for, flash
from . import app
from .forms import TestForm, TrainForm
from .models import db, Submission

from .ann_utils import ann

@app.route('/')
def home():
    return render_template('home.jinja')

@app.route('/train', methods=['GET', 'POST'])
def train():
    form = TrainForm()
    if form.is_submitted():
        if form.validate():
            text = form.user_input.data
            sample = Submission(sample=text)
            db.session.add(sample)
            db.session.commit()
            if (ann.verify_train()):
                ann.train_model()
            flash('Successful Submission!', 'success')
        else:
            flash('Input too short, try again!', 'warning')
    return render_template('train.jinja', title='Train', form=form)

@app.route('/test', methods=['GET', 'POST'])
def test():
    ann.load_weights()
    form = TestForm()
    if form.validate_on_submit():
        text = form.user_input.data
        ann_prob = np.round(ann.predict(text)*100, decimals=3)
        return render_template('test.jinja', title='Test', form=form, last_submission=text, ann_prob=ann_prob)
    
    return render_template('test.jinja', title='Test', form=form)

@app.route('/train-model', methods=['POST'])
def train_model():
    # ann.train_model()

    return 'Successfully trained!'

if __name__ == '__main__':
    app.run(debug=True)