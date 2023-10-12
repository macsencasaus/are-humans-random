import numpy as np
import tensorflow as tf

from numpy.random import randint
from keras import Sequential
from keras.layers import Dense
from datetime import datetime

from ..models import Submission, Ann_Model
from .. import app, db

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

input_size = 32
hidden_layer_size = 64
output_size =  2

model = Sequential([
    Dense(hidden_layer_size, activation='relu', input_shape=(input_size,)),
    Dense(hidden_layer_size, activation='relu'),
    Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def load_weights():
    
    with app.app_context:
        last_model = Ann_Model.query.order_by(Ann_Model.id.desc()).first().weight_path

    model.load_weights(last_model)

def train_model():
    batch_size = 16
    epochs = 64

    weight_path = f'../ann_models/model_{datetime.utcnow()}.h5'

    with app.app_context():
        x_sub = Submission.query.all()

    x_str = [x.sample for x in x_sub]

    x_list = []

    for binary_str in x_str:
        binary_list = [int(char) for char in binary_str]
        x_list.append(binary_list)

    x_human = np.array(x_list)
    x_real = np.array([randint(2) for _ in x_human])
    x = np.concatenate(x_human, x_real)

    # One hot encode labels
    y_human = np.concatenate(np.zeros(x_human.shape), np.ones(x_human.shape), axis=1)
    y_real = np.concatenate(np.ones(x_human.shape), np.zeros(x_human.shape), axis=1)
    y = np.concatenate(y_human, y_real)

    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

    model.save_weights(weight_path)
    ann_model = Ann_Model(weight_path=weight_path)

    db.session.add(ann_model)
    db.session.commit()

def predict(submission_str: str):

    submission = [int(char) for char in submission_str]

    return model.predict(submission)[0]




if __name__ == '__main__':
    print(model.summary())