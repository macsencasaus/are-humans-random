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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_weights():
    
    with app.app_context():
        last_model = Ann_Model.query.order_by(Ann_Model.id.desc()).first().weight_path

    model.load_weights(last_model)

def verify_train():

    with app.app_context():
        try:
            last_trained_on = Ann_Model.query.order_by(Ann_Model.id.desc()).first().trained_on
            count = Submission.query.count()
        except:
            return True
    
    return count >= last_trained_on * 2 

def train_model():
    batch_size = 16
    epochs = 64

    weight_path = f"./detector/ann_models/model_{str(datetime.utcnow()).replace(' ', '_').replace(':', '.')}.h5"

    with app.app_context():
        x_sub = Submission.query.all()

    x_str = [x.sample for x in x_sub]

    x_list = []

    for binary_str in x_str:
        binary_list = [int(char) for char in binary_str]
        x_list.append(binary_list)
    x_human = np.array(x_list)

    x_bot = np.random.randint(2, size=(len(x_human), input_size))
    x = np.concatenate((x_human, x_bot))

    # One hot encode labels
    y_human = np.repeat([np.array([0,1])], repeats=len(x_human), axis=0)
    y_bot = np.repeat([np.array([1,0])], repeats=len(x_bot), axis=0)
    y = np.concatenate((y_human, y_bot), axis=0)

    # print(f'x shape: {x.shape}, y shape: {y.shape}')

    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)

    model.save_weights(filepath=weight_path)
    ann_model = Ann_Model(weight_path=weight_path, trained_on=len(x_human))

    db.session.add(ann_model)
    db.session.commit()

def predict(submission_str: str):

    submission = [int(char) for char in submission_str]
    prediction = model.predict([submission], verbose=0)
    print(prediction)
    return prediction[0][1]

if __name__ == '__main__':
    print(model.summary())