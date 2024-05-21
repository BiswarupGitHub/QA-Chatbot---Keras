from flask import Flask, request, render_template, jsonify
from app.preprocess import load_and_preprocess_data
from app.model import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)
app.secret_key = '12345'

# Load the model and preprocessing data
model = load_model('./app/my_model.h5')
X, y, tokenizer, label_encoder, max_sequence_len = load_and_preprocess_data('./data/my.csv')

def predict_answer(question):
    seq = tokenizer.texts_to_sequences([question])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_len, padding='post')
    pred = model.predict(padded_seq)
    answer_index = pred.argmax(axis=1)[0]
    return label_encoder.inverse_transform([answer_index])[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.get_json()  # Parse JSON data from the request
    question = data.get('question')
    answer = predict_answer(question)
    return jsonify({'question': question, 'answer': answer})

if __name__ == "__main__":
    app.run(debug=True)
