import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    label_encoder = LabelEncoder()
    df['answer_encoded'] = label_encoder.fit_transform(df['answer'])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['question'])
    sequences = tokenizer.texts_to_sequences(df['question'])
    max_sequence_len = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
    y = df['answer_encoded'].values

    return X, y, tokenizer, label_encoder, max_sequence_len
