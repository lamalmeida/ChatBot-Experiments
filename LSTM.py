import json
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def preprocess_data(context_window_size):
    json_file = "data/arxiv-metadata-oai-snapshot.json" 
    count = 0
    vocab_dict = {}
    training_data = []
    training_labels = []
    with open(json_file, 'r') as f:
        for line in f:
            if count < 10000:
                count += 1
                abstract = json.loads(line)['abstract']

                tokens = word_tokenize(abstract)
                tokens = [token.lower() for token in tokens]
                tokens = [token for token in tokens if token not in string.punctuation]
                stop_words = set(stopwords.words("english"))
                #tokens = [token for token in tokens if token not in stop_words]

                for token in tokens:
                    if token not in vocab_dict:
                        vocab_dict[token] = len(vocab_dict) + 1

                for i in range(len(tokens) - context_window_size):
                    input_tokens = tokens[i:i+context_window_size]
                    output_token = tokens[i+context_window_size]
                    training_data.append(input_tokens)
                    training_labels.append(output_token)

            else:
                break
    return vocab_dict, training_data, training_labels

def load_glove_embeddings(embedding_vector_size):
    glove_file = 'data/glove.6B.' + str(embedding_vector_size) + 'd.txt'
    embeddings = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings

def convert_to_vector_embeddings(training_data, training_labels, embeddings, embedding_vector_size):
    encoded_data = []
    encoded_labels = []
    for input_tokens, output_token in zip(training_data, training_labels):
        is_invalid_flag = False
        input_vector = []
        output_vector = []
        for token in input_tokens:
            if token in embeddings:
                input_vector.append(embeddings[token])
            else:
                is_invalid_flag = True

        if is_invalid_flag:
            continue

        # Encode output token
        if output_token in embeddings:
            output_vector.append(embeddings[output_token])
        else:
            continue

        encoded_data.append(input_vector)
        encoded_labels.append(output_vector)
    
    return np.array(encoded_data), np.array(encoded_labels)

def find_closest_word(predicted_embedding, word_embeddings):
    distance = []
    words = []
    for word, embedding in word_embeddings.items():
        distance.append(np.linalg.norm(predicted_embedding - embedding))
        words.append(word)
    sorted_pairs = sorted(zip(distance, words))
    sorted_list = [pair[1] for pair in sorted_pairs]

    return sorted_list[:5]

class LSTMModel:
    def __init__(self, context_window_size, embedding_vector_size):
        ###################### Model Loss: 100 epochs 1.8098 (still improving, batch size 16) (6 correct)
        ###################### Model Loss: 100 epochs 1.8157 (still improving, batch size 32)
        ###################### Model Loss: 100 epochs 1.8484 (still improving, batch size 64) (4 correct)
        ###################### Model Loss: 100 epochs 1.9423 (still improving, batch size 128) (4 correct)
        # self.model.add(Bidirectional(LSTM(256, return_sequences=False), input_shape=(self.window_size, embedding_vector_size)))
        # self.model.add(Dense(embedding_vector_size))
        ###################### Model Loss: 100 epochs 0.9503 (still improving, batch size 16) (7 correct)
        self.window_size = context_window_size
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(512, return_sequences=False), input_shape=(self.window_size, embedding_vector_size)))
        self.model.add(Dense(256))
        self.model.add(Dense(embedding_vector_size))
        ###################### Model Loss: 100 epochs 3.2725 (still improving, batch size 16)
        ###################### Model Loss: 100 epochs 3.2131 (still improving, batch size 32)
        ###################### Model Loss: 100 epochs 3.2965 (still improving, batch size 64)
        ###################### Model Loss: 100 epochs 3.3514 (still improving, batch size 128)
        # self.model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(context_window_size, embedding_vector_size)))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        # self.model.add(Bidirectional(LSTM(128, return_sequences=False)))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dense(embedding_vector_size, activation='linear'))
        self.model.compile(loss=self.euclidean_loss, optimizer='adam')

    def euclidean_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))
    
    def train_model(self, encoded_data, encoded_labels, epochs=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(encoded_data, encoded_labels.squeeze(), epochs=epochs, batch_size=32, callbacks=[early_stopping], validation_split=0.2)

    def generate_output(self, input_data):
        output_vector = self.model.predict(input_data)

        return output_vector

if __name__ == "__main__":
    context_window_size = 32
    embedding_vector_size = 50

    vocab_dict, training_data, training_labels = preprocess_data(context_window_size)
    embeddings = load_glove_embeddings(embedding_vector_size)
    training_data, training_labels = convert_to_vector_embeddings(training_data, training_labels, embeddings, embedding_vector_size)
    print(training_data.shape)
    test_amount = 10
    test_data = training_data[-test_amount:]
    training_data = training_data[:-test_amount]
    test_labels = training_labels[-test_amount:]
    training_labels = training_labels[:-test_amount]
    
    model = LSTMModel(context_window_size, embedding_vector_size)
    model.train_model(training_data, training_labels)

    correct = 0
    for i in range(1, len(test_data) + 1):
        prediction = model.generate_output(test_data[i-1:i])
        predicted_words = find_closest_word(prediction, embeddings)
        real_word = find_closest_word(test_labels[i-1:i], embeddings)
        print(predicted_words)
        print(real_word[0])
        if real_word[0] in predicted_words:
            correct += 1

    print(correct)