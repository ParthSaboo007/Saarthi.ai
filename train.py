import datetime
import os
import pandas as pd
import numpy as np
import nltk
import unicodedata
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
from mxnet import npx
from keras.models import Sequential
from keras.layers import Dense
import pickle

from sklearn.preprocessing import LabelEncoder

vectorizer_count = False


def pre_process_text(text):
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['ok']
    stopwords.extend(newStopWords)
    text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore')
            .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    text = [wnl.lemmatize(word) for word in words if word not in stopwords]
    text = " ".join(text)
    return text


def get_features_and_labels(text_docs, labels):
    global vectorizer_count
    x1 = text_docs
    y1 = labels
    X_train, X_val, y_train, y_val = train_test_split(x1, y1, test_size=0.25, random_state=42)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 stop_words='english', ngram_range=(1, 5))
    if not vectorizer_count:
        X_train_transformed = vectorizer.fit_transform(X_train)
        vectorizer_count = True
        X_test_transformed = vectorizer.transform(X_val)
        X_train_transformed = X_train_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()
        pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
        return X_train_transformed, X_test_transformed, y_train, y_val
    else:
        vectorizer = pickle.load(open("vectorizer.pickle","rb"))
        X_train_transformed = vectorizer.transform(X_train)
        X_test_transformed = vectorizer.transform(X_val)
        X_train_transformed = X_train_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()
        pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
        return X_train_transformed, X_test_transformed, y_train, y_val





def model(input_shape, output_shape):
    classifier = Sequential()
    classifier.add(Dense(1000, activation='relu', input_dim=input_shape))
    classifier.add(Dense(500, activation='relu'))
    classifier.add(Dense(output_shape, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


def train_model(X_train, X_val, y_train, y_val, logdir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    n_gpus = npx.num_gpus()
    print(n_gpus)
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(
        device_type)
    devices_names = [d.name.split("e:")[1] for d in devices]
    strategy = tf.distribute.MirroredStrategy(
        devices=devices_names[:n_gpus])
    if n_gpus:
        with strategy.scope():
            classifier = model(X_train[1].shape[1], y_train[1].shape[1])
            history = classifier.fit(X_train, y_train, batch_size=128,
                                     validation_data=(X_val, y_val),
                                     epochs=100, verbose=1)
            return classifier
    else:
        print(int(X_train.shape[1]), int(y_train.shape[1]))
        classifier = model(X_train.shape[1], y_train.shape[1])
        history = classifier.fit(X_train, y_train, batch_size=128,
                                 validation_data=(X_val, y_val), callbacks=[tensorboard_callback],
                                 epochs=1, verbose=1)
        return classifier


def train_for(df, column_name):
    labelencoder = LabelEncoder()

    df[column_name] = labelencoder.fit_transform(df[column_name])
    np.save(column_name + "_classes.npy", labelencoder.classes_)
    X_train_transformed, X_test_transformed, y_train, y_val = get_features_and_labels(df["transcription"],
                                                                                      df[column_name].values)
    logdir = os.path.join(column_name + "_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = train_model(X_train_transformed, X_test_transformed, y_train, y_val, logdir)
    model.save("text_to_" + column_name + ".h5")


def main():
    df = pd.read_csv("train_data.csv")
    df["transcription"] = df["transcription"].apply(lambda text: pre_process_text(text))
    # object model
    train_for(df, "object")
    # action model
    train_for(df, "action")
    # location model
    train_for(df, "location")


main()
