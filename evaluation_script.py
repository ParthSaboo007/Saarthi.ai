import pandas as pd
import numpy as np
import nltk
import unicodedata
import re
import pickle
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import  classification_report
from sklearn.preprocessing import LabelEncoder
from keras import backend as K


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


def get_features(total_docs):
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    X_transformed = vectorizer.transform(total_docs)
    X_transformed = X_transformed.toarray()
    return X_transformed


def get_labels(labels, column_name):
    labelencoder = LabelEncoder()
    labelencoder.classes_ = np.load(column_name + "_classes.npy", allow_pickle=True)
    labels = labelencoder.transform(labels)
    labels = to_categorical(labels)
    return labels


def test_for(df, column_name):
    X = get_features(df['transcription'])
    y = get_labels(df[column_name], column_name)
    model = load_model('text_to_' + column_name + ".h5")
    y_pred = model.predict(X)
    y_pred = y_pred.argmax(axis=-1)
    y  = y.argmax(axis=1)
    print(classification_report(y,y_pred))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def main():
    df = pd.read_csv("train_data.csv")
    df["transcription"] = df["transcription"].apply(lambda text: pre_process_text(text))
    # object model
    test_for(df, "object")
    # action model
    test_for(df, "action")
    # location model
    test_for(df, "location")


main()
