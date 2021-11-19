# Saarthi.ai

### train.py 
input = train.csv
output = (models = text_to_object.h5,  text_to_action.h5,  text_to_location.h5), (labelencoder_action.npy, labelencoder_object.npy, labelencoder_location.npy), vectorizer = vectorizer.pickle

### In evaluation.py
input = (models = text_to_object.h5,  text_to_action.h5,  text_to_location.h5), (labelencoder_action.npy, labelencoder_object.npy, labelencoder_location.npy), vectorizer = vectorizer.pickle

output = print(f1_scores)
