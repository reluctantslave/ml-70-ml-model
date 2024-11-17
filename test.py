
import pickle
import tensorflow as tf
import pandas as pd
import scipy
from tensorflow import keras

if __name__ == '__main__':
    model = keras.models.load_model('in_the_name_of_allah.keras')
    
    # Load the vectorizer later
    with open("vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    
    query = vectorizer.transform([" select * from users where id  =  '1' or $ 1  =  1 union select 1,@@VERSION -- 1'"])
    transformed_query = scipy.sparse.csr_matrix.todense(query)
    prediction = model.predict(transformed_query)
    
    print(prediction)
    
    