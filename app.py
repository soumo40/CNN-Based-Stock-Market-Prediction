#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error




# In[51]:


def testgen(data, seq_len, targetcol):
    #Return array of all test samples
    batch = []
    df=data
    input_cols = [c for c in df.columns if c != targetcol]
    # extract sample using a sliding window
    for i in range(len(df), len(df)+1):
        frame = df.iloc[i-seq_len:i]
        batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)


# In[52]:


# Create flask app
flask_app = Flask(__name__)
model = tf.keras.models.load_model('mymodel.h5', compile=False)     
model.compile()

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/pipeline", methods = ["POST"])
def pipeline():
    
    f = request.files['data']
    X = pd.read_csv(f, index_col="Date", parse_dates=True)
      
    
    name = X["Name"][0]
    del X["Name"]
    # Extracting column names
    cols = X.columns
    # Save the target variable as a column in dataframe column in dataframe for easier dropna()
    X["Target"] = (X["Close"].pct_change().shift(-1) > 0).astype(int)
    X.dropna(inplace=True)        
    # Fit the Standard Scalar
    scaler = StandardScaler().fit(X.loc[:, cols])
    # Save scale transformed dataframe
    X[cols] = scaler.transform(X[cols])
    data = X       
    
    
    seq_len = 60
    batch_size = 128
    n_epochs = 20
    n_features = 82
    
    # Prepare test data
    test_data, test_target = testgen(data, seq_len, "Target")
        
    # Test the model
    test_out = model.predict(test_data)
    test_pred = (test_out > 0.5).astype(int)
    if test_pred==[[1]]:
        output='Positive'
    else:
        output='Negative'

    return render_template("index.html", prediction_text = f"The change in NIFTY50 for the next day is going to be {output}")

if __name__ == "__main__":
    flask_app.run(debug = True)

