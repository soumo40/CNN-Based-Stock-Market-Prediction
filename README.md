
This repository stores all the code which is a part of the project 'CNN based stock market prediction'

## Week 1-3: Checkpoint 1
First, I explored the different concepts related to Time Series modelling such as seasonality, trend and long term oscillations. This was followed by different classifications of time series data. Next, I learnt about the different methods associated with time series modelling and when to use them. These include AR, MA, ARIMA, SARIMA, Exponential Smoothing etc. This was followed by learning about basic concepts related to machine learning and deep learning. These include regression, regularisation, classification, ensemble techniques and ANNs.

## Week 5
Next, I moved to learning about Convolutional Neural Networks. Here, I performed a CNN based image classification for the presence of viral pneumonia in CT scans. I learnt about the basic algorithms and architecture of CNN. Following this, I moved to the project and started with the process of Exploratory Data Analysis for the dataset. Link to the dataset can be found at https://machinelearningmastery.com/?attachment_id=13057

## Week 6
I read about the paper on CNNpred: CNN-based stock market prediction using a diverse set of variables (https://www.sciencedirect.com/science/article/pii/S0957417419301915). The author explained two methodolgies using 2D CNN and 3D CNN which to predict stock market fluctuations.

## Week 7
Next, I implemented the paper using Tensorflow.keras. I read about the different types of feature engineering and tried implementing a few. Some relevant feature engineering in the case of financial markets would be Moving Average, Relative Change of Volume etc.

## Week 8 and 9
I tried to tweak the model and improve on the performance. For this I tried increasing the number of epochs, changing the filter sizes, adding number of layers, changing the dropout rate and using classic CNN architectures like LeNet-5 and AlexNet (modified it to suit this applications) in order to predict the stock market fluctuations. The CNN model can also be improved using Hyperas.

## Week 10
The model was automated using Flask and tested on new market data in NIFTY50. For running the model, download app.py, templates and mymodel.h5 from https://drive.google.com/file/d/1Geb878SrcgZ4tYICqFVP22DuGOyK-jvA/view and save in a directory. Next, run the python application and go to http://127.0.0.1:5000 where you will be asked to upload the file. Upload Processed_NSE (from the data folder) as an example to check the future fluctuation.

## Instructions for running code
All code can be run on Jupyter Notebooks or Google Colab. The app.py file needs to be run on VS code with anaconda environment.

