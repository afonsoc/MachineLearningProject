import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')


st.write("Stock Market APP")


#creating sidebar
st.sidebar.header('User input')



#get users input
def get_input():
    start_date = st.sidebar.text_input("Start Date MM/DD/YYY", "04-01-2018")
    end_date = st.sidebar.text_input("End Date MM/DD/YYY", "01-01-2021")
    ticker = st.sidebar.text_input("Ticker", "AMZN")
    return start_date, end_date, ticker


#function to choose company by ticker
def get_compticker(ticker):
    if ticker == 'NNDM':
        return 'Nano Dimensions'
    elif ticker == 'TSLA':
        return 'Tesla'
    elif ticker == 'AAPL':
        return 'Apple'
    elif ticker == 'AMD':
        return 'AMD'
    elif ticker == 'AMZN':
        return 'Amazon'
    elif ticker == 'APHA':
        return 'Aphria Inc'
    elif ticker == 'CRM':
        return 'Salesforce'
    elif ticker == 'CRSP':
        return 'CRISPR Therapeutics'
    elif ticker == 'FVRR':
        return 'Fiverr International'
    elif ticker == 'HIMX':
        return 'Himax Technologies, Inc.'
    elif ticker == 'NIO':
        return 'NIO Limited'
    elif ticker == 'NVDA':
        return 'Nvidia Corporation'
    elif ticker == 'NVTA':
        return 'InVitae Corporation'
    elif ticker == 'PLTR':
        return 'Palantir Technologies'
    elif ticker == 'PTON':
        return 'Peloton Interactive, Inc.'
    elif ticker == 'PYPL':
        return 'Paypal Holdings, Inc.'
    elif ticker == 'QS':
        return 'QuantumScape Corporation'
    elif ticker == 'SQ':
        return 'Square, Inc.'
    elif ticker == 'XPEV':
        return 'XPeng Inc.'
    elif ticker == 'Z':
        return 'Zillow Group, Inc.'
    else:
        return 'No ticker'

#function to load data into the dataframe
def get_data(ticker, start, end):


    if ticker.upper() == 'NNDM':
        df = pd.read_csv("stocks/NNDM.csv")
    elif ticker.upper() == 'TSLA':
        df = pd.read_csv("stocks/TSLA.csv")
    elif ticker.upper() == 'AAPL':
        df = pd.read_csv("stocks/AAPL.csv")
    elif ticker.upper() == 'AMD':
        df = pd.read_csv("stocks/AMD.csv")
    elif ticker.upper() == 'AMZN':
        df = pd.read_csv("stocks/AMZN.csv")
    elif ticker.upper() == 'APHA':
        df = pd.read_csv("stocks/APHA.csv")
    elif ticker.upper() == 'CRM':
        df = pd.read_csv("stocks/CRM.csv")
    elif ticker.upper() == 'CRSP':
        df = pd.read_csv("stocks/CRSP.csv")
    elif ticker.upper() == 'FVRR':
        df = pd.read_csv("stocks/FVRR.csv")
    elif ticker.upper() == 'HIMX':
        df = pd.read_csv("stocks/HIMX.csv")
    elif ticker.upper() == 'NIO':
        df = pd.read_csv("stocks/NIO.csv")
    elif ticker.upper() == 'NVDA':
        df = pd.read_csv("stocks/NVDA.csv")
    elif ticker.upper() == 'NVTA':
        df = pd.read_csv("stocks/NVTA.csv")
    elif ticker.upper() == 'PLTR':
        df = pd.read_csv("stocks/PLTR.csv")
    elif ticker.upper() == 'PTON':
        df = pd.read_csv("stocks/PTON.csv")
    elif ticker.upper() == 'PYPL':
        df = pd.read_csv("stocks/PYPL.csv")
    elif ticker.upper() == 'QS':
        df = pd.read_csv("stocks/QS.csv")
    elif ticker.upper() == 'SQ':
        df = pd.read_csv("stocks/SQ.csv")
    elif ticker.upper() == 'XPEV':
        df = pd.read_csv("stocks/XPEV.csv")
    elif ticker.upper() == 'Z':
        df = pd.read_csv("stocks/Z.csv")
    else:
        df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    #get date range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    #set rows at 0
    start_row = 0
    end_row = 0


    for i in range(0, len(df)):
        if start <=pd.to_datetime(df['Date'][i]):
            start_row = i
            break

    for j in range(0, len(df)):
        if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
            end_row = len(df) -1 -j
            break

#
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))


    return df.iloc[start_row:end_row +1, :]



#variables that store the user input are being defined here
start, end, ticker = get_input()

#created a varaible to store get_data function
df = get_data(ticker, start, end)



data = df.filter(['Close'])
datasettest = data.values
training_data_len = math.ceil(len(datasettest)* .8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(datasettest)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 60:
        print(x_train)
        print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

#make the array 3 dimensional
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = datasettest[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'upper left')

st.pyplot()

#created a variable to store get_compticker function
compticker = get_compticker(ticker.upper())

st.header('RMSE')
st.write(rmse)

st.header('Scaled Data')
scaled_data
st.line_chart(scaled_data)

st.header(compticker + " Close Price\n")
st.area_chart(df['Close'])

st.header(compticker + " Volume\n")
st.bar_chart(df['Volume'])

st.header('Data Statistics')
st.write(df.describe())


