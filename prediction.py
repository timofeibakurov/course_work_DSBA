import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

ticker = ''

# The stocks we'll use for this analysis
tech_list = ['MSFT', 'NVDA', 'ADBE', 'INTC', 'CSCO']
minig_list = ['XOM', 'CVX', 'BHP', 'BP', 'ENB']
prod_list = ['HON', 'LMT', 'CAT', 'UNP', 'UPS']
finnance_list = ['JPM', 'BAC', 'C', 'GS', 'MS']
food_list = ['KO', 'PEP', 'PG', 'UL', 'GIS']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download stock data and assign to variables
NVDA = yf.download('NVDA', start, end)
MSFT = yf.download('MSFT', start, end)
ADBE = yf.download('ADBE', start, end)
INTC = yf.download('INTC', start, end)
CSCO = yf.download('CSCO', start, end)

XOM = yf.download('XOM', start, end)
CVX = yf.download('CVX', start, end)
BHP = yf.download('BHP', start, end)
BP = yf.download('BP', start, end)
ENB = yf.download('ENB', start, end)

HON = yf.download('HON', start, end)
LMT = yf.download('LMT', start, end)
CAT = yf.download('CAT', start, end)
UNP = yf.download('UNP', start, end)
UPS = yf.download('UPS', start, end)

JPM = yf.download('JPM', start, end)
BAC = yf.download('BAC', start, end)
C = yf.download('C', start, end)
GS = yf.download('GS', start, end)
MS = yf.download('MS', start, end)

KO = yf.download('KO', start, end)
PEP = yf.download('PEP', start, end)
PG = yf.download('PG', start, end)
UL = yf.download('UL', start, end)
GIS = yf.download('GIS', start, end)

# Create a list of company dataframes
tech_company_list = [MSFT, NVDA, ADBE, INTC, CSCO]
tech_company_name = ['Microsoft Corporation', 'NVIDIA Corporation', 'Adobe Inc.', 'Intel Corporation', 'Cisco Systems, Inc.']

mining_company_list = [XOM, CVX, BHP, BP, ENB]
mining_company_name = ['Exxon Mobil Corporation', 'Chevron Corporation', 'BHP Group', 'BP p.l.c.', 'Enbridge Inc.']

prod_company_list = [HON, LMT, CAT, UNP, UPS]
prod_company_name = ['Honeywell International Inc.', 'Lockheed Martin Corporation', 'Caterpillar Inc.', 'Union Pacific Corporation', 'United Parcel Service, Inc.']

finance_company_list = [JPM, BAC, C, GS, MS]
finance_company_name = ['JPMorgan Chase & Co.', 'Bank of America Corporation', 'Citigroup Inc.', 'The Goldman Sachs Group, Inc.', 'Morgan Stanley']

food_company_list = [KO, PEP, PG, UL, GIS]
food_company_name = ['The Coca-Cola Company', 'PepsiCo, Inc.', 'Procter & Gamble Company', 'Unilever PLC', 'General Mills, Inc.']



# Add company name to each dataframe
for company, com_name in zip(tech_company_list, tech_company_name):
    company["company_name"] = com_name
df_tech = pd.concat(tech_company_list, axis=0)

for company, com_name in zip(mining_company_list, mining_company_name):
    company["company_name"] = com_name
df_mining = pd.concat(mining_company_list, axis=0)

for company, com_name in zip(prod_company_list, prod_company_name):
    company["company_name"] = com_name
df_prod = pd.concat(prod_company_list, axis=0)

for company, com_name in zip(finance_company_list, finance_company_name):
    company["company_name"] = com_name
df_finance = pd.concat(finance_company_list, axis=0)

for company, com_name in zip(food_company_list, food_company_name):
    company["company_name"] = com_name
df_food = pd.concat(food_company_list, axis=0)






# Grab all the closing prices for the tech stock list into one DataFrame

closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

# Make a new tech returns DataFrame
tech_rets = closing_df.pct_change()

# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# Get the stock quote
df = pdr.get_data_yahoo(ticker, start='2012-01-01', end=datetime.now())

res_plt = plt.figure(figsize=(16, 6))
res_plt = res_plt.title('Close Price History')
res_plt = res_plt.plot(df['Close'])
res_plt = res_plt.xlabel('Date', fontsize=18)
res_plt = res_plt.ylabel('Close Price USD ($)', fontsize=18)

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .95))

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
fin_plt = plt.figure(figsize=(16,6))
fin_plt = fin_plt.title('Model')
fin_plt = fin_plt.xlabel('Date', fontsize=18)
fin_plt = fin_plt.ylabel('Close Price USD ($)', fontsize=18)
fin_plt = fin_plt.plot(train['Close'])
fin_plt = fin_plt.plot(valid[['Close', 'Predictions']])
fin_plt = fin_plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')