import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('ModuleDataset_processed.csv')
print(dataset_train.columns)
test_num=23
training_set = dataset_train.iloc[:-test_num,2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with timesteps and t+1 output
X_train = []
y_train = []
timestep=12

for i in range(timestep, training_set.shape[0]):
    X_train.append(training_set_scaled[i-timestep:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape,y_train.shape)
# print(X_train)
# print(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1),return_sequences=True)) # if >1 lstm layer put return_sequences=True here

# Adding a second LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True))

# Adding a third LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True))

# # Adding a fourth LSTM layer
regressor.add(LSTM(units = 3)) #last lstm layer return_sequences=False for Next dense layer


# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 12)

# test_num=863-24
dataset_test = dataset_train.iloc[-test_num:,:]
test_set = dataset_test.iloc[:,2:3].values
real_rainfall = np.concatenate((dataset_train.iloc[:-test_num,2:3].values,test_set),axis = 0)
print(real_rainfall.shape)
scaled_real_rainfall = sc.fit_transform(real_rainfall)

inputs = []
for i in range(len(dataset_train)-test_num, 863+1):
    inputs.append(scaled_real_rainfall[i-timestep:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_rainfall = regressor.predict(inputs)
predicted_rainfall = sc.inverse_transform(predicted_rainfall)
print(predicted_rainfall.shape)

# Visualising the results
plt.plot(real_rainfall[len(dataset_train)-test_num:], color = 'red', label = 'Real')
plt.plot(predicted_rainfall, color = 'blue', label = 'Predicted')
plt.title('Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.tight_layout()
plt.show()

test_num=len(dataset_train)-24
dataset_test = dataset_train.iloc[-test_num:,:]
test_set = dataset_test.iloc[:,2:3].values
real_rainfall = np.concatenate((dataset_train.iloc[:-test_num,2:3].values,test_set),axis = 0)
print(real_rainfall.shape)
scaled_real_rainfall = sc.fit_transform(real_rainfall)

inputs = []
for i in range(len(dataset_train)-test_num, len(dataset_train)+1):
    inputs.append(scaled_real_rainfall[i-timestep:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_rainfall = regressor.predict(inputs)
predicted_rainfall = sc.inverse_transform(predicted_rainfall)
print(predicted_rainfall.shape)

# Visualising the results
plt.plot(real_rainfall[len(dataset_train)-test_num:], color = 'red', label = 'Real')
plt.plot(predicted_rainfall, color = 'blue', label = 'Predicted')
plt.title('Rainfall Prediction')
plt.xlabel('Time')
plt.ylabel('Rainfall')
plt.legend()
plt.tight_layout()
plt.show()