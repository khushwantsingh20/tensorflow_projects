import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Example dataset: Sine wave
data = np.sin(np.linspace(0, 100, 1000))

# Reshape data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Function to create dataset with time steps
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Prepare the dataset
time_step = 50  # Number of time steps for LSTM
X, y = create_dataset(data_scaled, time_step)

# Reshape X for LSTM input [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.67)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
predicted = model.predict(X_test)

# Inverse scale the predictions
predicted = scaler.inverse_transform(predicted)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.plot(y_test_rescaled, color='blue', label='Actual Data')
plt.plot(predicted, color='red', label='Predicted Data')
plt.title('Time Series Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Save the plot instead of showing
plt.savefig('time_series_prediction_lstm.png')  # Save to a file
plt.close()  # Close the plot to free up memory
