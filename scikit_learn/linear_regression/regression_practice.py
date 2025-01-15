from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
X = [[1], [2], [3], [4], [5]]
y = [1.5, 3.7, 4.2, 6.8, 9.5]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
