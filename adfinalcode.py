import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("ADproject.csv")

x = df.iloc[:, :10]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

reg = RandomForestRegressor(n_estimators=100, random_state=10)  # You can adjust hyperparameters like n_estimators

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("R-squared score:", r2)

prediction = reg.predict([[3, 3, 1, 1, 1, 18,9, 6, 24, 4]])
print("Prediction:", prediction)
