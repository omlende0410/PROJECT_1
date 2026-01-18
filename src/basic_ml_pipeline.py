# Basic ML pipeline using Pandas, NumPy, Scikit-Learn, Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = pd.read_csv("../data/study_marks.csv")

# 2. Separate features (X) and target (y)
X = data[['Hours_Studied']]
y = data['Marks']

# 3. Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model evaluation
score = model.score(X_test, y_test)
print("Model R² score:", score) # add more metrics into this , revaluation 

# 6. Prediction for new input
hours = np.array([[9]])
prediction = model.predict(hours)
print("Predicted marks for 9 hours study:", prediction[0])

# 7. Visualization
plt.scatter(X, y)                 # actual data points
plt.plot(X, model.predict(X))     # regression line
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()
