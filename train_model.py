import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a pickle file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
