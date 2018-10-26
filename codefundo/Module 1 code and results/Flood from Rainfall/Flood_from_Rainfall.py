import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset = pd.read_csv('ModuleDataset_processed.csv')
print(dataset.columns)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:,2:3], y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Visualising the Training set results
plt.scatter(sc.inverse_transform(X_train),y_train)
plt.plot(np.arange(1100),classifier.predict(sc.transform(np.arange(1100).reshape(-1,1))))
plt.show()

# Visualising the Training set results
plt.scatter(sc.inverse_transform(X_test),y_test)
plt.plot(np.arange(1100),classifier.predict(sc.transform(np.arange(1100).reshape(-1,1))))
plt.show()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)