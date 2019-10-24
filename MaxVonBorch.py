import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# import preprocessed dataset
data = pd.read_csv(".../datascience_challenge.csv")
data = data.drop(["Sample Age (Weeks)", "Transparent Window in Package"], axis=1)

# automatically add dummy colums for appropriate columns
dummy_cols = ["Product Type", "Base Ingredient", "Process Type", "Storage Conditions"]
train = pd.get_dummies(data, columns=dummy_cols)

# encode binary variables
cleanup_nums = {"Packaging Stabilizer Added":     {"Y": 1, "N": 0, "NaN":0},
             "Preservative Added":     {"Y": 1, "N": 0, "NaN":0}}

train.replace(cleanup_nums, inplace=True)
train.replace(np.NaN, 0, inplace=True)

train = train.drop(["Study Number", "Sample ID", "shelf life"], axis=1)


#split into training and testing set
training = train.sample(n=int(0.8*len(train)))
testing = train.drop(training.index)

x_train=training.drop("Class", axis=1)
y_train=training["Class"]
x_test=testing.drop("Class", axis=1)
y_test=testing["Class"]

#scaler = StandardScaler()
#x_train=scaler.fit_transform(x_train)
#x_test=scaler.transform(x_test)

#hyperparameter tuning using CV
kernels = ["linear", "rbf"]
# gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set
gamma = [0.1, 0.2, 0.5, 1,5, 10]
#C : C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
C = [1,5, 10, 25, 30, 40, 50, 100, 1000]

random_grid = {'C': C,
               'kernel': kernels,
                'gamma': gamma}

SVM = SVC()
SVM_random = RandomizedSearchCV(estimator = SVM, param_distributions = random_grid, n_iter = 1000, cv = 3,  random_state=42, scoring="f1_weighted")
SVM_random.fit(x_train, y_train)
SVM_random.best_params_

# train and test model
svm = SVC(gamma="auto", kernel="rbf", C=1000, random_state=1)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
print(classification_report(y_test, y_pred))

# test the whole dataset (each sample)
y_pred = svm.predict(train.drop("Class", axis=1))
print(classification_report(train["Class"], y_pred))