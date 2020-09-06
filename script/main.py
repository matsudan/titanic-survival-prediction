import pandas as pd
from feature.preprocessing import preprocessing
from model.lgbm import objectives

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

y_train, x_train, x_test = preprocessing(train=train, test=test)

categorical_features = ["Sex", "Embarked", "Pclass"]

objectives(x_train=x_train, y_train=y_train, x_test=x_test, categorical_features=categorical_features)
