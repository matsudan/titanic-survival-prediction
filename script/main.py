import pandas as pd
from feature.preprocessing import preprocessing
from model.lgbm import tuning
from model.lgbm import predict


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
sub = pd.read_csv("../input/titanic/gender_submission.csv")

categorical_features = ["Sex", "Embarked", "Pclass"]

y_train, x_train, x_test = preprocessing(train=train, test=test)

best_params = tuning(x_train=x_train, y_train=y_train, categorical_features=categorical_features)

predict(best_params, x_train, y_train, x_test, categorical_features, sub)
