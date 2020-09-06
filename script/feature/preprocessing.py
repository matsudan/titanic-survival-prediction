import pandas as pd


def preprocessing(train, test):
    data = pd.concat([train, test], sort=False)

    # encoding categorical features
    data["Sex"].replace(["male", "female"], [0, 1], inplace=True)
    data["Embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)

    # imputation of missing values
    data["Embarked"].fillna(data.Embarked.median(), inplace=True)
    data["Fare"].fillna(data.Fare.median(), inplace=True)
    data["Age"].fillna(data.Age.median(), inplace=True)

    # drop columns
    delete_columns = ["Name", "PassengerId",  "Ticket", "Cabin"]
    data.drop(delete_columns, axis=1, inplace=True)

    f_train = data[:len(train)]
    f_test = data[len(train):]

    y_train = f_train["Survived"]
    x_train = f_train.drop("Survived", axis=1)
    x_test = f_test.drop("Survived", axis=1)

    return y_train, x_train, x_test

