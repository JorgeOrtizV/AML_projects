import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        #self.classifier = RandomForestClassifier(n_estimators=200, random_state=42) #KNeighborsClassifier(n_neighbors=5)
        self.classifier = KNeighborsClassifier(n_neighbors=3)

    def train(self, train_x, train_y):
        self.classifier.fit(train_x, train_y)

    def predict(self, x):
        return self.classifier.predict(x)


def dataProcessing():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_x = train_df.drop(columns=["id", "target"]).to_numpy()
    train_y = train_df["target"].to_numpy()
    test_x = test_df.drop(columns=["id"]).to_numpy()

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, train_y, test_x

def split_indices(n, val_pct):
    n_val = int(val_pct*n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def acc_measurement(prediction, truth):
    return accuracy_score(truth, prediction)


if __name__ == "__main__":
    np.random.seed(32)

    train_x, train_y, test_x = dataProcessing()
    #train_idxs, val_idxs = split_indices(len(train_x), 0.2)
    #val_x = train_x[val_idxs]
    #val_y = train_y[val_idxs]
    #train_x = train_x[train_idxs]
    #train_y = train_y[train_idxs]

    classifier = Model()
    classifier.train(train_x, train_y)
    predictions = classifier.predict(test_x)

    #print("Accuracy : {}".format(accuracy_score(predictions, val_y)))
    # Create a submission file
    test_df = pd.read_csv('test.csv')
    submission = pd.DataFrame({'id': test_df['id'], 'target': predictions})
    submission.to_csv('submission2.csv', index=False)

