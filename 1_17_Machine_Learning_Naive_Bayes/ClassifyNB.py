from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf

