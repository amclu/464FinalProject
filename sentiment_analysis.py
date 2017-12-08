
# coding: utf-8

#create pandas dataframe from article data
import pandas as pd
import numpy as np

columns = ['text', 'source']
df = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

from json import loads
from zipfile import ZipFile

sources = ['CNN International', 'The New York Times', 'The Guardian', 'Fox News', 'BBC News', 'Bloomberg Business', 'USA Today', 'The Wall Street Journal', 'Reuters', 'CNN Money']

with ZipFile('top sites.zip') as myzip:
    for source in sources:
        for i in range(15, 30):
            with myzip.open('top sites/{}-{}.json'.format(source,i),'r') as f:
                data = f.read().decode('utf-8')
                results = loads(data)["articles"]["results"]
                for result in results:
                    df = df.append({'text':result["title"] + "\n" + result["body"], 'source':result["source"]["title"]},ignore_index=True)
                f.close()
        print ("Loaded {}".format(source))
    myzip.close()

df.tail()


#randomize order
df = df.sample(frac=1)
D = df.text.values.tolist()
c = df.source.values


print (type(D), type(c))
print (set(c))
print (len(D))
print (len(c))


SPLIT_PERC = 0.8
split_size = int(len(D)*SPLIT_PERC)
X_train = D[:split_size]
X_test = D[split_size:]
y_train = c[:split_size]
y_test = c[split_size:]


from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print (("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores)))


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

clf_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
clf_2 = Pipeline([
    ('vect', HashingVectorizer(non_negative=True)),
    ('clf', MultinomialNB()),
])
clf_3 = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])


clfs = [clf_1, clf_2, clf_3]
for clf in clfs:
    evaluate_cross_validation(clf, X_train, y_train, 5)


clf_4 = Pipeline([
    ('vect', CountVectorizer(
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",
    )),
    ('clf', MultinomialNB()),
])


evaluate_cross_validation(clf_4, X_train, y_train, 5)


def get_stop_words():
    result = set()
    for line in open('stopwords_en.txt', 'r').readlines():
        result.add(line.strip())
    return result


stop_words = get_stop_words()


clf_5 = Pipeline([
    ('vect', CountVectorizer(
                stop_words=stop_words,
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",    
    )),
    ('clf', MultinomialNB()),
])


evaluate_cross_validation(clf_5, X_train, y_train, 5)

clf_7 = Pipeline([
    ('vect', CountVectorizer(
                stop_words=stop_words,
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",         
    )),
    ('clf', MultinomialNB(alpha=0.01)),
])


evaluate_cross_validation(clf_7, X_train, y_train, 5)

from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    mcm = metrics.confusion_matrix(y_test, y_pred)
    print(mcm)
    return mcm


mcm = train_and_evaluate(clf_7, X_train, X_test, y_train, y_test)

import matplotlib.pyplot as plt

def plotCM(conf_arr, labels, name):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), labels[:width], rotation=17, fontsize='small')
    plt.yticks(range(height), labels[:height], rotation=17, fontsize='small')
    plt.savefig(name, format='png')


plotCM(mcm, sorted(set(c)), 'top_NB_conf.png')


#print (len(clf_7.named_steps['vect'].get_feature_names()))


def print_top10(clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = clf.named_steps['vect'].get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.named_steps['clf'].coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))
        for k in range(0,10):
            print(clf.named_steps['clf'].coef_[i][top10[k]])


print_top10(clf_7, sorted(set(c)))


from sklearn.svm import LinearSVC

clf_8 = Pipeline([
    ('vect', CountVectorizer(
                stop_words=stop_words,
                token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b",         
    )),
    ('clf', LinearSVC()),
])


evaluate_cross_validation(clf_8, X_train, y_train, 5)


mcm = train_and_evaluate(clf_8, X_train, X_test, y_train, y_test)


#plotCM(mcm, sorted(set(c)), 'top_SVM_conf.png')


print_top10(clf_8, sorted(set(c)))
