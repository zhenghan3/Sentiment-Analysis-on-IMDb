import numpy as np
import nltk
import sklearn
import datetime

from bs4 import BeautifulSoup
import re
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

imdb="IMDb"

# record the running time of the program
start = datetime.datetime.now()

nltk.download('stopwords') # If needed

# the pre-processer function of the vectorizer to remove the html symbols
def remove_html(s):
    # another way to remove html symbols while the result get no difference here
    # return BeautifulSoup(s,'html').get_text()
    return(re.sub('<[^>]*>', '', s))

# read the dataset from path into dataset_file_full
def preprocess(path_pos,path_neg):
    file = open(path_pos,encoding='utf-8').readlines()
    dataset_file_pos = ''.join(file).split("\n");
    file = open(path_neg,encoding='utf-8').readlines()
    dataset_file_neg = ''.join(file).split("\n");
    dataset_file_full = []
    for pos_review in dataset_file_pos:
        dataset_file_full.append((pos_review, 1))
    for neg_review in dataset_file_neg:
        dataset_file_full.append((neg_review, 0))
    return dataset_file_full

# random shuffle the dataset_file_full and read the data into X and Y
def random_shuffle(dataset_file_full):
    random.shuffle(dataset_file_full)
    dataset=[]
    Y=[]
    for instance in dataset_file_full:
        dataset.append(instance[0])
        Y.append(instance[1])
    return dataset,Y

# Read-in the training data set
path_pos='./'+imdb+'/train/imdb_train_pos.txt'
path_neg='./'+imdb+'/train/imdb_train_neg.txt'
training_full=preprocess(path_pos,path_neg)

# Read-in the test data set
path_pos='./'+imdb+'/test/imdb_test_pos.txt'
path_neg='./'+imdb+'/test/imdb_test_neg.txt'
test_full=preprocess(path_pos,path_neg)

# Read-in the dev data set
path_pos='./'+imdb+'/dev/imdb_dev_pos.txt'
path_neg='./'+imdb+'/dev/imdb_dev_neg.txt'
dev_full=preprocess(path_pos,path_neg)

# construct the stopwords dictionary
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

# train the data set with the clf classifier model using chi-squared test
def train_classifier(clf,X_train,Y_train,k=500):
    chi2_analysis = SelectKBest(chi2, k).fit(X_train, Y_train)
    X_train_new=chi2_analysis.transform(X_train)
    clf.fit(np.asarray(X_train_new),np.asarray(Y_train))
    return clf,chi2_analysis

# get the classification report of the test set
def get_res_test(vectorizer,classfier,chi2_analysis,bool):
    X_test = vectorizer.transform(test_set).toarray()
    Y_test_gold = np.asarray(Y_test)
    X_test = np.asarray(X_test)
    Y_test_predictions = classfier.predict(chi2_analysis.transform(X_test))
    if bool==1:
        print(classification_report(Y_test_gold,Y_test_predictions,digits=4))
        print('pred_0    pred_1')
        print('gold_0')
        print('gold_1')
        print(confusion_matrix(Y_test_gold,Y_test_predictions))
#    print(datetime.datetime.now() - start)
    return X_test

# get the accuracy of the development set
def get_res_dev(vectorizer,classfier,chi2_analysis):
    X_dev = vectorizer.transform(dev_set).toarray()
    Y_dev_gold = np.asarray(Y_dev)
    X_dev = np.asarray(X_dev)
    Y_dev_predictions = classfier.predict(chi2_analysis.transform(X_dev))
    accuracy = accuracy_score(Y_dev_gold, Y_dev_predictions)
    print(prefix+"Accuracy" + ',' + str(
        round(accuracy, 5)) + ',' + vectorizer.__class__.__name__ + ',' + classfier.__class__.__name__)
#    print(datetime.datetime.now() - start)
    return accuracy

training_set,Y_train=random_shuffle(training_full)
test_set,Y_test=random_shuffle(test_full)
dev_set,Y_dev=random_shuffle(dev_full)

# num of feature and num of chi-squared test choosed
num_features=1000
num_features_chi2=500
svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
prefix=''

# create the vectorizer of three different features
count_vectorizer_wobs = CountVectorizer(
    preprocessor=remove_html,
    stop_words = stopwords,
    lowercase = True,
    ngram_range=(1,1),
    max_features=num_features)
count_vectorizer_bi = CountVectorizer(
    preprocessor=remove_html,
    stop_words = stopwords,
    lowercase = True,
    ngram_range=(2,2),
    max_features=num_features)
tfidf_vectorizer = TfidfVectorizer(
    preprocessor=remove_html,
    stop_words = stopwords,
    lowercase = True,
    ngram_range=(1,1),
    max_features=num_features)

# feature_1 word of bags (wobs)
X_train1=count_vectorizer_wobs.fit_transform(training_set).toarray()
svm_clf1,chi2_analysis=train_classifier(svm_clf,X_train1,Y_train,num_features_chi2)
get_res_dev(count_vectorizer_wobs,svm_clf1,chi2_analysis)

# feature_2 bigrams
X_train2=count_vectorizer_bi.fit_transform(training_set).toarray()
svm_clf2,chi2_analysis=train_classifier(svm_clf,X_train2,Y_train,num_features_chi2)
get_res_dev(count_vectorizer_bi,svm_clf2,chi2_analysis)

# feature_3 tf-idf
X_train3=tfidf_vectorizer.fit_transform(training_set).toarray()
svm_clf3,chi2_analysis=train_classifier(svm_clf,X_train3,Y_train,num_features_chi2)
get_res_dev(tfidf_vectorizer,svm_clf3,chi2_analysis)

# combine the feature vectors wobs and tf-idf together
svm_clf,chi2_analysis=train_classifier(svm_clf,np.hstack((X_train1,X_train3)),Y_train,num_features_chi2)
X_test = np.hstack((count_vectorizer_wobs.transform(test_set).toarray(),tfidf_vectorizer.transform(test_set).toarray()))
Y_test_gold = np.asarray(Y_test)
X_test = np.asarray(X_test)
Y_test_predictions = svm_clf.predict(chi2_analysis.transform(X_test))
accuracy = accuracy_score(Y_test_gold, Y_test_predictions)
print(prefix + "Accuracy" + ',' + str(
    round(accuracy, 5)) + ',' + 'wobs+tf-idf')

# get the wobs vector and transform it with tf-idf
X_train=count_vectorizer_wobs.fit_transform(training_set).toarray()
trans=TfidfTransformer()
X_train=trans.fit_transform(X_train).toarray()
svm_clf,chi2_analysis=train_classifier(svm_clf,X_train,Y_train,num_features_chi2)
get_res_dev(count_vectorizer_wobs,svm_clf,chi2_analysis)

# tf-idf with (1,3) gram
tfidf_vectorizer.ngram_range=(1,3)
X_train3=tfidf_vectorizer.fit_transform(training_set).toarray()
svm_clf3,chi2_analysis=train_classifier(svm_clf,X_train3,Y_train,num_features_chi2)
get_res_dev(tfidf_vectorizer,svm_clf3,chi2_analysis)

# the best feature above is tfidf with ngram(1,3)
tfidf_vectorizer.ngram_range=(1,3)

# inspect the accuracy of different models based on tf-idf feature using the development set
log_clf = LogisticRegression(solver="liblinear")
rnd_clf = RandomForestClassifier(n_estimators=50)
svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
voting_clf = VotingClassifier(
    estimators=[('1', svm_clf), ('2', log_clf), ('3', rnd_clf)],
    voting='hard')

voting_clf_svc = VotingClassifier(
    estimators=[('1', svm_clf1), ('2', svm_clf2), ('3', svm_clf3)],
    voting='hard')
  
X_train=tfidf_vectorizer.fit_transform(training_set).toarray()
chi2_analysis = SelectKBest(chi2, num_features_chi2).fit(X_train, Y_train)
for clf in (log_clf, rnd_clf, svm_clf, voting_clf,voting_clf_svc):
    X_train_new = chi2_analysis.transform(X_train)
    clf.fit(np.asarray(X_train_new), np.asarray(Y_train))
    get_res_dev(tfidf_vectorizer, clf, chi2_analysis)

# the best model above is voting_clf classifier
# try different num_features and num_chi2 using td-idf with (1,3) gram to tune the voting_clf model
best_result=0
num_features_array=list(range(500,5001))[::500]+list(range(6000,10001))[::1000]+[20000,30000]
for num_features in num_features_array:
    num_features_chi2_array = [100]+list(range(500, num_features+1))[::500]
    tfidf_vectorizer.max_features=num_features
    X_train = tfidf_vectorizer.fit_transform(training_set).toarray()
    for num_features_chi2 in num_features_chi2_array:
        if (num_features>5000) and (num_features_chi2!=num_features):
            continue
        voting_clf,chi2_analysis=train_classifier(voting_clf,X_train,Y_train,num_features_chi2)
        prefix='num_features' + ',' + str(num_features) + ',' + 'num_chi2' + ',' + str(
            num_features_chi2) + ','
        result=get_res_dev(tfidf_vectorizer, voting_clf, chi2_analysis)
        if result>best_result:
            best_result=result
            best_num_features=num_features
            best_num_features_chi2=num_features_chi2

print('best result,num_features,'+str(best_num_features)+',num_chi2,'+str(best_num_features_chi2))
# best result tf-idf, voting_clf, num_features 30000, num_chi2 30000

# use the voting_clf and tf-idf with the best feature to get classification report of the test set
tfidf_vectorizer.max_features = best_num_features
X_train = tfidf_vectorizer.fit_transform(training_set).toarray()
voting_clf, chi2_analysis = train_classifier(voting_clf, X_train, Y_train, best_num_features_chi2)
get_res_test(tfidf_vectorizer, voting_clf, chi2_analysis,1)

# select the parameter (1500,1000) to draw the learning curve due to efficiency
best_num_features=1500
best_num_features_chi2=1000
tfidf_vectorizer.max_features = best_num_features
X_train = tfidf_vectorizer.fit_transform(training_set).toarray()
voting_clf, chi2_analysis = train_classifier(voting_clf, X_train, Y_train, best_num_features_chi2)
X_test=get_res_test(tfidf_vectorizer, voting_clf, chi2_analysis,0)
# show the learning curve of the model used for analysing the model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def plot_learning_curves(clf):
    train_errors, test_errors = [], []
    for m in range(5, len(X_train))[::100]:
        clf,chi2_analysis=train_classifier(clf,X_train[:m],Y_train[:m],best_num_features_chi2)
        Y_train_predict = clf.predict(chi2_analysis.transform(X_train[:m]))
        Y_test_predict = clf.predict(chi2_analysis.transform((X_test)))
        train_errors.append(mean_squared_error(Y_train_predict, Y_train[:m]))
        test_errors.append(mean_squared_error(Y_test_predict, Y_test))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b-", linewidth=3, label="test")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
plot_learning_curves(voting_clf)
plt.show()
