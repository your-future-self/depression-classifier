
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dep = "data/dp_posts.tsv"
nondep = "data/nondp_posts.tsv"


def read_file(filename):
    text = []
    with open(filename, 'r',) as file:
        for line in file:
            if line != '\n':
                line = line.replace("\t", "")
                text.append(line.rstrip())
    return text


# read the files
depressed_posts = read_file(dep)
non_depressed_posts = read_file(nondep)

# create labels
depressed_labels = [1] * len(depressed_posts)
non_depressed_labels = [0] * len(non_depressed_posts)

all_posts = depressed_posts + non_depressed_posts
all_labels = depressed_labels + non_depressed_labels

# vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_posts)
y = all_labels

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build the classifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

KNN_clf = KNeighborsClassifier()
KNN_clf.fit(X_train, y_train)

catb_clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=False)
catb_clf.fit(X_train, y_train)

svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# evaluate the classifiers

y_pred_xgboost = xgb_clf.predict(X_test)
acc_xg = accuracy_score(y_test, y_pred_xgboost)
print("XGBoost Accuracy:", acc_xg)
xgboost_report = classification_report(y_test, y_pred_xgboost, output_dict=True)
print(xgboost_report)

y_pred_knn = KNN_clf.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", acc_knn)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True)
print(knn_report)

y_pred_catboost = catb_clf.predict(X_test)
acc_cat = accuracy_score(y_test, y_pred_catboost)
print("CatBoost Accuracy:", acc_cat)
catboost_report = classification_report(y_test, y_pred_catboost, output_dict=True)
print(catboost_report)

y_pred_svm = svm_clf.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", acc_svm)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True)
print(svm_report)

accuracy_values = {
    'XGBoost': acc_xg,
    'KNN': acc_knn,
    'CatBoost': acc_cat,
    'SVM': acc_svm
}
models = list(accuracy_values.keys())
accuracies = list(accuracy_values.values())

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])

# Add title and labels
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')

# Show plot
plt.show()
