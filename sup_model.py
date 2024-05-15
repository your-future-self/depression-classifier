
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from catboost import CatBoostClassifier

dep = "data/dp_posts.tsv"
nondep = "data/nondp_posts.tsv"


def read_file(filename):
    text = []
    with open(filename, 'r',) as file:
        for line in file:
            if line != '\n':
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
clf = XGBClassifier()
clf.fit(X_train, y_train)

KNN_clf = KNeighborsClassifier()
KNN_clf.fit(X_train, y_train)

catboost_clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=False)
catboost_clf.fit(X_train, y_train)

# evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

y_pred = KNN_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# Explain model predictions using SHAP
# explainer = shap.Explainer(clf, X_train)
# shap_values = explainer(X_test)

# Plot summary of feature importance
# shap.summary_plot(shap_values, X_test, feature_names=vectorizer.get_feature_names_out())

# Explain individual predictions
# for i in range(5):  # Explain the first 5 test samples
#    print(f"Explaining prediction for test sample {i}:")
#    shap.plots.waterfall(shap_values[i], feature_names=vectorizer.get_feature_names_out())
