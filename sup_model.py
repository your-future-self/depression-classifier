from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
from catboost import CatBoostClassifier
import csv
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import pandas as pd
from collections import defaultdict
from nltk.help import upenn_tagset


def read_file(filename):
    text = []
    with open(filename, "r") as file:
        for line in file:
            if line:
                text.extend(line.strip("\t\n ").split("\t"))
    return text


def tag_time(corpus):
    output = []
    for line in corpus:
        tokens = pos_tag(word_tokenize(line))
        just_tags = [token[1] for token in tokens]
        tag_counter = defaultdict(int)
        for tag in just_tags:
            tag_counter[tag] += 1
        output.append(tag_counter)
    return output


def main():
    dep = "depression.tsv"
    nondep = "control.tsv"

    # read the files
    depressed_posts = read_file(dep)
    non_depressed_posts = read_file(nondep)
    depressed_posts = tag_time(depressed_posts)
    non_depressed_posts = tag_time(non_depressed_posts)

    # create labels
    depressed_labels = [1] * len(depressed_posts)
    non_depressed_labels = [0] * len(non_depressed_posts)

    all_posts = depressed_posts
    all_posts.extend(non_depressed_posts)
    df = pd.DataFrame(all_posts)
    df = df.fillna(0)
    df.dtypes
    df = df.astype(int)
    len(all_labels)
    all_labels = depressed_labels + non_depressed_labels
    df["labels"] = all_labels
    df
    print(all_posts[:3], type(all_posts))

    # # vectorize the text data
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(all_posts)
    # y = all_labels
    X = df.drop(columns="labels")
    y = df["labels"]
    X

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # build the classifier
    clf = XGBClassifier()
    clf.fit(X_train, y_train)

    KNN_clf = KNeighborsClassifier()
    KNN_clf.fit(X_train, y_train)

    catboost_clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1, verbose=False
    )
    catboost_clf.fit(X_train, y_train)

    # evaluate the classifier
    print("XGboost")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    y_pred = KNN_clf.predict(X_test)
    print("KNN")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    print("catboost")
    y_pred = catboost_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Explain model predictions using SHAP
    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_test)

    # Plot summary of feature importance
    shap.summary_plot(shap_values, X_test)


upenn_tagset()


# Explain individual predictions
# for i in range(5):  # Explain the first 5 test samples
#    print(f"Explaining prediction for test sample {i}:")
#    shap.plots.waterfall(shap_values[i], feature_names=vectorizer.get_feature_names_out())

if __name__ == "__main__":
    main()
