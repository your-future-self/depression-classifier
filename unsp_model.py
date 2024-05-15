
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np
from sup_model import read_file

dep = "data/dp_posts.tsv"
nondep = "data/nondp_posts.tsv"

depressed_posts = read_file(dep)
print(depressed_posts[:2])
print(len(depressed_posts))
non_depressed_posts = read_file(nondep)
print(non_depressed_posts[:2])
print(len(non_depressed_posts))


all_posts = depressed_posts + non_depressed_posts

# vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_posts)

num_clusters = 2  # Assume we want to group the posts into 2 clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

clusters = kmeans.labels_

# Get the posts in each cluster
cluster_0 = np.where(clusters == 0)[0]
cluster_1 = np.where(clusters == 1)[0]

print("Cluster 0 posts:")
for idx in cluster_0[:5]:  # Print first 5 posts in cluster 0
    print(all_posts[idx])
    print()

print("Cluster 1 posts:")
for idx in cluster_1[:5]:  # Print first 5 posts in cluster 1
    print(all_posts[idx])
    print()

num_topics = 2  # Assume we want to discover 2 topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Display the top words in each topic
tf_feature_names = vectorizer.get_feature_names_out()


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


display_topics(lda, tf_feature_names, 10)
