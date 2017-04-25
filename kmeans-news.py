#!/usr/bin/python3

# This program will calculate k-means clustering from headlines gathered
# from newsapi.org.  K-value needs to be passed in as a command line argument.
# Keep in mind that headlines will change often and no outcome of clustering
# can be predicted.

# For UMSL CS 4342

# Requires requests library and python3
# Make sure the file is executable with:
#    chmod +x kmeans-news.py
# Then, run the file with
# ./kmeans-news.py INTEGER

# I have found larger K values work better for this dataset. 15-20 seems
# to yield expected results often


from collections import defaultdict
from math import sqrt
from random import randint
import string
import sys
from urllib.parse import urlencode

import requests


STOPWORDS = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',
    'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
    'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers',
    'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its',
    'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so',
    'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
    "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll",
    "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's",
    'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why',
    "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll",
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', "'n", "'s",
    "'t", '', '…', '_', '—',
]

NEWSAPIORG_KEY = 'XXXX'
SOURCES_URL = 'https://newsapi.org/v1/sources?language=en'
ARTICLES_URL = 'https://newsapi.org/v1/articles?'


def _get_sources():
    """Get sources from newsapi.org"""
    resp = requests.get(SOURCES_URL)
    data = resp.json()
    sources = [s['id'] for s in data['sources']]

    return sources


def _clean_text(document):
    """Transform text into list of feature words"""
    document = document.lower()

    # Remove punctuation
    table = str.maketrans({key: ' ' for key in string.punctuation})
    document = document.translate(table)

    # Convert string into list
    document = document.split(' ')

    # Remove numbers
    document = [w for w in document if not w.isdigit()]

    # Remove STOPWORDS
    document = [w for w in document if w not in STOPWORDS]

    return document


def get_documents():
    """Get documents from each source"""
    sources = _get_sources()
    documents = {}

    params = {
        'apiKey': NEWSAPIORG_KEY,
    }

    for source in sources:
        params['source'] = source
        resp = requests.get(ARTICLES_URL + urlencode(params))
        data = resp.json()

        document = ''
        for article in data['articles']:
            title = article['title'] or ''
            description = article['description'] or ''
            article = title + description
            document += article
            ' '.join([document, title, description])

        document = _clean_text(document)
        documents[source] = document
    return documents


def generate_histogram(documents):
    """Count occurrences of words in the document"""
    histograms = {}
    for source, data in documents.items():
        histograms[source] = defaultdict(int)
        for word in data:
            histograms[source][word] += 1

    for source in histograms.keys():
        documents[source] = histograms[source]

    return documents


def _generate_vector_structure(histograms):
    """Generate sorted vector of all unique words"""
    words = set()
    for source, histogram in histograms.items():
        for word in histogram.keys():
            words.add(word)

    words = sorted(list(words))
    return words


def generate_vectors(histograms):
    """Convert histograms to vectors"""
    words = _generate_vector_structure(histograms)

    for source, histogram in histograms.items():
        vector = [0 for x in range(len(words))]

        for word, count in histogram.items():
            index = words.index(word)
            vector[index] = count
        histograms[source] = vector

    return histograms


def centerpoint(v1, v2):
    """Computer the centerpoint of two vectors"""
    return [(x + y) / 2 for x, y in zip(v1, v2)]


def distance(v1, v2):
    """Measure the distance between two vectors"""
    radicand = 0
    for i in range(len(v1)):
        radicand += (v1[i] - v2[i]) ** 2

    return sqrt(radicand)


def average(vectors):
    """Compute the average of vectors"""
    return [sum(col) / len(vectors) for col in zip(*vectors)]


def initialize_centroids(k, vectors):
    """Create initial centroids"""
    centroids = []
    used_sources = []

    # Choose each starting centroid from random elements in our dataset
    sources = sorted(list(vectors.keys()))
    for x in range(k):
        index = randint(0, len(sources) - 1)
        source = sources.pop(index)
        used_sources.append(source)

        centroid = {'value': vectors[source]}
        centroids.append(centroid)

    print('initial centroids:', used_sources)
    return centroids


def cluster(centroids, vectors):
    """Compute mean for each vector and assign to a cluster"""
    # Clear centroid members
    for centroid in centroids:
        centroid['members'] = []

    for source, vector in vectors.items():
        i = 0
        min_index = 0
        current_min = 0

        for centroid in centroids:
            centroid_distance = distance(centroid['value'], vector)
            if i == 0:
                current_min = centroid_distance

            if centroid_distance < current_min:
                current_min = centroid_distance
                min_index = i

            i += 1
        centroids[min_index]['members'].append(source)

    return centroids


def recompute_centroids(centroids, vectors):
    """Find new centroids"""
    for centroid in centroids:
        centroid['value'] = average(
            [v for s, v in vectors.items() if s in centroid['members']]
        )

    return centroids


def print_centroids(centroids):
    """Print centroids to console"""
    print('\n')
    for i, centroid in enumerate(centroids):
        print('centroid {}:'.format(i), centroid['members'])


def kmeans(k):
    """Compute k-means clustering on news headlines and descriptions"""
    documents = get_documents()
    histograms = generate_histogram(documents)
    vectors = generate_vectors(histograms)
    centroids = initialize_centroids(k, vectors)
    centroids = cluster(centroids, vectors)

    iterations = 1

    while(True):
        old_clusters = [c['members'] for c in centroids]

        recompute_centroids(centroids, vectors)
        centroids = cluster(centroids, vectors)
        print_centroids(centroids)

        new_clusters = [c['members'] for c in centroids]
        if old_clusters == new_clusters:
            break

        iterations += 1

    print('Stopped after {} iterations'.format(iterations))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please enter a k value as an argument')
        exit(1)

    k = int(sys.argv[1])
    kmeans(k)
