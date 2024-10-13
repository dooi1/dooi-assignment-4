from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words)
term_doc_matrix = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=100)
svd_matrix = svd.fit_transform(term_doc_matrix)

svd_components = svd.components_   # This holds the right singular vectors
explained_variance = svd.explained_variance_


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
    query_vec = vectorizer.transform([query])

    # Project query into the reduced LSA space (manually apply the SVD components to query_vec)
    query_lsa = query_vec.dot(svd_components.T)

    # Compute cosine similarity between the query and all documents in the reduced space
    cos_similarities = cosine_similarity(query_lsa, svd_matrix)[0]
    
    # Get the top 5 most similar documents
    top_indices = np.argsort(cos_similarities)[-5:][::-1]
    top_similarities = cos_similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]
    
    return top_documents, top_similarities.tolist(), top_indices.tolist()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
