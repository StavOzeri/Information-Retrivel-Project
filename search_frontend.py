from flask import Flask, request, jsonify
import sys
import collections
import math
import pickle
import pandas as pd
import numpy as np
import os
from google.cloud import storage
import builtins
from collections import Counter, defaultdict
import re
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex, MultiFileWriter


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


BUCKET_NAME = 'final_project_ir_stav_hen_bucket'

def download_file(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} -> {destination_file_name}")
    except Exception as e:
        print(f"Failed to download {source_blob_name}: {e}")

def check_and_download():
    if not os.path.exists('postings_gcp'):
        os.makedirs('postings_gcp')
        
    files = [
        ('postings_gcp/index_body.pkl', 'index_body.pkl'), 
        ('index_title.pkl', 'index_title.pkl'), 
        ('index_norms.pkl', 'index_norms.pkl'),
        ('postings_gcp/id_to_title.pkl', 'id_to_title.pkl'),
        ('pageviews-202108-user.pkl', 'pageviews-202108-user.pkl')
    ]
    
    print("Checking required files")
    for src, dst in files:
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            print(f"Downloading {dst} form {src}...")
            download_file(BUCKET_NAME, src, dst)
        else:
             print(f"File {dst} already exists.")

    print("Download Check Complete")

check_and_download()

# LOAD INDICES

DATA_DIR = "."

def load_pickle(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: {filename} missing.")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

print("Loading indices into memory...")
idx_body = load_pickle('index_body.pkl')
idx_title = load_pickle('index_title.pkl')
idx_norms = load_pickle('index_norms.pkl')
id_to_title = load_pickle('id_to_title.pkl')
page_views = load_pickle('pageviews-202108-user.pkl')
print("Indices loaded successfully.")


# HELPER FUNCTIONS

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) 
            if token.group() not in all_stopwords]

def get_top_n_results(sim_dict, N=100):
    sorted_results = sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)[:N]
    final_res = []
    for doc_id, score in sorted_results:
        if id_to_title:
            title = id_to_title.get(doc_id, "Unknown Title")
        else:
            title = "Unknown Title"
        final_res.append((str(doc_id), title))
    return final_res


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    tokens = tokenize(query)
    combined_scores = Counter()
    
    # Weights
    w_title = 0.6    
    w_body = 0.4     
    w_views = 0.1    
    
    # Title Score
    if idx_title:
        for term in tokens:
            if term in idx_title:
                for doc_id, tf in idx_title[term]:
                    combined_scores[doc_id] += w_title

    # Body Score (Cosine Similarity)
    if idx_body and idx_norms:
        for term in set(tokens):
            try:
                posting_list = idx_body.read_a_posting_list(".", term, BUCKET_NAME)
                for doc_id, tf in posting_list:
                    if doc_id in idx_norms:
                        norm = idx_norms[doc_id]
                        if norm > 0:
                            combined_scores[doc_id] += (tf / norm) * w_body
            except:
                continue

    # PageViews Boost
    if page_views:
        for doc_id in combined_scores.keys():
            views = page_views.get(doc_id, 0)
            if views > 0:
                combined_scores[doc_id] += math.log(views, 10) * w_views

    res = get_top_n_results(combined_scores, 100)

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    tokens = tokenize(query)
    scores = Counter()

    if idx_body and idx_norms:
        for term in set(tokens):
            try:
                posting_list = idx_body.read_a_posting_list(".", term, BUCKET_NAME)
                for doc_id, tf in posting_list:
                    if doc_id in idx_norms:
                        norm = idx_norms[doc_id]
                        if norm > 0:
                            scores[doc_id] += (tf / norm)
            except:
                continue

    res = get_top_n_results(scores, 100)

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    tokens = tokenize(query)
    scores = Counter()

    if idx_title:
        for term in tokens:
            if term in idx_title:
                for doc_id, tf in idx_title[term]:
                    scores[doc_id] += 1
    
    res = get_top_n_results(scores, 100)

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # Return 0.0 for each requested ID since we don't have the data.
    res = [0.0] * len(wiki_ids)

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    for doc_id in wiki_ids:
        res.append(page_views.get(doc_id, 0))

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
