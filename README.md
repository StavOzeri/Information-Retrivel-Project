# Information-Retrivel-Project

# Wikipedia Search Engine

A scalable, Flask-based Information Retrieval search engine deployed on Google Cloud Platform (GCP). This project implements a retrieval system for English Wikipedia that ranks documents based on Title matches, Body text cosine similarity, and PageViews importance.

## Project Structure

* **search_frontend.py**: The main application entry point.
    * Initializes the Flask web server.
    * **Auto-Ingestion**: Automatically downloads missing indices and data from the GCP Bucket (final_project_ir_stav_hen_bucket) upon startup.
    * **Memory Management**: Loads inverted indices (.pkl files) into RAM for fast retrieval.
    * **Search Logic**: Implements the ranking algorithm and query processing pipeline.

* **inverted_index_gcp.py**: A helper module used to read posting lists efficiently from binary files stored on the local disk.

* **postings_gcp/**: Local directory hosting the binary index files (shards) for the document body and title, optimized for low-latency random access.

## Core Functionality

### 1. Initialization & Data Loading
Upon execution, the system performs a self-check (check_and_download function):
* Verifies if the postings_gcp folder and essential pickle files (index_body.pkl, index_title.pkl, index_norms.pkl, id_to_title.pkl, pageviews.pkl) exist locally.
* If missing, it utilizes google.cloud.storage to fetch them securely from the defined project bucket.

**Indices:**
* **idx_body**: Inverted index for article text.
* **idx_title**: Inverted index for article titles.
* **idx_norms**: Pre-computed norms for Cosine Similarity calculations.
* **page_views**: Dictionary of page view counts.

### 2. The Search Algorithm
The engine processes queries through a multi-stage pipeline:

* **Tokenization**: The query is tokenized using a Regex tokenizer, filtering out English stopwords and corpus-specific stopwords.
* **Scoring Components**:
    * **Title Score** ($W_{title} = 0.6$): Keyword matching in article titles.
    * **Body Score** ($W_{body} = 0.4$): Cosine Similarity calculated using TF-IDF statistics from the body index.
    * **PageViews Boost** ($W_{views} = 0.1$): A logarithmic boost is added to favor popular articles.
* **Ranking**: Results are aggregated, sorted by descending score, and the top 100 are returned.

### Implemented Functions

* **`search()`**: The main retrieval function. It accepts a query, tokenizes it, and calculates a weighted score based on Title matches (0.6), Body Cosine Similarity (0.4), and PageView boosts (0.1) to return the top 100 results.

* **`search_body()`**: Performs a search based exclusively on the TF-IDF Cosine Similarity of the query against the article body text.

* **`search_title()`**: Performs a search based solely on binary keyword matches found within the article titles.

* **`get_pageview()`**: A helper function that receives a list of document IDs and returns their respective page view counts.
