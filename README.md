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
The engine processes queries through an optimized multi-stage retrieval pipeline designed for high precision and efficiency:

* **Tokenization**: Queries are processed using a Regex tokenizer that filters out standard English stopwords alongside a customized list of corpus-specific "noise" words (e.g., "category", "references", "external") to focus on core content.
* **Scoring Components**: The final ranking is an aggregate of multiple relevance signals:
    * **Title Score** (Weight = 0.6): Matches query terms within article titles, weighted by term frequency (TF) to prioritize exact and descriptive title matches.
    * **Body Score** (Weight = 0.3): Implements a Vector Space Model approach. It calculates Cosine Similarity using a full TF-IDF implementation. Scores are normalized by pre-computed Euclidean norms (idx_norms) to ensure fairness across documents of varying lengths.
    * **PageViews Boost** (Weight = 0.1): Incorporates a logarithmic PageViews boost to act as a "tie-breaker," favoring high-quality, popular articles when textual relevance is comparable.
* **Pseudo-Proximity Boost**: To reward the accumulation of evidence across multiple query terms, the engine applies a bonus (10% per additional unique term) for documents that contain a higher variety of terms from the user's query.
* **Heuristics for Efficiency**: To maintain low latency (under the 35-second requirement), the engine employs a Champion List heuristic, processing only the top 1,000 posting entries for each query term.

### Implemented Functions

* **`search()`**: The primary retrieval function. It integrates Title matches, Body TF-IDF Cosine Similarity, and PageView boosts into a weighted final score, returning the top 100 most relevant results.

* **`search_body()`**: A focused retrieval method that ranks results based exclusively on the TF-IDF Cosine Similarity between the query and the article's body text.

* **`search_title()`**: A binary retrieval function that ranks documents based on the count of distinct query words appearing in their titles.

* **`get_pageview()`**: A utility function that accepts a list of document IDs and returns their respective PageView counts.
