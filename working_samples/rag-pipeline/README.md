# ANLP Assignment 2

## Environment
Please use virtual environment and install dependencies from `requirements.txt`.
We are using python 3.11.
Please format your code with `black` before pusing to remote.

## How to run the code
We ordered the main scripts by the stages of implementing the RAG system.

### Offline raw data collection and indexing
Inside `/offline`,
- `01_gather_urls.py` recursively srapes urls from provided key knowledge resources.
- `02_gather_pubs.py` and `02_google_scholar_pubs.py` collects faculty publications and their metadata using the Semantic Scholar API and Google Scholar, respectively.
-  `03_index.py` builds FAISS index of pages of collected urls and publication json files, and stores them on disk.

### Online computation
- `04_prompt_lm.py` prompts language models for question-answering, either with retrieval using the KNN retriever from the FAISS liberary or without retrieval.

### Post evaluation and analysis
- `05_eval.py` evaluates the model's predictions against the gold answers, utilizing functions from `eval_utils.py`.
- `06_analysis.py` analyzes outputs of the question-answering system across multiple system configurations (prompt, top-k documents, context format) and question categories (yes/no, date, numeric, etc.), compiling detailed performance metrics.
- `07_plots.ipynb`: visualizes the metrics in line plots.
