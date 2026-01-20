"""
1) Build FAISS Index of 
    - pages of urls
    - publication json file
2) and store them on disk
"""

import argparse
import sys
from pathlib import Path
import os
from Index import Index
from langchain_community.vectorstores.faiss import FAISS
from index_utils import load_hf_embeddings


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
URL_JSON_FP = os.path.join(DIR_PATH, "dataset", "collected_links_depth_1.json")
FACULTY_JSON_FP = os.path.join(DIR_PATH, "dataset", "faculty.json")
PUB_JSON_FP = os.path.join(
    DIR_PATH, "dataset", "google_scholar_pubs.json"
)  # need pdf ingestion


if not Path(URL_JSON_FP).exists():
    raise FileNotFoundError("Run 01_gather_urls.py to create a json file")

if not Path(FACULTY_JSON_FP).exists() or not Path(PUB_JSON_FP).exists():
    raise FileNotFoundError("Run 02_gather_pubs.py to create a json files")


def main(args):
    device = "gpu" if args.use_gpu else "cpu"
    INDEX_TOP_DIR_PATH = os.path.join(
        DIR_PATH, f"faiss-{str(args.hf_model_path).split('/')[-1]}"
    )

    # Load passage embedding function
    embedding_fn = load_hf_embeddings(
        model_path=args.hf_model_path,
        instruct_model=args.instruct_embedding,
        device=device,
    )
    print(f"Progress: {args.hf_model_path} is loaded")

    # Instantiate Index Class with embedder
    index_creation = Index(
        embedding_fn=embedding_fn,
        index_top_dir_path=INDEX_TOP_DIR_PATH,
    )

    ## 1. Index URL
    if args.process_table:
        index_paths_1 = index_creation.from_urls_table_extraction(
            url_json_fp=URL_JSON_FP, key_only=False
        )
    else:
        index_paths_1 = index_creation.from_urls(
            url_json_fp=URL_JSON_FP, key_only=False
        )

    ## 2. Index faculty information (JSON)
    index_path_2 = index_creation.from_json(
        json_fp=FACULTY_JSON_FP, save_index_name="faculty"
    )

    ## 3. Index faculty publication info (PDFs)
    index_paths_3 = index_creation.from_json(
        json_fp=PUB_JSON_FP, save_index_name="faculty-pubs"
    )
    index_paths_4 = index_creation.from_pdfs(
        pub_json_fp=PUB_JSON_FP, save_index_name="papers"
    )

    ## 4. Index Fusion
    index_paths = index_paths_1 + index_path_2 + index_paths_3 + index_paths_4
    merged_index = FAISS.load_local(index_paths[0], embedding_fn)
    for index_path in index_paths[1:]:
        # laod index and fuse
        next_index = FAISS.load_local(index_path, embedding_fn)
        merged_index.merge_from(next_index)
    # save fused index to new path
    merged_index.save_local(os.path.join(INDEX_TOP_DIR_PATH, "merged"))


if __name__ == "__main__":
    # - different text splitting strategies.
    # - different emedding models
    parser = argparse.ArgumentParser(description="indexing strategies")

    # Add arguments
    # Refer to MTEB leaderboard for choosing embedding model: https://huggingface.co/spaces/mteb/leaderboard
    # sentence-transformers/all-MiniLM-l6-v2
    # hkunlp/instructor-xl
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="sentence-transformers/all-MiniLM-l6-v2",
        help="First argument",
    )
    parser.add_argument(
        "--instruct_embedding",
        action="store_true",
        help="use instruction-finetuned embedding model",
    )
    parser.add_argument(
        "--process_table",
        action="store_true",
        help="enable speical treatment for tabular data",
    )
    parser.add_argument("--use_gpu", action="store_true", help="compute in gpu")

    # Parse the arguments
    args = parser.parse_args()
    main(args)
