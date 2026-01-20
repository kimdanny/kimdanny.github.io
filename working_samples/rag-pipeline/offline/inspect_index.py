from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os
import pandas as pd


def load_hf_embeddings(model_path: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs,  # Pass the encoding options
    )
    return embeddings


def vdb_to_df(vdb):
    """
    Show Vector DB in pandas DataFrame form for better inspection
    Warning: Use this for testing purpose when index is not huge.
    """
    v_dict = vdb.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata["source"]
        content = v_dict[k].page_content
        data_rows.append({"chunk_id": k, "document": doc_name, "content": content})
    vector_df = pd.DataFrame(data_rows)
    return vector_df


if __name__ == "__main__":
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    embedding_fn = load_hf_embeddings(
        model_path="sentence-transformers/all-MiniLM-l6-v2"
    )
    db = FAISS.load_local(
        os.path.join(DIR_PATH, "faiss-all-MiniLM-l6-v2/faculty_publications"),
        embedding_fn,
    )
    # https:/athletics.cmu.edu/athletics/tartanfacts
    # https:/enr-apps.as.cmu.edu/open/SOC/SOCServlet/completeSchedule
    # https:/www.cs.cmu.edu/scs25/25things/
    # faiss-all-MiniLM-l6-v2/faculty_publications

    v_dict = db.docstore._dict
    # print(len(v_dict))
    # print(v_dict)

    doc_id = db.index_to_docstore_id[54]
    document = db.docstore.search(doc_id)
    print(document)

    # docs = db.similarity_search("Why should humans adapt to fit computers? ", k=2)
    # print(docs)
