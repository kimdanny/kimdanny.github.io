# Loader
from StringLoader import StringLoader
from langchain_community.document_loaders import UnstructuredURLLoader

# Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from RecursiveJsonSplitter import RecursiveJsonSplitter
from langchain.text_splitter import CharacterTextSplitter

# Emeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector DB
from langchain_community.vectorstores.faiss import FAISS

# Others
from langchain_core.documents import Document
import json

import os
from tqdm import tqdm
from typing import List
from load_utils import (
    get_document,
    get_user_agent,
    get_unique_pdfs,
    wget_file_to_dir,
    remove_downloaded_file,
)
from sys import exit


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Index:
    def __init__(
        self,
        embedding_fn: HuggingFaceEmbeddings,
        index_top_dir_path: str,
    ) -> None:
        self.embedding_fn = embedding_fn
        self.INDEX_TOP_DIR_PATH = index_top_dir_path

    @staticmethod
    def __create_minibatches(passages, batch_size=3):
        mini_batches = []
        for i in range(0, len(passages), batch_size):
            minibatch = passages[i : i + batch_size]
            mini_batches.append(minibatch)
        return mini_batches

    def __create_index(self, passages):
        """
        Index passages with embedding function and return the DB object

        Beware: Bulk ingestion needs huge RAM capacity, therefore,
        Index passage embeddings into the Vector DB by mini batch ingestion
        """
        index = FAISS.from_documents(passages[:3], self.embedding_fn)
        mini_batches = self.__create_minibatches(passages[3:])
        for mini_batch in tqdm(mini_batches):
            index.add_documents(
                mini_batch
            )  # returns doc_ids. For inspection, use inspect_index.py
        return index

    @staticmethod
    def __make_index_name_from_url(url) -> str:
        url = url.replace("https://", "")
        url = url.replace("/", "")
        url = url.replace(".", "")
        return url

    def from_urls(self, url_json_fp: str, key_only=False) -> List[str]:
        """
        Index from url_dict and return list of index paths

        :param: key_only: set True to only ingest key urls (can be found in base_urls.txt)
        """
        with open(url_json_fp, "r") as f:
            url_dict = json.load(f)
        f.close()

        index_paths = []
        for key_url, value_urls in url_dict.items():
            print(f"==={key_url}===")
            # Load data
            urls_to_load = [key_url] if key_only else [key_url] + value_urls
            loader = UnstructuredURLLoader(urls=urls_to_load, headers=get_user_agent())
            documents: List[Document] = loader.load()
            print("Progress: Documents are loaded")

            # Split the document into passages
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            passages = text_splitter.split_documents(documents)
            print("Progress: Documents are split into passages")

            db = self.__create_index(passages)
            print("Progress: Indexing Finished")

            # Save to disk
            index_name = self.__make_index_name_from_url(key_url)
            if key_only:
                index_name += "-keyOnly"
            save_path = os.path.join(self.INDEX_TOP_DIR_PATH, index_name)
            db.save_local(save_path)
            print("Progress: Index mounted on disk")
            index_paths.append(save_path)

        return index_paths

    def from_urls_table_extraction(self, url_json_fp: str, key_only=False) -> List[str]:
        with open(url_json_fp, "r") as f:
            url_dict = json.load(f)
        f.close()

        index_paths = []
        for key_url, value_urls in url_dict.items():
            print(f"==={key_url}===")
            urls_to_load = [key_url] if key_only else [key_url] + value_urls
            # Load data
            text_documents: List[Document] = []
            table_documents: List[Document] = []
            for url in urls_to_load:
                try:
                    text_string, table_string = get_document(
                        url=url, process_table=True
                    )
                except:
                    continue
                if text_string:
                    text_loader = StringLoader(text_string, source_url=url)
                    text_doc: List[Document] = text_loader.load()
                    text_documents.extend(text_doc)
                if table_string:
                    table_loader = StringLoader(table_string, source_url=url)
                    table_doc: List[Document] = table_loader.load()
                    table_documents.extend(table_doc)

            print("Progress: Text and Table Documents are loaded")

            # Split the document into passages
            # For text documents, use RecursiveCharacterTextSplitter,
            # For table documents, use CharacterTextSplitter and split by <SEP>
            if not len(text_documents) == 0:
                text_doc_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=150
                )
                text_passages = text_doc_splitter.split_documents(text_documents)
                print("Progress: Text documents are split into passages")
                # index text
                text_index = self.__create_index(text_passages)

            if not len(table_documents) == 0:
                table_doc_splitter = CharacterTextSplitter(
                    separator="<SEP>", chunk_overlap=0, chunk_size=330
                )
                table_passages = table_doc_splitter.split_documents(table_documents)
                print("Progress: Table documents are split into passages")
                # index table
                table_index = self.__create_index(table_passages)

            # Merge two indices
            if len(text_documents) != 0 and len(table_documents) == 0:
                merged_index = text_index
            elif len(text_documents) == 0 and len(table_documents) != 0:
                merged_index = table_index
            else:
                # both present
                text_index.merge_from(table_index)
                merged_index = text_index
            print("Progress: Indexing Finished")

            # Save to disk
            index_name = self.__make_index_name_from_url(key_url)
            if key_only:
                index_name += "-keyOnly"
            save_path = os.path.join(self.INDEX_TOP_DIR_PATH, index_name)

            merged_index.save_local(save_path)
            print("Progress: Index mounted on disk")
            index_paths.append(save_path)

        return index_paths

    def from_json(self, json_fp: str, save_index_name: str) -> List[str]:
        """
        Index json by RecursiveJsonSplitter and return index path
        RecursiveJsonSplitter keep the json format with all the previous keys so we do not lose context.
        """
        # Load json file
        # Documents are separated by Faculty by jq schema
        # loader = JSONLoader(file_path=json_fp, jq_schema=".[]", text_content=False)
        # documents = loader.load()
        with open(json_fp, "r") as f:
            json_dict = json.load(f)
        f.close()
        print("Progress: Documents are loaded")

        # Split json into passages
        # This will keep the json format with all the previous keys so we do not lose context!!
        json_splitter = RecursiveJsonSplitter(max_chunk_size=200)
        passages = json_splitter.create_documents(texts=[json_dict])
        print("Progress: Documents are split into passages")

        index = self.__create_index(passages)
        print("Progress: Indexing Finished")

        # Save to disk
        save_path = os.path.join(self.INDEX_TOP_DIR_PATH, save_index_name)
        index.save_local(save_path)
        print("Progress: Index mounted on disk")

        return list(save_path)

    def from_pdfs(self, pub_json_fp: str, save_index_name: str) -> List[str]:
        with open(pub_json_fp, "r") as f:
            json_dict = json.load(f)
        f.close()

        index_paths = []
        uniq_pdf_urls: list = get_unique_pdfs(json_dict)  # 477 pdfs
        url_batches = self.__create_minibatches(uniq_pdf_urls, batch_size=100)
        for i, url_batch in enumerate(url_batches):
            loader = UnstructuredURLLoader(urls=url_batch, headers=get_user_agent())
            documents: List[Document] = loader.load()
            print(f"Progress: Document batch {i} are loaded")

            # Split the document into passages
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=150
            )
            passages = text_splitter.split_documents(documents)
            print(f"Progress: Document batch {i} are split into passages")

            index = self.__create_index(passages)
            print(f"Progress: batch {i} Indexing Finished")

            save_path = os.path.join(self.INDEX_TOP_DIR_PATH, f"{save_index_name}-{i}")
            index.save_local(save_path)
            print(f"Progress: batch {i} Index mounted on disk")
            index_paths.append(save_path)

        return index_paths
