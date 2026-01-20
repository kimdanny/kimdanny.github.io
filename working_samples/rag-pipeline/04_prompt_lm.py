"""
Prompt for question answering using Hugging Face models including 
1) retrieval-based prompting 
2) no-retrieval (baseline)
"""

import os
import argparse
import torch
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS


class RetrievalSystem:
    """
    A class for document retrieval using a FAISS index, supporting similarity searches for given queries.

    Attributes:
        index_path (str): The filesystem path to the FAISS index.
        embedding_model (str): The identifier for the embedding model used to encode documents.
        db (FAISS): The loaded FAISS database instance.
    """

    def __init__(
        self,
        index_path: str,
        embedding_model: str,
        k: int = 2,
        document_format: str = "simple",
    ):
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.db = self._load_db()
        self.k = k
        self.document_format = document_format

    def _load_hf_embeddings(
        self, model_path: str, device: str = "cpu"
    ) -> HuggingFaceEmbeddings:
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print(f"Progress: {model_path} is loaded")
        return embeddings

    def _load_db(self):
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        embedding_fn = self._load_hf_embeddings(model_path=self.embedding_model)
        db_path = os.path.join(DIR_PATH, self.index_path)
        db = FAISS.load_local(db_path, embedding_fn)
        print(f"Progress: {self.index_path} is loaded")
        return db

    def retrieve_documents(self, query: str) -> List[str]:
        docs = self.db.similarity_search(query, k=self.k)
        print(f"Progress: Retrieved {len(docs)} documents")
        return self._format_docs(docs)

    @staticmethod
    def _format_docs(docs, document_format: str = "simple") -> str:
        if document_format == "simple":
            return "Context:\n\n" + "\n\n".join(doc.page_content for doc in docs)
        else:
            context = ""
            for doc in docs:
                context += (
                    "Context "
                    + str(docs.index(doc) + 1)
                    + ":\n"
                    + doc.page_content
                    + "\n\n"
                )

            return context


class PromptLM:
    """
    A class to encapsulate the question answering functionality using a specified language model, with optional retrieval support.

    Attributes:
        model_id (str): The identifier for the Hugging Face model to use.
        use_retrieval (bool): Flag indicating whether retrieval-based prompting is used.
        pipeline_kwargs (Dict): Additional arguments to configure the Hugging Face pipeline.
        hf_pipeline (HuggingFacePipeline): The initialized Hugging Face pipeline for text generation.
        prompt (PromptTemplate): The template used to format prompts for the model.
        chain (PromptTemplate | HuggingFacePipeline): The chain of operations to be executed for question answering.
    """

    def __init__(
        self, model_id: str, use_retrieval: bool = False, pipeline_kwargs: dict = None
    ):
        self.model_id = model_id
        self.use_retrieval = use_retrieval
        self.pipeline_kwargs = pipeline_kwargs or {"max_new_tokens": 50}
        self.hf_pipeline = self._initialize_pipeline()
        self.prompt = self._choose_prompt_template()

        self.chain = self.prompt | self.hf_pipeline

    def _initialize_pipeline(self) -> HuggingFacePipeline:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        return HuggingFacePipeline.from_model_id(
            model_id=self.model_id,
            task="text-generation",
            device=0,
            pipeline_kwargs=self.pipeline_kwargs,
        )

    def _choose_prompt_template(self) -> PromptTemplate:
        if self.use_retrieval:
            return PromptTemplate.from_template(
                """
                <s>[INST] <<SYS>>
                Provide a short phrase ONLY to answer the following question using the provided context. DO NOT provide a complete sentence.
                <</SYS>>
                Question:\n {question}\n\n
                {retrieved}
                Answer:
                [/INST]"""
            )
        else:
            return PromptTemplate.from_template(
                """
                <s>[INST] <<SYS>>
                Provide a short phrase ONLY to answer the following question. DO NOT provide a complete sentence.
                <</SYS>>
                Question:\n {question}\n\n
                Answer:
                [/INST]"""
            )

    def answer_question(self, question: str, retrieved: str = None) -> Dict[str, str]:
        response = self.chain.invoke(
            {"question": question, "retrieved": retrieved}
        ).strip()
        return {"question": question, "answer": response}


def read_files(file_path: str) -> List[str]:
    """
    Read a list of questions/answers from a txt file.
    """
    with open(file_path, "r") as file:
        data = file.readlines()
    return data


def main(args):
    model_id = args.model_id
    use_retrieval = args.use_retrieval
    k = args.k
    document_format = args.document_format

    retrieval_system, retrieved = None, None
    system_output = []

    questions = read_files("offline/dataset/test/questions.txt")

    if use_retrieval:
        retrieval_system = RetrievalSystem(
            index_path="offline/faiss-instructor-xl/lticscmuedupeoplefacultyindexhtml",
            embedding_model="hkunlp/instructor-xl",
            k=k,
            document_format=document_format,
        )
        retrieved = retrieval_system.retrieve_documents(question)
        # print(f"Retrieved: {retrieved}\n")

    qa_model = PromptLM(model_id=model_id, use_retrieval=use_retrieval)

    for q in questions:
        if use_retrieval:
            retrieved = retrieval_system.retrieve_documents(q)
        answer = qa_model.answer_question(q, retrieved)
        system_output.append(answer)

    with open(
        f"offline/dataset/test/system_out_{str(use_retrieval)}_{str(k)}_{document_format}_.txt",
        "w",
    ) as file:
        for item in system_output:
            file.write(item["answer"] + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prompt-based question answering")

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model id",
    )

    parser.add_argument(
        "--use_retrieval",
        default=False,
        help="Enable retrieval-based prompting",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of documents to retrieve",
    )

    parser.add_argument(
        "--document_format",
        type=str,
        default="simple",
    )

    args = parser.parse_args()

    main(args)
