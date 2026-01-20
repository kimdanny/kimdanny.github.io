# https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from typing import Union


def load_hf_embeddings(
    model_path: str, instruct_model=False, device: str = "cpu"
) -> Union[HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings]:
    model_kwargs = {"device": device}

    # Initialize an Embedding instance with the specified parameters
    if instruct_model:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_path,  # Provide the pre-trained model's path
            model_kwargs=model_kwargs,  # Pass the model configuration options
            encode_kwargs={"normalize_embeddings": True},  # Pass the encoding options
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,  # Provide the pre-trained model's path
            model_kwargs=model_kwargs,  # Pass the model configuration options
            encode_kwargs={"normalize_embeddings": False},  # Pass the encoding options
        )
    return embeddings
