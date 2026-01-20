from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class StringLoader(BaseLoader):
    """
    Load string.
    """

    def __init__(self, string_payload: str, source_url=None):
        self.string_payload = string_payload
        self.source_url = source_url

    def load(self) -> List[Document]:
        """Load from file path."""
        if self.source_url is not None:
            metadata = {"source": self.source_url}
            return [Document(page_content=self.string_payload, metadata=metadata)]
        return [Document(page_content=self.string_payload)]
