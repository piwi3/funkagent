from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.schema import Document

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        Callbacks,
    )

from langchain.retrievers import AzureCognitiveSearchRetriever as AzureCognitiveSearchRetriever


class AzureCognitiveSearchRetrieverWithFilter(AzureCognitiveSearchRetriever):

    filter: Optional[str] = None
    """Filter for search results. Set to None to retrieve all results."""

    async def aget_relevant_documents_filter(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filter: Optional[str] = None, 
        **kwargs: Any,
    ) -> List[Document]:
        self.filter = filter
        return await self.aget_relevant_documents(
            query = query,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            **kwargs
        )
    
    def _build_search_url(self, query: str, **kwargs) -> str:
        base_url = f"https://{self.service_name}.search.windows.net/"
        endpoint_path = f"indexes/{self.index_name}/docs?api-version={self.api_version}"
        top_param = f"&$top={self.top_k}" if self.top_k else ""
        filter = "&$filter=" + self.filter if self.filter else "" 
        return base_url + endpoint_path + f"&search={query}" + top_param + filter