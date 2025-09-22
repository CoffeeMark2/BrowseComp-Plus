#
# BC Search Tool for CK framework
#

import os
import sys
from typing import List, Dict, Any
import traceback

# Add BC project to path

from ..agents.tool import Tool
from ..agents.utils import rprint

# Import BC project modules
try:
    from .bc.searcher.tools import SearchToolHandler
    from .bc.searcher.searchers import SearcherType
    from .bc.searcher.searchers.custom_searcher import CustomSearcher
    HAS_BC = True
except ImportError:
    HAS_BC = False
    traceback.print_exc()
    rprint("Warning: BC project not found. BCSearchTool will not be available.")


class BCSearchTool(Tool):
    def __init__(self, searcher=None, max_results=5, snippet_max_tokens=512, list_enum=True, **kwargs):
        """
        Initialize BC Search Tool

        Args:
            searcher: BC searcher instance (uses CustomSearcher by default)
            max_results: Maximum number of results to return
            snippet_max_tokens: Maximum tokens for snippet truncation
            list_enum: Whether to enumerate results
        """
        super().__init__(name="bc_web_search")
        self.max_results = max_results
        self.snippet_max_tokens = snippet_max_tokens
        self.list_enum = list_enum

        if not HAS_BC:
            traceback.print_exc()
            raise RuntimeError("BC project not found. Cannot initialize BCSearchTool.")

        # Initialize searcher
        if searcher is None:
            # Create default args object for CustomSearcher
            class Args:
                pass
            args = Args()
            searcher = CustomSearcher(args)

        # Initialize search tool handler
        self.search_tool_handler = SearchToolHandler(
            searcher=searcher,
            snippet_max_tokens=snippet_max_tokens,
            k=max_results,
            include_get_document=False
        )

    def get_function_definition(self, short: bool):
        if short:
            return """- def bc_web_search(query: str) -> str:  # Perform a search using BC's advanced search capabilities."""
        else:
            return """- bc_web_search
```python
def bc_web_search(query: str) -> str:
    \""" Perform a search using BC's advanced search capabilities.
    Args:
        query (str): A search query string.
    Returns:
        str: A string containing search results, including titles, URLs, and snippets.
    Notes:
        - Uses BC's advanced search infrastructure for better results.
        - Supports both local knowledge bases and web search capabilities.
    Examples:
        >>> answer = bc_web_search(query="latest developments in AI")
        >>> print(answer)
    \"""
```"""

    def __call__(self, query: str) -> str:
        """
        Execute search query using BC's search infrastructure

        Args:
            query: Search query string

        Returns:
            Formatted search results as string
        """
        try:
            # Use the search tool handler to perform search
            results = self.search_tool_handler._search(query)

            # Parse the JSON results
            import json
            results_list = json.loads(results)

            # Format results similar to SimpleSearchTool
            if len(results_list) == 0:
                ret = "Search Results: No results found! Try a different query."
            else:
                formatted_results = []
                for ii, result in enumerate(results_list):
                    # Extract fields - BC uses different field names than web search
                    title = result.get("docid", f"Result {ii+1}")
                    link = f"docid:{result.get('docid', 'N/A')}"
                    content = result.get("snippet", result.get("text", "No content available"))

                    if self.list_enum:
                        formatted_results.append(f"({ii}) title={repr(title)}, link={repr(link)}, content={repr(content)}")
                    else:
                        formatted_results.append(f"- title={repr(title)}, link={repr(link)}, content={repr(content)}")

                ret = "Search Results:\n" + "\n".join(formatted_results)

            return ret

        except Exception as e:
            rprint(f"Error in BCSearchTool: {e}")
            return f"Search Results: Error occurred during search: {str(e)}"


# For backward compatibility, also create an alias
BCWebSearchTool = BCSearchTool