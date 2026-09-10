"""
Kaggle Dataset Search & Discovery Tool
Queries Kaggle API to discover supporting datasets for AI implementation use cases.
"""

import logging
from typing import List, Dict, Any
from ai.state import DatasetReference
from ai.settings import get_settings

logger = logging.getLogger(__name__)

class KaggleDatasetTool:
    """
    Tool for searching datasets on Kaggle based on extracted keywords.
    """

    def __init__(self):
        self.api = None
        self._init_api()

    def _init_api(self):
        """Authenticate Kaggle API if credentials exist."""
        try:
            settings = get_settings()
            if settings.kaggle_configured:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self.api = KaggleApi()
                self.api.authenticate()
                logger.info("Kaggle API successfully authenticated.")
        except Exception as e:
            logger.warning(f"Kaggle API authentication unavailable ({e}). Dataset search will use open fallback mocks.")
            self.api = None

    def search_datasets(self, keywords: List[str], max_per_keyword: int = 2) -> List[DatasetReference]:
        """
        Search datasets for keywords and return structured DatasetReference objects.
        """
        results: List[DatasetReference] = []

        if not keywords:
            return results

        if self.api:
            for kw in keywords[:3]:
                try:
                    datasets = self.api.dataset_list(search=kw, sort_by="hottest")
                    for ds in datasets[:max_per_keyword]:
                        results.append(DatasetReference(
                            ref=ds.ref,
                            title=ds.title,
                            size=str(getattr(ds, 'size', 'N/A')),
                            votes=int(getattr(ds, 'voteCount', 0)),
                            keyword_matched=kw
                        ))
                except Exception as e:
                    logger.error(f"Kaggle search error for keyword '{kw}': {e}")
        else:
            # Clearly-labelled suggestions only. They are not verified Kaggle results.
            for kw in keywords[:3]:
                results.append(DatasetReference(
                    ref="unverified://dataset-suggestion",
                    title=f"Unverified dataset suggestion for {kw.title()}",
                    size="45 MB",
                    votes=128,
                    keyword_matched=kw
                ))

        return results
