"""
Episodic Memory module for WorldMM.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

from hipporag import HippoRAG

logger = logging.getLogger(__name__)


@dataclass
class CaptionEntry:
    """Represents a single caption entry with its metadata."""
    id: str
    text: str
    start_time: str
    end_time: str
    date: str
    granularity: str
    video_path: Optional[str] = None
    
    @property
    def timestamp_int(self) -> Tuple[int, int]:
        """Convert start and end times to integer format (day + time.zfill(8))."""
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts
    
    def to_display_str(self) -> str:
        """Format caption for display with time range."""
        start_ts, end_ts = self.timestamp_int
        return f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"


def _transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


def _load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


class EpisodicMemory:
    """
    Episodic Memory module that implements multiscale retrieval and filtering.
    
    This class manages episodic captions at multiple temporal granularities
    (30sec, 3min, 10min, 1h) and provides retrieval functionality using
    HippoRAG for indexing/retrieval with LLM-based multiscale filtering.
    
    The retrieval process:
    1. Index captions up to a given timestamp using HippoRAG
    2. Retrieve top-k candidates from each granularity level using HippoRAG
    3. Use LLM with multiscale_filter template to filter and rank the most relevant captions
    4. Return the filtered captions in ranked order
    
    Attributes:
        granularities: List of granularity levels to use
        captions: Dictionary mapping granularity -> list of CaptionEntry
        hipporag: Dictionary mapping granularity -> HippoRAG instance
        llm_model: Language model for filtering
        prompt_template_manager: Manager for prompt templates
    """
    
    GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        prompt_template_manager: PromptTemplateManager,
        granularities: Optional[List[str]] = None,
    ):
        """
        Initialize EpisodicMemory.
        
        Args:
            embedding_model: Embedding model for HippoRAG
            llm_model: LLM model for filtering
            prompt_template_manager: Prompt template manager
            granularities: List of granularity levels to use (default: all)
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.prompt_template_manager = prompt_template_manager
        self.granularities = granularities or self.GRANULARITY_ORDER
        
        # Storage for captions
        self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
        self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
        
        # Mapping from caption text to CaptionEntry for reverse lookup after HippoRAG retrieval
        self.text_to_entry: Dict[str, CaptionEntry] = {}
        
        # HippoRAG instance for each granularity
        self.hipporag: Dict[str, HippoRAG] = {}
        
        # Track indexed entries (entries that have been indexed up to indexed_time)
        self.indexed_entries: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
        self.indexed_time: int = 0  # 0 means nothing indexed yet
    
    def _get_or_create_hipporag(self, granularity: str) -> HippoRAG:
        """Get or create HippoRAG instance for a granularity level."""
        if granularity not in self.hipporag:
            self.hipporag[granularity] = HippoRAG(
                save_dir=f".cache/episodic_memory/{granularity}",
                llm_model=self.llm_model,
                embedding_model=self.embedding_model)
        return self.hipporag[granularity]
    
    def load_captions_from_files(
        self,
        caption_files: Dict[str, str],
    ) -> None:
        """
        Load captions from JSON files for each granularity level.
        
        Args:
            caption_files: Dict mapping granularity -> JSON file path
        """
        for granularity, file_path in caption_files.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
                continue
            
            try:
                data = _load_json(file_path)
                self._process_caption_data(data, granularity)
                logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
            except Exception as e:
                logger.error(f"Failed to load captions from {file_path}: {e}")
    
    def load_captions_from_data(
        self,
        caption_data: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """
        Load captions from in-memory data for each granularity level.
        
        Args:
            caption_data: Dict mapping granularity -> list of caption dicts
        """
        for granularity, data in caption_data.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not in configured granularities")
                continue
            
            self._process_caption_data(data, granularity)
            logger.info(f"Loaded {len(self.captions[granularity])} captions for granularity {granularity}")
    
    def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
        """Process raw caption data and create CaptionEntry objects."""
        for idx, entry in enumerate(data):
            caption_id = f"{granularity}_{idx}"
            caption_entry = CaptionEntry(
                id=caption_id,
                text=entry.get("text", ""),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=str(entry.get("date", "")),
                granularity=granularity,
                video_path=entry.get("video_path"),
            )
            self.captions[granularity].append(caption_entry)
            self.caption_id_to_entry[caption_id] = caption_entry
            self.text_to_entry[caption_entry.text] = caption_entry
    
    def index(self, until_time: int) -> None:
        """
        Index captions up to the specified timestamp using HippoRAG.
        
        This method provides all captions with end_time <= until_time to HippoRAG
        for indexing. Embeddings are computed inside HippoRAG.
        
        If captions have already been indexed up to a later time, this is a no-op.
        If called with a later time than previously indexed, it will re-index
        with the expanded set of captions.
        
        Args:
            until_time: Timestamp in integer format (day + time.zfill(8)) - index all 
                       captions with end_time <= this value
        """
        # If already indexed beyond this time, no need to recompute
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed up to {self.indexed_time}, skipping index for {until_time}")
            return
        
        for granularity in self.granularities:
            if not self.captions[granularity]:
                logger.warning(f"No captions loaded for granularity {granularity}")
                continue
            
            # Get entries that should be indexed (end_time <= until_time)
            entries_to_index = [
                entry for entry in self.captions[granularity]
                if entry.timestamp_int[1] <= until_time
            ]
            
            if not entries_to_index:
                logger.debug(f"No entries to index for granularity {granularity} up to {until_time}")
                continue
            
            # Get caption texts for HippoRAG
            caption_texts = [entry.text for entry in entries_to_index]
            
            # Get or create HippoRAG instance and update index
            hipporag = self._get_or_create_hipporag(granularity)
            hipporag.update(docs=caption_texts)
            
            # Update indexed entries
            self.indexed_entries[granularity] = entries_to_index
            
            logger.info(f"Indexed {len(entries_to_index)} captions for granularity {granularity}")
        
        self.indexed_time = until_time

    def retrieve_captions_as_str(self, entries: List[CaptionEntry]) -> str:
        """
        Format a list of caption entries as context string.
        
        Args:
            entries: List of CaptionEntry objects
            
        Returns:
            Formatted context string
        """
        return "\n\n".join(entry.to_display_str() for entry in entries)
    
    def retrieve(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = {
            "30sec": 10,
            "3min": 5,
            "10min": 5,
            "1h": 3
        },
        final_top_k: int = 3,
        as_context: bool = True
    ) -> Union[List[CaptionEntry], str]:
        """
        Retrieve relevant captions using HippoRAG and multiscale filtering.
        
        This method retrieves from the indexed captions using HippoRAG.
        Make sure to call index(until_time) before calling retrieve().
        
        The retrieval process:
        1. Retrieves top-k candidates from each granularity level using HippoRAG
        2. Uses LLM with multiscale_filter template to filter and rank results
        
        Args:
            query: The search query
            top_k_per_granularity: Number of candidates to retrieve per granularity level.
                Can be an int (same for all granularities) or a dict mapping granularity -> top_k.
            final_top_k: Final number of results to return after filtering
            as_context: Whether to return results as context strings instead of CaptionEntry objects
            
        Returns:
            List of CaptionEntry objects in ranked order
        """
        if self.indexed_time == 0:
            logger.warning("No captions indexed. Call index(until_time) before retrieve().")
            return []
        
        # Retrieve from each granularity level using HippoRAG
        all_candidates: List[Tuple[CaptionEntry, float]] = []
        
        for granularity in self.granularities:
            if granularity not in self.hipporag:
                continue
            
            # Get top_k for this granularity
            if isinstance(top_k_per_granularity, dict):
                granularity_top_k = top_k_per_granularity.get(granularity, 5)  # default to 5 if not specified
            else:
                granularity_top_k = top_k_per_granularity
            
            hipporag = self.hipporag[granularity]
            
            retrieval_result = hipporag.retrieve(
                queries=[query], 
                num_to_retrieve=granularity_top_k
            )
            
            if not retrieval_result or not retrieval_result[0].docs:
                continue
            
            # Convert retrieved docs back to CaptionEntry
            retrieved_docs = retrieval_result[0].docs
            retrieved_scores = retrieval_result[0].doc_scores if hasattr(retrieval_result[0], 'doc_scores') else [1.0] * len(retrieved_docs)
            
            count = 0
            for doc_text, score in zip(retrieved_docs, retrieved_scores):
                if count >= granularity_top_k:
                    break
                
                # Look up the CaptionEntry from text
                entry = self.text_to_entry.get(doc_text)
                if entry is None:
                    logger.warning(f"Could not find CaptionEntry for retrieved text: {doc_text[:50]}...")
                    continue
                
                all_candidates.append((entry, score))
                count += 1
        
        if not all_candidates:
            logger.warning("No candidates retrieved from any granularity level")
            return [] if not as_context else ""
        
        # Use LLM to filter and rank candidates
        filtered_entries = self._filter_with_llm(
            query=query,
            candidates=all_candidates,
            final_top_k=final_top_k,
        )
        
        if as_context:
            return self.retrieve_captions_as_str(filtered_entries)
        
        return filtered_entries
    
    def _filter_with_llm(
        self,
        query: str,
        candidates: List[Tuple[CaptionEntry, float]],
        final_top_k: int,
    ) -> List[CaptionEntry]:
        """
        Use LLM to filter and rank candidates using multiscale_filter template.
        
        Args:
            query: Original search query
            candidates: List of (CaptionEntry, score) tuples from all granularities
            final_top_k: Number of results to return
            
        Returns:
            Filtered and ranked list of CaptionEntry objects
        """
        if len(candidates) <= final_top_k:
            # No need to filter if we have fewer candidates than requested
            return [entry for entry, _ in candidates]
        
        # Format candidates for the LLM
        caption_list = []
        id_to_entry = {}
        for entry, score in candidates:
            start_ts, end_ts = entry.timestamp_int
            caption_info = {
                "id": entry.id,
                "granularity": entry.granularity,
                "start_time": _transform_timestamp(str(start_ts)),
                "end_time": _transform_timestamp(str(end_ts)),
                "text": entry.text,
            }
            caption_list.append(caption_info)
            id_to_entry[entry.id] = entry
        
        # Build prompt using template
        try:
            prompt = self.prompt_template_manager.render("multiscale_filter")
        except Exception as e:
            logger.error(f"Failed to render multiscale_filter template: {e}")
            # Fallback: return top candidates by score
            return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]
        
        # Add the query and candidates to the prompt
        filter_message = {
            "role": "user",
            "content": f"""Question: {query}

Retrieved Captions:
{json.dumps(caption_list, indent=2)}

Select the top {final_top_k} most relevant caption IDs to answer the question.
Return ONLY a JSON array of caption IDs in order of relevance (most relevant first)."""
        }
        prompt.append(filter_message)
        
        try:
            response = self.llm_model.generate(prompt)
            
            # Parse response to get selected IDs
            selected_ids = self._parse_filter_response(response, set(id_to_entry.keys()))
            
            # Return entries in the order specified by LLM
            result = []
            for cap_id in selected_ids[:final_top_k]:
                if cap_id in id_to_entry:
                    result.append(id_to_entry[cap_id])
            
            # If LLM returned fewer than requested, fill with top-scoring candidates
            if len(result) < final_top_k:
                existing_ids = {e.id for e in result}
                for entry, _ in sorted(candidates, key=lambda x: -x[1]):
                    if entry.id not in existing_ids:
                        result.append(entry)
                        if len(result) >= final_top_k:
                            break
            
            return result
            
        except Exception as e:
            logger.error(f"LLM filtering failed: {e}")
            # Fallback: return top candidates by score
            return [entry for entry, _ in sorted(candidates, key=lambda x: -x[1])[:final_top_k]]
    
    def _parse_filter_response(self, response: str, valid_ids: set) -> List[str]:
        """
        Parse LLM response to extract selected caption IDs.
        
        Args:
            response: LLM response string
            valid_ids: Set of valid caption IDs
            
        Returns:
            List of caption IDs
        """
        import re
        
        # Try to extract JSON array from response
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                ids = json.loads(match.group())
                if isinstance(ids, list):
                    return [str(id) for id in ids if str(id) in valid_ids]
        except json.JSONDecodeError:
            pass
        
        # Fallback: look for ID patterns in response
        found_ids = []
        for valid_id in valid_ids:
            if valid_id in response:
                found_ids.append(valid_id)
        
        return found_ids
    
    def reset_index(self) -> None:
        """Reset the indexed state, clearing HippoRAG instances and indexed entries."""
        self.hipporag.clear()
        for g in self.granularities:
            self.indexed_entries[g] = []
        self.indexed_time = 0
        logger.info("Index reset - all HippoRAG instances and indexed entries cleared")
    
    def get_indexed_time(self) -> str:
        """Get the current indexed time boundary."""
        return _transform_timestamp(str(self.indexed_time))
    
    def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
        """Get a caption entry by its ID."""
        return self.caption_id_to_entry.get(caption_id)
