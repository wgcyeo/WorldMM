"""
Multiscale caption generation using LLM summarization.

Adapted from the summary generation logic in EgoRAG (https://github.com/EvolvingLMMs-Lab/EgoLife), with a modified prompt and optimizations.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from ...llm import LLMModel

# ---------------------------------------------------------------------------
# System prompts -- keyed by perspective ("egocentric" or "general")
# Each has three levels with increasing word-count constraints.
# ---------------------------------------------------------------------------

_PROMPTS: Dict[str, List[str]] = {}

_PROMPTS["egocentric"] = [
    # L1
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

2. **Retain Key Information**
   - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
   - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
   - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
   - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
""",
    # L2 (300 words)
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

2. **Retain Key Information**
   - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
   - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
   - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
   - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Strictly output below 300 words in total. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
""",
    # L3 (1500 words)
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

2. **Retain Key Information**
   - The primary character's (**"I"**) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.
   - Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.
   - The **purpose and method** of key actions must be recorded, such as "ordering takeout using a phone" or "documenting a plan on a whiteboard."

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into **task arrangements, decision points, and specific execution details**.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.
   - Maintain the correct **chronological order** of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Strictly output below 1500 words in total. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of my key actions and other relevant characters' critical actions + Conclusion/recap to ensure logical clarity.
""",
]

_PROMPTS["general"] = [
    # L1
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points.

2. **Retain Key Information**
   - All subjects' decisions and actions must be fully presented, including all critical activities. Transitions between different parts should be seamless.
   - Any discussions, decisions, and task execution involving subjects that impact the main storyline must be reflected, but in a concise manner.
   - The **purpose and method** of key actions must be recorded.

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information.
   - Maintain the correct **chronological order** of events.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of key actions and relevant details + Conclusion/recap to ensure logical clarity.
""",
    # L2 (300 words)
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points.

2. **Retain Key Information**
   - All subjects' decisions and actions must be fully presented, including all critical activities. Transitions between different parts should be seamless.
   - Any discussions, decisions, and task execution involving subjects that impact the main storyline must be reflected, but in a concise manner.
   - The **purpose and method** of key actions must be recorded.

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information.
   - Maintain the correct **chronological order** of events.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Strictly output below 300 words in total. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of key actions and relevant details + Conclusion/recap to ensure logical clarity.
""",
    # L3 (1500 words)
    """As an **Event Summary Documentation Specialist**, your role is to systematically structure and summarize event information from a video, ensuring that all key actions are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

#### **Specific Requirements**

1. **Structure the Events Clearly**
   - **Merge related events:** Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.
   - **Logical segmentation:** Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points.

2. **Retain Key Information**
   - All subjects' decisions and actions must be fully presented, including all critical activities. Transitions between different parts should be seamless.
   - Any discussions, decisions, and task execution involving subjects that impact the main storyline must be reflected, but in a concise manner.
   - The **purpose and method** of key actions must be recorded.

3. **Concise Expression, Remove Redundancies**
   - Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.
   - Remove trivial conversations and extract only the core topics and conclusions of discussions.

4. **Strictly Adhere to Facts, No Assumptions**
   - Do not make assumptions or add interpretations—strictly organize content based on available information.
   - Maintain the correct **chronological order** of events.

#### **Output Format**
Each paragraph should represent **one major event**, structured in a **summary-detail-summary** format. Strictly output below 1500 words in total. Do not report the word count in the output.
Example output of one event:
**[write down the summary of the event, no index]** One sentence summarizing the overall situation + Description of key actions and relevant details + Conclusion/recap to ensure logical clarity.
""",
]


def _time_to_seconds(time_val) -> int:
    time_val = int(time_val)
    hours = time_val // 1000000
    minutes = (time_val % 1000000) // 10000
    seconds = (time_val % 10000) // 100
    return hours * 3600 + minutes * 60 + seconds


def _seconds_to_time(seconds: int) -> int:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return hours * 1000000 + minutes * 10000 + secs * 100


def _summarize_batch(
    batch: list,
    window_start: int,
    window_end: int,
    system_message: str,
    date: str,
    llm: LLMModel,
    batch_index: int,
) -> Optional[dict]:
    context = "\n".join(item.get("text", "") for item in batch)
    try:
        response = llm.generate([
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"All descriptions: {context}"},
        ])
    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")
        return None

    if response:
        return {
            "text": response,
            "date": date,
            "start_time": str(_seconds_to_time(window_start)).zfill(8),
            "end_time": str(_seconds_to_time(window_end)).zfill(8),
            "video_path": batch[0].get("video_path", ""),
        }
    return None


def _summarize_level(
    input_data: list,
    window_seconds: int,
    system_message: str,
    llm: LLMModel,
    date: str = "DAY1",
) -> list:
    window_batches = []
    items_by_date: Dict[str, list] = {}
    for item in input_data:
        item_date = str(item.get("date") or date)
        items_by_date.setdefault(item_date, []).append(item)

    for item_date, items in items_by_date.items():
        batch: list = []
        current_window_start = None

        for item in sorted(items, key=lambda entry: _time_to_seconds(entry["start_time"])):
            start_sec = _time_to_seconds(item["start_time"])

            if current_window_start is None:
                current_window_start = start_sec

            if start_sec < current_window_start + window_seconds:
                batch.append(item)
            else:
                if batch:
                    window_batches.append((batch, current_window_start, current_window_start + window_seconds, item_date))

                current_window_start = (start_sec // window_seconds) * window_seconds
                batch = [item]

        if batch and current_window_start is not None:
            window_batches.append((batch, current_window_start, current_window_start + window_seconds, item_date))

    if not window_batches:
        return []
    
    results_by_index: dict[int, dict] = {}

    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(
                _summarize_batch,
                batch,
                window_start,
                window_end,
                system_message,
                item_date,
                llm,
                window_index,
            ): window_index
            for window_index, (batch, window_start, window_end, item_date) in enumerate(
                window_batches
            )
        }
        progress_bar = tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc=f"Summarizing {window_seconds}s windows",
            unit="window",
            leave=False,
        )
        for future in progress_bar:
            window_index = future_to_index[future]
            result = future.result()
            if result:
                results_by_index[window_index] = result
        progress_bar.close()

    if len(results_by_index) != len(window_batches):
        raise RuntimeError(
            f"Failed to summarize {len(window_batches) - len(results_by_index)} "
            f"of {len(window_batches)} windows"
        )

    return [results_by_index[idx] for idx in sorted(results_by_index)]


def gen_multiscale(
    input_json: str,
    save_dir: str,
    llm: LLMModel,
    windows: List[int] = (30, 180, 600),
    granularity_names: List[str] = ("30sec", "3min", "10min"),
    perspective: str = "general",
    date: str = "DAY1",
):
    """
    Generate multiscale event summaries from fine-grained captions.

    Reads a base caption JSON and hierarchically summarizes into
    coarser granularities: each level reads the previous level's output.

    Args:
        input_json: Path to the base caption JSON (e.g., 10sec.json).
        save_dir: Directory to write output JSON files.
        llm: LLMModel instance for summarization calls.
        windows: Window sizes in seconds for each level.
        granularity_names: Output filenames (without .json) for each level.
        perspective: "egocentric" for first-person or "general" for third-person prompts.
        date: Fallback date label for entries without one.
    """
    assert len(windows) == len(granularity_names)
    prompts = _PROMPTS.get(perspective, _PROMPTS["general"])

    os.makedirs(save_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        prev_level_data = json.load(f)

    for level, (window_sec, name) in enumerate(zip(windows, granularity_names)):
        output_path = os.path.join(save_dir, f"{name}.json")

        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    prev_level_data = json.load(f)
                print(f"  {name} already exists, skipping: {output_path}")
                continue
            except json.JSONDecodeError:
                print(f"  {name} is corrupt, regenerating...")

        sys_msg = prompts[min(level, len(prompts) - 1)]
        print(f"  Generating {name} (window={window_sec}s) from {len(prev_level_data)} entries...")

        results = _summarize_level(prev_level_data, window_sec, sys_msg, llm, date)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"    -> {len(results)} entries written to {output_path}")
        prev_level_data = results
