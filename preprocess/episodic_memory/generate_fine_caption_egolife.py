#!/usr/bin/env python3
"""
Generate first-person video captions from sync files using the worldmm LLMModel.
Processes EgoLife sync files and rewrites the camera wearer's perspective into I/me.
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from worldmm.llm import LLMModel

model = LLMModel(model_name="gpt-5-mini")

SYSTEM_PROMPT_TEMPLATE = """# Role and Objective

You are an expert egocentric video captioner. Your task is to transform raw video caption and transcript segments into concise, item-focused, first-person captions that clearly describe key actions, objects, and interactions from the camera wearer's perspective.

# Input Data Format

The input will be a JSON array where each element is an object with these required fields:

- `start`: Start timestamp in seconds (integer or float)
- `end`: End timestamp in seconds (integer or float)
- `text`: Caption or transcript content (string)
- `type`: Specifies either `caption` (visual action) or `transcript` (spoken dialogue)

# Caption Handling

- Captions describe what the camera wearer does; refine or merge them for clarity and focus on items/actions.
- Emphasize objects, tools in use, and the wearer's interactions.

# Transcript Handling

- Transcripts are lines of spoken dialogue.
- Preserve other speakers' names (e.g., `Shure:`).
- For the camera wearer, remove the name (e.g., `__SPEAKER__:`) and present as direct first-person speech.

# Guidelines

1. Use first-person language (I/me/my).
2. Highlight key objects, tools, or interactions.
3. Keep captions concise, direct, and focused on items or actions.
4. Merge overlapping or redundant content.
5. Ensure captions remain in timestamp order and correspond to the original sequence.
6. Favor clarity and succinctness over extensive narration.

# Output Requirement

Output **only** the final consolidated caption string in plain text. Do **not** include checklists, explanations, timestamps, JSON, or any other formatting.

# Examples

**Example 1**
Input:
```json
[
  {"start": 120200, "end": 120202, "text": "Reach down and open the drawer.", "type": "caption"},
  {"start": 120203, "end": 120205, "text": "Grab the scissors.", "type": "caption"},
  {"start": 120206, "end": 120207, "text": "__SPEAKER__: These should work.", "type": "transcript"}
]
```

Output:
```
I open the drawer and grab the scissors. "These should work," I say.
```

**Example 2**
Input:
```json
[
  {"start": 130310, "end": 130311, "text": "Set the plate on the table.", "type": "caption"},
  {"start": 130312, "end": 130313, "text": "__SPEAKER__: Make sure it's centered.", "type": "transcript"},
  {"start": 130315, "end": 130316, "text": "Shure: Got it.", "type": "transcript"}
]
```

Output:
```
I place the plate on the table. "Make sure it's centered," I say. Shure replies, "Got it."
```
"""


def get_speaker_name(person: str) -> str:
    """Extract a display speaker name from a person id such as A1_JAKE."""
    _, _, speaker = person.partition("_")
    speaker = speaker or person
    return speaker.replace("_", " ").title()


def build_system_prompt(person: str) -> str:
    """Return the system prompt customized to the current camera wearer."""
    return SYSTEM_PROMPT_TEMPLATE.replace("__SPEAKER__", get_speaker_name(person))


def load_sync_files(sync_dir: str, person: str) -> List[str]:
    """Return sorted list of sync files for the requested person."""
    pattern = os.path.join(sync_dir, f"{person}*.json")
    return sorted(glob.glob(pattern))


def extract_time_segments(sync_data: List[Dict]) -> List[Dict]:
    """Group consecutive caption/transcript entries by video file."""
    segments: List[Dict] = []

    for video_entry in sync_data:
        video_file = video_entry.get("video_file", "")
        data_entries = video_entry.get("data", [])

        if not data_entries:
            continue

        current_segment: List[Dict] = []
        for entry in data_entries:
            if isinstance(entry, dict) and "text" in entry:
                current_segment.append(entry)
                continue

            if current_segment:
                segments.append({"video_file": video_file, "entries": current_segment.copy()})
                current_segment = []

        if current_segment:
            segments.append({"video_file": video_file, "entries": current_segment.copy()})

    return segments


def create_prompt(segment_entries: List[Dict]) -> str:
    """Build a JSON prompt for the LLM."""
    return json.dumps(segment_entries, indent=2)


def normalize_camera_wearer_text(text: str, person: str) -> str:
    """Rewrite the camera wearer's explicit name back into first person."""
    speaker_name = re.escape(get_speaker_name(person))
    text = re.sub(rf"(?<!\w){speaker_name}:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){speaker_name}'s(?!\w)", "my", text, flags=re.IGNORECASE)
    text = re.sub(rf"(?<!\w){speaker_name}(?!\w)", "I", text, flags=re.IGNORECASE)
    return text


def generate_caption(segment_entries: List[Dict], person: str) -> str:
    """Generate caption text using the LLM."""
    try:
        prompt = create_prompt(segment_entries)
        content = model.generate(
            [
                {"role": "system", "content": build_system_prompt(person)},
                {"role": "user", "content": prompt},
            ]
        )

        generated_text = content.strip() if content else ""
        return normalize_camera_wearer_text(generated_text, person)

    except Exception as e:
        tqdm.write(f"      Error generating caption: {e}")
        return "Error generating caption."


def extract_time_info(segment_entries: List[Dict]) -> Tuple[str | None, str | None]:
    """Return earliest start and latest end time as strings, or (None, None)."""
    if not segment_entries:
        return None, None

    start_times = [entry.get("start") for entry in segment_entries if entry.get("start") is not None]
    end_times = [entry.get("end") for entry in segment_entries if entry.get("end") is not None]

    if not start_times or not end_times:
        return None, None

    start_time = f"{int(min(start_times)):06d}00"
    end_time = f"{int(max(end_times)):06d}00"
    return start_time, end_time


def extract_date_from_filename(filename: str) -> str:
    """Extract day information from filename."""
    match = re.search(r"DAY(\d+)", filename)
    return f"DAY{match.group(1)}" if match else "DAY1"


def extract_day_number(day_str: str) -> int:
    """Extract numeric value from DAY string (e.g., 'DAY6' -> 6)."""
    match = re.search(r"DAY(\d+)", day_str)
    return int(match.group(1)) if match else 0


def create_video_path(video_file: str, date: str, person: str) -> str:
    """Create video path from video filename with correct DAY directory."""
    return f"data/EgoLife/{person}/{date}/{video_file}" if video_file else ""


def build_caption_entry(entries: List[Dict], video_file: str, sync_file: str, caption_text: str, person: str) -> Dict:
    """Assemble the final caption dictionary."""
    start_time, end_time = extract_time_info(entries)
    if not start_time or not end_time:
        return {}

    segment_date = extract_date_from_filename(sync_file)
    return {
        "start_time": start_time,
        "end_time": end_time,
        "text": caption_text,
        "date": segment_date,
        "video_path": create_video_path(video_file, segment_date, person),
    }


def process_sync_files(sync_dir: str, output_file: str, person: str, overwrite: bool = False) -> None:
    """Process all sync files and generate captions in parallel."""

    if os.path.exists(output_file) and not overwrite:
        print(f"Finished fine caption generation: processed=0 skipped=1 failed=0 output={output_file}")
        return

    sync_files = load_sync_files(sync_dir, person)
    if not sync_files:
        print(f"No sync files found for {person}!")
        return

    print(f"Found {len(sync_files)} sync files to process...")
    all_captions = []

    for sync_file in sync_files:
        print(f"\033[91mProcessing: {os.path.basename(sync_file)}\033[0m")
        try:
            with open(sync_file, "r", encoding="utf-8") as f:
                sync_data = json.load(f)
            segments = extract_time_segments(sync_data)
            usable_segments = [segment for segment in segments if segment.get("entries")]
            segment_entries_list = [segment["entries"] for segment in usable_segments]
            segment_video_files = [segment["video_file"] for segment in usable_segments]

            results = []
            with ThreadPoolExecutor() as executor:
                future_to_idx = {
                    executor.submit(generate_caption, entries, person): idx
                    for idx, entries in enumerate(segment_entries_list)
                }
                progress = tqdm(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="  Generating captions",
                    unit="segment",
                    leave=False,
                )
                for future in progress:
                    idx = future_to_idx[future]
                    try:
                        caption_text = future.result()
                    except Exception as e:
                        tqdm.write(f"      Error generating caption: {e}")
                        caption_text = "Error generating caption."
                    results.append((idx, caption_text))

            ordered_results = sorted(results, key=lambda x: x[0])
            for i, caption_text in ordered_results:
                caption_entry = build_caption_entry(
                    entries=segment_entries_list[i],
                    video_file=segment_video_files[i],
                    sync_file=sync_file,
                    caption_text=caption_text,
                    person=person,
                )
                if caption_entry:
                    all_captions.append(caption_entry)
                    tqdm.write(
                        f"    Generated caption for {caption_entry['video_path']}: "
                        f"{caption_entry['start_time']}-{caption_entry['end_time']}"
                    )
        except Exception as e:
            print(f"    \033[91mError processing {sync_file}: {e}\033[0m")
            continue

    print(f"\nSorting {len(all_captions)} captions by date and start time...")
    all_captions.sort(key=lambda x: (extract_day_number(x["date"]), int(x["start_time"])))
    print("Saving output file...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=4, ensure_ascii=False)
    print(f"\nGenerated {len(all_captions)} captions")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate first-person video captions using the worldmm LLMModel")
    parser.add_argument("--person", default="A1_JAKE", help="Person identifier to process, e.g. A1_JAKE.")
    parser.add_argument("--sync-dir", default="data/EgoLife/EgoLifeCap/Sync", help="Directory containing sync files")
    parser.add_argument("--output", default=None, help="Output file path. Defaults to data/EgoLife/EgoLifeCap/<person>/<person>_30sec.json.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files.")

    args = parser.parse_args()
    output_file = args.output or f"data/EgoLife/EgoLifeCap/{args.person}/{args.person}_30sec.json"
    process_sync_files(args.sync_dir, output_file, args.person, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
