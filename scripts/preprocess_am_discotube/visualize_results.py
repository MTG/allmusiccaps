#!/usr/bin/env python
"""
Streamlit app to visualize music description generation results.

This app displays:
- YouTube video player for each sample
- The prompt used to generate the description
- Generated captions from the model
- Old reference captions (if available)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


@st.cache_data
def load_jsonl(file_path: Path) -> List[Dict]:
    """Load all records from a JSONL file."""
    if not file_path.exists():
        return []

    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@st.cache_data
def load_all_generations(out_dir: Path) -> Dict[str, Dict]:
    """Load all generated responses from the output directory.

    Args:
        out_dir: Directory containing subdirectories with .response.json files

    Returns:
        Dictionary mapping sample_id to parsed response data
    """
    generations = {}

    if not out_dir.exists():
        return generations

    # Search for all .response.json files in subdirectories
    for response_file in out_dir.rglob("*.response.json"):
        sample_id = response_file.stem.replace(".response", "")
        with open(response_file, "r") as f:
            generations[sample_id] = json.load(f)

    return generations


@st.cache_data
def load_all_prompts(out_dir: Path) -> Dict[str, str]:
    """Load all prompts from the output directory.

    Args:
        out_dir: Directory containing subdirectories with .query files

    Returns:
        Dictionary mapping sample_id to prompt text
    """
    prompts = {}

    if not out_dir.exists():
        return prompts

    # Search for all .query files in subdirectories
    for query_file in out_dir.rglob("*.query"):
        sample_id = query_file.stem
        prompts[sample_id] = query_file.read_text()

    return prompts


@st.cache_data
def load_old_annotations(annotation_file: Path) -> Dict[str, Dict]:
    """Load old reference annotations from JSONL file.

    Args:
        annotation_file: Path to the old annotations JSONL file

    Returns:
        Dictionary mapping sample_id to annotation data
    """
    annotations = {}

    if not annotation_file.exists():
        return annotations

    records = load_jsonl(annotation_file)
    for record in records:
        # Try different possible ID fields
        for yt_id, values in record.items():
            for d_id, sentences in values.items():
                annotations[str(yt_id)] = sentences

    return annotations


def render_youtube_video(youtube_id: str, height: int = 300):
    """Render an embedded YouTube video player with custom height."""
    youtube_url = f"https://www.youtube.com/embed/{youtube_id}"
    # Use HTML iframe for better size control
    iframe_html = f"""
    <iframe width="100%" height="{height}" src="{youtube_url}" 
            frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen></iframe>
    """
    st.markdown(iframe_html, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Music Description Results", layout="wide")
    st.title("🎵 Music Description Generation Results")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    out_dir = st.sidebar.text_input(
        "Output Directory",
        value="sample_outputs_vllm",
        help="Directory containing generated responses",
    )

    old_annotation_file = (
        "/path/to/discotube/metadata/Qwen_Qwen2.5-32B__chatgpt_v2__t0.5__1.1.jsonl"
    )

    # Load all data
    out_dir_path = Path(out_dir)
    annotation_file_path = Path(old_annotation_file)

    with st.spinner("Loading data..."):
        generations = load_all_generations(out_dir_path)
        prompts = load_all_prompts(out_dir_path)
        old_annotations = load_old_annotations(annotation_file_path)

    if not generations:
        st.error(f"No generations found in {out_dir}. Please check the directory path.")
        st.info("Make sure inference_vllm.py has been run and generated output files.")
        return

    # Display statistics
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Samples", len(generations))
    st.sidebar.metric("Samples with Prompts", len(prompts))
    st.sidebar.metric("Reference Annotations", len(old_annotations))

    # Get list of sample IDs
    sample_ids = sorted(generations.keys())

    # Sample selector
    st.sidebar.markdown("---")
    st.sidebar.header("Sample Selection")

    # Option to navigate by index or search by ID
    nav_mode = st.sidebar.radio("Navigation Mode", ["By Index", "Search by ID"])

    if nav_mode == "By Index":
        sample_idx = st.sidebar.number_input(
            "Sample Index", min_value=0, max_value=len(sample_ids) - 1, value=0, step=1
        )
        selected_id = sample_ids[sample_idx]
    else:
        search_query = st.sidebar.text_input("Search Sample ID", value="")
        matching_ids = [
            sid for sid in sample_ids if search_query.lower() in sid.lower()
        ]

        if matching_ids:
            selected_id = st.sidebar.selectbox("Matching IDs", matching_ids)
        else:
            st.warning("No matching sample IDs found")
            selected_id = sample_ids[0]

    # Navigation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    current_idx = sample_ids.index(selected_id)

    if col1.button("⬅️ Previous") and current_idx > 0:
        st.rerun()

    col2.markdown(f"**{current_idx + 1}/{len(sample_ids)}**")

    if col3.button("Next ➡️") and current_idx < len(sample_ids) - 1:
        st.rerun()

    # Display selected sample
    st.header(f"Sample: {selected_id}")

    # Create two-column layout
    left_col, right_col = st.columns([1.5, 1.5])

    with left_col:
        st.markdown("#### 📹 Video")
        try:
            render_youtube_video(selected_id, height=250)
        except Exception as e:
            st.error(f"Failed to load video: {e}")

        # Generated data - with smaller font
        st.markdown("<h5>Generated Data</h5>", unsafe_allow_html=True)
        st.markdown(
            """<style>
            .small-json { font-size: 0.85em !important; }
            .small-json pre { font-size: 0.85em !important; }
            </style>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="small-json">', unsafe_allow_html=True)
        generated_data = generations[selected_id]
        st.json(generated_data)
        st.markdown("</div>", unsafe_allow_html=True)

        # Reference data - with smaller font
        st.markdown("<h5>Reference Annotations</h5>", unsafe_allow_html=True)
        old_data = old_annotations.get(selected_id, {})
        if old_data:
            st.markdown('<div class="small-json">', unsafe_allow_html=True)
            st.json(old_data)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No reference annotation available for this sample")

    with right_col:
        # Prompt at the top with small font
        st.markdown("<h5>📝 Original Prompt</h5>", unsafe_allow_html=True)
        if selected_id in prompts:
            st.markdown(
                f"<div style='font-size: 0.8em; color: #333; background-color: #f0f0f0; padding: 10px; border-radius: 5px; white-space: pre-wrap; max-height: 800px; overflow-y: auto;'>{prompts[selected_id]}</div>",
                unsafe_allow_html=True,
            )
            # st.markdown(
            #     f"<div style='font-size: 0.75em; color: #666; background-color: #f0f0f0; padding: 10px; border-radius: 5px; white-space: pre-wrap;'>{prompts[selected_id]}</div>",
        else:
            st.warning("Prompt not found")


if __name__ == "__main__":
    main()
