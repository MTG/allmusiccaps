import streamlit as st
import glob
import os
import numpy as np
from scipy.spatial.distance import cosine
import torch
from pathlib import Path

from amclap import get_model
from mtrpp.utils.eval_utils import load_ttmr_pp


model_dir = Path("/home/<user>/reps/clap-mtg/model_weights")
baseline_model = "id_ttmrpp"


def load_embeddings(embed_dir):
    embeddings = {}
    from pathlib import Path

    for file in Path(embed_dir).rglob("*.npy"):
        key = file.stem
        embeddings[key] = np.fromfile(file, dtype=np.float32)

    return embeddings


@st.cache_data
def load_baseline_embeddings(embed_dir):
    return load_embeddings(embed_dir)


@st.cache_data
def load_omar_embeddings(embed_dir):
    return load_embeddings(embed_dir)


# Init streamlit app
st.title("Retrieval Demo")

# Use glob to search all directories in `embeddings_/`, every folder corresponds to a model
model_dirs = [d for d in glob.glob("embeddings/*") if os.path.isdir(d)]
model_dirs = [m for m in model_dirs if m != f"embeddings/{baseline_model}"]

if not model_dirs:
    st.error("No model directories found in 'embeddings_/'. Please add some models.")
else:
    # Select a model with a dropdown
    selected_model = st.selectbox("Select a CLAP OMAR model", model_dirs)

    # Load all embeddings and put them in a simple dict
    embeddings_omar = load_omar_embeddings(selected_model)
    embeddings_baseline = load_baseline_embeddings(f"embeddings/{baseline_model}")

    model_id = Path(selected_model).stem[3:]

    model_cfg_dir = model_dir / model_id / "checkpoints"
    model_cfg_file = list(model_cfg_dir.glob("*.gin"))[0]

    model = get_model(config_file=model_cfg_file)

    save_dir = "/home/<user>/music-text-representation-pp/"
    model_baseline, sr, duration = load_ttmr_pp(save_dir=save_dir, model_types="best")

    # Print how many audio embeddings we have
    st.write(f"Loaded {len(embeddings_omar)} OMAR embeddings.")
    st.write(f"Loaded {len(embeddings_baseline)} TTMR embeddings.")

    # Show a text input to enter a query
    query = st.text_input("Enter your query:")

    if query:
        # Compute cosine similarity between the text embedding and all audio embeddings
        with torch.no_grad():
            text_embedding = model.forward_text([query])
            text_embedding = text_embedding.cpu().numpy().squeeze()

        similarities_omar = {
            key: 1 - cosine(text_embedding, emb) for key, emb in embeddings_omar.items()
        }

        with torch.no_grad():
            text_embedding = model_baseline.text_forward([query])
            text_embedding = text_embedding.cpu().numpy().squeeze()

        similarities_baseline = {
            key: 1 - cosine(text_embedding, emb)
            for key, emb in embeddings_baseline.items()
        }

        # Show the top 5 results with audio players for OMAR and baseline
        top_5_omar = sorted(
            similarities_omar.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_5_baseline = sorted(
            similarities_baseline.items(), key=lambda x: x[1], reverse=True
        )[:5]

        col1, col2 = st.columns(2)

        with col1:
            st.header(f"Baseline\n(TTMR-PP)")
            for y_id, score in top_5_baseline:
                st.markdown(
                    f"""
        <iframe width="300" height="200"
        src="https://www.youtube.com/embed/{y_id}"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
        </iframe>
        """,
                    unsafe_allow_html=True,
                )

                st.write(f"Similarity: {score:.3f}")

        with col2:
            st.header(f"CLAP OMAR\n({model_id})")
            for y_id, score in top_5_omar:
                st.markdown(
                    f"""
        <iframe width="300" height="200"
        src="https://www.youtube.com/embed/{y_id}"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
        </iframe>
        """,
                    unsafe_allow_html=True,
                )

                st.write(f"Similarity: {score:.3f}")
