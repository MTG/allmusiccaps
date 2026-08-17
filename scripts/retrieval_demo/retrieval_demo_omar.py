import streamlit as st
import glob
import os
import numpy as np
from scipy.spatial.distance import cosine
import torch
from pathlib import Path

from amclap import get_model
# from mtrpp.utils.eval_utils import load_ttmr_pp

st.set_page_config(layout="wide")

model_dir = Path("/home/<user>/reps/clap-mtg/model_weights")
# baseline_model = "id_ttmrpp"

# insert image "omar-rq.png"


def load_embeddings(embed_dir):
    embeddings = {}
    from pathlib import Path

    for file in Path(embed_dir).rglob("*.npy"):
        key = file.stem
        embeddings[key] = np.fromfile(file, dtype=np.float32)

    return embeddings


@st.cache_data
def load_omar_embeddings(embed_dir):
    return load_embeddings(embed_dir)


# Init streamlit app
# Make centered title
# st.set_page_config(page_title="OMAR-RQ Demo", layout="centered")
# st.markdown("<h1 style='text-align: center;'>OMAR-RQ</h1>", unsafe_allow_html=True)
# st.markdown("### *Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction*")
st.markdown(
    "<h2 style='text-align: center;'>Music retrieval demo using OMAR-RQ</h2>",
    unsafe_allow_html=True,
)
# st.markdown("### Music retrieval demo based on OMAR-RQ")

# st.image("omar-rq.png", width=500)
left, center, right = st.columns([1, 2, 1])
with center:
    st.image("omar-rq.png", use_container_width=True)


# Use glob to search all directories in `embeddings_/`, every folder corresponds to a model
# model_dirs = [d for d in glob.glob("embeddings/*") if os.path.isdir(d)]

# if not model_dirs:
#     st.error("No model directories found in 'embeddings_/'. Please add some models.")
# else:

# Select a model with a dropdown
# selected_model = st.selectbox("Select a CLAP OMAR model", model_dirs)

# HARD-CODED MODEL FOR THE DEMO
selected_model = "embeddings/id_99eycym5/"

# Load all embeddings and put them in a simple dict
embeddings_omar = load_omar_embeddings(selected_model)

model_id = Path(selected_model).stem[3:]

model_cfg_dir = model_dir / model_id / "checkpoints"
model_cfg_file = list(model_cfg_dir.glob("*.gin"))[0]

model = get_model(config_file=model_cfg_file)

# Print how many audio embeddings we have
st.write(f"Loaded {len(embeddings_omar)} embeddings.")

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

    # Show the top 5 results with audio players for OMAR and baseline
    top_5_omar = sorted(similarities_omar.items(), key=lambda x: x[1], reverse=True)[:5]

    cols = st.columns(5)

    for i, (y_id, score) in enumerate(top_5_omar):
        with cols[i]:
            st.markdown(
                f"""
                <iframe width="200" height="200"
                src="https://www.youtube.com/embed/{y_id}"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
                </iframe>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"Similarity: {score:.3f}")
