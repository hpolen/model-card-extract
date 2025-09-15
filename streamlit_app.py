import re
import io
import datetime as dt
import streamlit as st
from huggingface_hub import ModelCard, model_info

st.set_page_config(page_title="HF Model Card → Markdown", page_icon="📄", layout="centered")

st.title("📄 Hugging Face Model Card → Markdown")
st.write("Paste a **Hugging Face model URL or repo ID** (e.g., `meta-llama/Llama-3.1-8B`) and click **Generate** to download a Markdown summary.")

def extract_repo_id(url_or_id: str) -> str:
    m = re.search(r"huggingface\.co/(?:models/)?([^/?#]+/[^/?#]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

@st.cache_data(show_spinner=False)
def build_markdown(repo_id: str) -> str:
    card = ModelCard.load(repo_id)
    info = model_info(repo_id)

    card_data = info.cardData or {}
    # Common front-matter keys (with safe fallbacks)
    license_     = card_data.get("license") or getattr(card.data, "license", None)
    pipeline     = card_data.get("pipeline_tag") or getattr(card.data, "pipeline_tag", None)
    tags         = as_list(card_data.get("tags") or getattr(card.data, "tags", []))
    datasets     = as_list(card_data.get("datasets") or getattr(card.data, "datasets", []))
    metrics      = as_list(card_data.get("metrics") or getattr(card.data, "metrics", []))
    languages    = as_list(card_data.get("language") or getattr(card.data, "language", []))
    base_model   = card_data.get("base_model") or getattr(card.data, "base_model", None)
    library_name = card_data.get("library_name") or getattr(card.data, "library_name", None)
    model_type   = card_data.get("model_type") or getattr(card.data, "model_type", None)

    # Some fields on ModelInfo
    last_mod = getattr(info, "lastModified", None)
    sha      = getattr(info, "sha", None)
    downloads = getattr(info, "downloads", "N/A")

    md = f"""# {repo_id} – Model Summary

**Repo:** https://huggingface.co/{repo_id}  
**Last modified:** {last_mod}  
**SHA (main):** {sha}  
**Downloads (30d):** {downloads}  

## Key Facts
- **License:** {license_ or "—"}
- **Pipeline tag:** {pipeline or "—"}
- **Library:** {library_name or "—"}
- **Model type:** {model_type or "—"}
- **Base model:** {base_model or "—"}
- **Languages:** {", ".join(languages) or "—"}
- **Tags:** {", ".join(tags) or "—"}
- **Datasets:** {", ".join(datasets) or "—"}
- **Reported metrics:** {", ".join([m if isinstance(m, str) else m.get('name','') for m in metrics]) or "—"}

## Full Model Card
{card.content}
"""
    return md

user_input = st.text_input("Model URL or repo ID", placeholder="e.g., meta-llama/Llama-3.1-8B")
generate = st.button("Generate Markdown", type="primary")

if generate:
    if not user_input.strip():
        st.error("Please paste a Hugging Face model URL or repo ID.")
    else:
        try:
            with st.spinner("Fetching model card…"):
                repo_id = extract_repo_id(user_input)
                md = build_markdown(repo_id)
                safe_name = repo_id.replace("/", "__")
                fname = f"{safe_name}.md"
            st.success("Generated!")
            st.download_button(
                label="⬇️ Download Markdown",
                data=md.encode("utf-8"),
                file_name=fname,
                mime="text/markdown",
            )
            with st.expander("Preview"):
                st.code(md, language="markdown")
        except Exception as e:
            st.error(f"Failed to build Markdown: {e}")
            st.stop()

st.caption("Tip: For private/gated repos, run `huggingface-cli login` in your environment first.")