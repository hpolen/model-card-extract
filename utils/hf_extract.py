import re
from typing import Any, Dict, Optional, List
from huggingface_hub import ModelCard, model_info

def as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def extract_repo_id(url_or_id: str) -> str:
    m = re.search(r"huggingface\\.co/(?:models/)?([^/?#]+/[^/?#]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def parse_params_b(text: str) -> Optional[float]:
    m = re.search(r"(\\d+(?:\\.\\d+)?)\\s*B\\b", text, re.I)
    return float(m.group(1)) if m else None

def build_markdown_and_meta(repo_id: str) -> Dict[str, Any]:
    card = ModelCard.load(repo_id)
    info = model_info(repo_id)

    card_data = info.cardData or {}
    license_     = card_data.get("license") or getattr(card.data, "license", None)
    pipeline     = card_data.get("pipeline_tag") or getattr(card.data, "pipeline_tag", None)
    tags         = as_list(card_data.get("tags") or getattr(card.data, "tags", []))
    datasets     = as_list(card_data.get("datasets") or getattr(card.data, "datasets", []))
    metrics      = as_list(card_data.get("metrics") or getattr(card.data, "metrics", []))
    languages    = as_list(card_data.get("language") or getattr(card.data, "language", []))
    base_model   = card_data.get("base_model") or getattr(card.data, "base_model", None)
    library_name = card_data.get("library_name") or getattr(card.data, "library_name", None)
    model_type   = card_data.get("model_type") or getattr(card.data, "model_type", None)

    last_mod  = getattr(info, "lastModified", None)
    sha       = getattr(info, "sha", None)
    downloads = getattr(info, "downloads", "N/A")
    likes     = getattr(info, "likes", 0)

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
    meta = {
        "repo_id": repo_id,
        "license": license_,
        "datasets": datasets,
        "training_data": datasets,
        "data_license": card_data.get("data_license"),
        "downloads_30d": downloads if isinstance(downloads, (int, float)) else 0,
        "last_modified": last_mod,
        "likes": likes if isinstance(likes, (int, float)) else 0,
        "params_b": parse_params_b(card.content or ""),
        "card_text": card.content or "",
        "pipeline_tag": pipeline,
        "library_name": library_name,
        "model_type": model_type,
        "tags": tags,
        "base_model": base_model,
        "sha": sha
    }
    return {"markdown": md, "meta": meta}
