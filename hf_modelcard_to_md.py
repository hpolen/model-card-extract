# hf_modelcard_to_md.py
import re, sys, textwrap, yaml
from pathlib import Path
from huggingface_hub import ModelCard, model_info

def extract_repo_id(url_or_id: str) -> str:
    # Accept either a full URL like https://huggingface.co/owner/name
    # or a repo_id like owner/name
    m = re.search(r"huggingface\.co/(?:models/)?([^/?#]+/[^/?#]+)", url_or_id)
    return m.group(1) if m else url_or_id

def as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def main(arg):
    repo_id = extract_repo_id(arg)
    card = ModelCard.load(repo_id)                # full Markdown, with YAML block
    info = model_info(repo_id)                    # hub metadata + parsed cardData

    # card.data is a structured object; fall back to info.cardData when needed
    card_data = info.cardData or {}

    # Common keys authors put in YAML/front-matter:
    # license, pipeline_tag, tags, datasets, metrics, language, base_model, library_name, model-type, etc.
    license_      = card_data.get("license") or getattr(card.data, "license", None)
    pipeline      = card_data.get("pipeline_tag") or getattr(card.data, "pipeline_tag", None)
    tags          = as_list(card_data.get("tags") or getattr(card.data, "tags", []))
    datasets      = as_list(card_data.get("datasets") or getattr(card.data, "datasets", []))
    metrics       = as_list(card_data.get("metrics") or getattr(card.data, "metrics", []))
    languages     = as_list(card_data.get("language") or getattr(card.data, "language", []))
    base_model    = card_data.get("base_model") or getattr(card.data, "base_model", None)
    library_name  = card_data.get("library_name") or getattr(card.data, "library_name", None)
    model_type    = card_data.get("model_type") or getattr(card.data, "model_type", None)

    md = f"""# {repo_id} – Model Summary

**Repo:** https://huggingface.co/{repo_id}  
**Last modified:** {info.lastModified}  
**SHA (main):** {info.sha}  
**Downloads (30d):** {getattr(info, "downloads", "N/A")}  

## Key Facts
- **License:** {license_ or "—"}
- **Pipeline tag:** {pipeline or "—"}
- **Library:** {library_name or "—"}
- **Model type:** {model_type or "—"}
- **Base model:** {base_model or "—"}
- **Languages:** {", ".join(languages) or "—"}
- **Tags:** {", ".join(tags) or "—"}
- **Datasets:** {", ".join(datasets) or "—"}
- **Reported metrics:** {", ".join([m if isinstance(m,str) else m.get('name','') for m in metrics]) or "—"}

## Full Model Card
{card.content}
"""
    out = Path(f"{repo_id.replace('/','__')}.md")
    out.write_text(md, encoding="utf-8")
    print(f"Written: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hf_modelcard_to_md.py <repo_id or HF URL>")
        sys.exit(1)
    main(sys.argv[1])
