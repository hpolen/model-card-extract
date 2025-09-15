import os
import re
import json
import pathlib
from typing import Dict, Any, Optional

import streamlit as st
from huggingface_hub import ModelCard, model_info

# --- Environment hardening (works locally & on Streamlit Cloud) ---
# Local cache inside the repo to avoid ~/.cache permission issues
os.environ.setdefault("HF_HOME", str(pathlib.Path(__file__).with_name("hf_cache")))

# Safely read an HF token (optional). Do NOT error if secrets are missing.
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
try:
    # st.secrets may raise if no secrets file; guard it.
    _secrets = dict(st.secrets)  # casting triggers parse once
except Exception:
    _secrets = {}
hf_token = hf_token or _secrets.get("HF_TOKEN")
if hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

# --- Streamlit page setup ---
st.set_page_config(page_title="HF Model Card → Markdown + Risk Score", page_icon="📄", layout="centered")
st.title("📄 Hugging Face Model Card → Markdown + 🔐 Risk Score")
st.write("Paste a **Hugging Face model URL or repo ID** (e.g., `meta-llama/Llama-3.1-8B`), then generate a Markdown summary and a risk scorecard.")

# ==========================
# Helpers: extraction / parsing
# ==========================
def extract_repo_id(url_or_id: str) -> str:
    m = re.search(r"huggingface\.co/(?:models/)?([^/?#]+/[^/?#]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def normalize_license(lic):
    if not lic:
        return "unknown"
    s = lic.strip().lower().replace(" ", "-")
    aliases = {
        "apache2": "apache-2.0",
        "apache-2": "apache-2.0",
        "apache-2.0": "apache-2.0",
        "mit": "mit",
        "bsd": "bsd-3-clause",
        "bsd-3": "bsd-3-clause",
        "gpl": "gpl-3.0",
        "gpl-3": "gpl-3.0",
        "cc-by-nc": "cc-by-nc-4.0",
        "cc-by-nc-4.0": "cc-by-nc-4.0",
    }
    return aliases.get(s, s)

def parse_params_b(card_text: str) -> Optional[float]:
    # Matches "7B", "13B", "70B" (optionally with 'parameters' nearby)
    m = re.search(r'(\d+(?:\.\d+)?)\s*B\b', card_text, re.I)
    return float(m.group(1)) if m else None

# ==========================
# Default Risk Policy (overridden by risk_policy.json if present)
# ==========================
DEFAULT_POLICY = {
    "weights": {
        "license": 2,
        "data_transparency": 2,
        "security_provenance": 2,
        "maturity_support": 1,
        "compliance_alignment": 2,
        "technical_feasibility": 1
    },
    "license": {
        "allow": ["apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"],
        "warn":  ["cc-by-4.0", "lgpl-3.0", "mpl-2.0", "epl-2.0"],
        "deny":  ["cc-by-nc-4.0", "gpl-3.0", "proprietary", "unknown", "no-license"]
    },
    "security_provenance": {
        "trusted_owners": ["meta-llama", "mistralai", "tiiuae", "microsoft", "google", "huggingface"],
        "min_downloads_30d": 1000
    },
    "data_transparency": {
        "require_any_of": ["datasets", "training_data", "data_license"]
    },
    "compliance_alignment": {
        "keywords_ok": ["hipaa", "pci", "gdpr", "pii handling", "privacy"],
        "keywords_bad": ["no restrictions", "unrestricted", "not for production"]
    },
    "technical_feasibility": {
        "max_params_b": 70,
        "warn_params_b": 20
    }
}

def load_policy(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_POLICY
    
# --- Policy Editor UI helpers ---
def render_policy_editor(policy: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.subheader("⚙️ Risk Policy Editor")

    # Weights
    st.sidebar.markdown("**Weights** (higher = more impact)")
    weights = {}
    for k, v in policy.get("weights", {}).items():
        weights[k] = st.sidebar.slider(f"{k}", 0, 5, int(v), 1)
    policy["weights"] = weights

    # License buckets
    st.sidebar.markdown("---")
    st.sidebar.markdown("**License policy**")
    allow = st.sidebar.text_area(
        "Allowed (comma-separated SPDX)",
        ",".join(policy["license"]["allow"])
    )
    warn = st.sidebar.text_area(
        "Warn (comma-separated SPDX)",
        ",".join(policy["license"]["warn"])
    )
    deny = st.sidebar.text_area(
        "Deny (comma-separated SPDX or keywords)",
        ",".join(policy["license"]["deny"])
    )
    policy["license"]["allow"] = [s.strip() for s in allow.split(",") if s.strip()]
    policy["license"]["warn"]  = [s.strip() for s in warn.split(",") if s.strip()]
    policy["license"]["deny"]  = [s.strip() for s in deny.split(",") if s.strip()]

    # Security provenance
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Security provenance**")
    owners = st.sidebar.text_input(
        "Trusted owners (comma-separated)",
        ",".join(policy["security_provenance"]["trusted_owners"])
    )
    policy["security_provenance"]["trusted_owners"] = [s.strip() for s in owners.split(",") if s.strip()]
    policy["security_provenance"]["min_downloads_30d"] = st.sidebar.number_input(
        "Min downloads (30d) for 'community OK'", min_value=0, step=100,
        value=int(policy["security_provenance"]["min_downloads_30d"])
    )

    # Data transparency
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data transparency**")
    req_any = st.sidebar.text_input(
        "Require any of (comma-separated fields)",
        ",".join(policy["data_transparency"]["require_any_of"])
    )
    policy["data_transparency"]["require_any_of"] = [s.strip() for s in req_any.split(",") if s.strip()]

    # Compliance alignment
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Compliance alignment**")
    ok = st.sidebar.text_area(
        "OK keywords (comma-separated)", ",".join(policy["compliance_alignment"]["keywords_ok"])
    )
    bad = st.sidebar.text_area(
        "Bad keywords (comma-separated)", ",".join(policy["compliance_alignment"]["keywords_bad"])
    )
    policy["compliance_alignment"]["keywords_ok"] = [s.strip().lower() for s in ok.split(",") if s.strip()]
    policy["compliance_alignment"]["keywords_bad"] = [s.strip().lower() for s in bad.split(",") if s.strip()]

    # Technical feasibility
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Technical feasibility**")
    policy["technical_feasibility"]["warn_params_b"] = st.sidebar.number_input(
        "Warn at size (B)", min_value=0, step=1,
        value=int(policy["technical_feasibility"]["warn_params_b"])
    )
    policy["technical_feasibility"]["max_params_b"] = st.sidebar.number_input(
        "Max size (B)", min_value=0, step=1,
        value=int(policy["technical_feasibility"]["max_params_b"])
    )

    return policy

def try_persist_policy(policy: Dict[str, Any], policy_path: str) -> bool:
    """Attempt to write to disk. Returns True on success."""
    try:
        with open(policy_path, "w") as f:
            json.dump(policy, f, indent=2)
        return True
    except Exception:
        return False

# ==========================
# Risk scoring engine
# ==========================
def evaluate_dimension(name: str, meta: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    # license
    if name == "license":
        lic = normalize_license(meta.get("license"))
        if lic in policy["license"]["allow"]:
            return {"score": 0, "rationale": f"License {lic} is allowed"}
        if lic in policy["license"]["warn"]:
            return {"score": 1, "rationale": f"License {lic} requires caution"}
        if lic in policy["license"]["deny"]:
            return {"score": 2, "rationale": f"License {lic} is not permitted"}
        return {"score": 2, "rationale": "License unknown or missing"}

    # data_transparency
    if name == "data_transparency":
        fields = policy["data_transparency"]["require_any_of"]
        has_any = any(bool(meta.get(f)) for f in fields)
        return {"score": 0 if has_any else 1, "rationale": "Datasets/training data disclosed" if has_any else "Data sources unclear"}

    # security_provenance
    if name == "security_provenance":
        owner = (meta.get("repo_id", "")).split("/")[0].lower()
        downloads = meta.get("downloads_30d") or 0
        trusted = owner in [o.lower() for o in policy["security_provenance"]["trusted_owners"]]
        if trusted:
            return {"score": 0, "rationale": f"Trusted owner {owner}"}
        if downloads >= policy["security_provenance"]["min_downloads_30d"]:
            return {"score": 1, "rationale": f"Community model with healthy adoption ({downloads} downloads/30d)"}
        return {"score": 2, "rationale": "Low-signal provenance (owner not trusted, low adoption)"}

    # maturity_support
    if name == "maturity_support":
        last_mod = meta.get("last_modified")
        likes = meta.get("likes") or 0
        if last_mod or likes > 200:
            return {"score": 0, "rationale": "Active or well-liked repository"}
        return {"score": 1, "rationale": "Limited maturity signals"}

    # compliance_alignment
    if name == "compliance_alignment":
        txt = (meta.get("card_text") or "").lower()
        bad = any(k in txt for k in policy["compliance_alignment"]["keywords_bad"])
        ok = any(k in txt for k in policy["compliance_alignment"]["keywords_ok"])
        if bad:
            return {"score": 2, "rationale": "Problematic compliance language in card"}
        if ok:
            return {"score": 0, "rationale": "Mentions compliance considerations"}
        return {"score": 1, "rationale": "No explicit compliance guidance in card"}

    # technical_feasibility
    if name == "technical_feasibility":
        params_b = meta.get("params_b")
        if params_b is None:
            return {"score": 1, "rationale": "Model size not stated; capacity risk unknown"}
        if params_b > policy["technical_feasibility"]["max_params_b"]:
            return {"score": 2, "rationale": f"Very large model (~{params_b}B) may exceed infra appetite"}
        if params_b > policy["technical_feasibility"]["warn_params_b"]:
            return {"score": 1, "rationale": f"Large model (~{params_b}B); review infra cost/latency"}
        return {"score": 0, "rationale": f"Model size (~{params_b}B) within appetite"}

    return {"score": 1, "rationale": "Not evaluated"}

def score_model(meta: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    dims = ["license","data_transparency","security_provenance","maturity_support","compliance_alignment","technical_feasibility"]
    details = {}
    total = 0
    max_total = 0
    for d in dims:
        w = policy["weights"].get(d, 1)
        res = evaluate_dimension(d, meta, policy)
        details[d] = {**res, "weight": w, "weighted": res["score"] * w}
        total += res["score"] * w
        max_total += 2 * w  # red=2
    pct = (total / max_total) if max_total else 0
    band = "Green" if pct <= 0.25 else ("Yellow" if pct <= 0.6 else "Red")
    return {"overall": {"score": total, "max": max_total, "band": band, "percent": round(pct*100,1)}, "details": details}

# ==========================
# Build full Markdown (and return some raw facts for scoring)
# ==========================
@st.cache_data(show_spinner=False)
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
    # meta for scoring
    meta = {
        "repo_id": repo_id,
        "license": license_,
        "datasets": datasets,
        "training_data": datasets,  # fallback
        "data_license": card_data.get("data_license"),
        "downloads_30d": downloads if isinstance(downloads, (int, float)) else 0,
        "last_modified": last_mod,
        "likes": likes if isinstance(likes, (int, float)) else 0,
        "params_b": parse_params_b(card.content or ""),
        "card_text": card.content or "",
    }
    return {"markdown": md, "meta": meta}

# ==========================
# UI
# ==========================

policy_path = os.path.join(os.path.dirname(__file__), "risk_policy.json")
policy = load_policy(policy_path)

# Optional admin PIN (only show editor when the PIN matches)
with st.sidebar.expander("Admin / Policy Controls", expanded=False):
    pin_ok = True
    admin_pin = os.getenv("ADMIN_PIN") or _secrets.get("ADMIN_PIN")
    if admin_pin:
        entered = st.text_input("Enter admin PIN to edit policy", type="password")
        pin_ok = (entered == admin_pin)
        if not pin_ok:
            st.info("Enter correct PIN to edit policy.")

    if pin_ok:
        if st.checkbox("Enable Policy Editor"):
            policy = render_policy_editor(policy)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Save policy to file"):
                    if try_persist_policy(policy, policy_path):
                        st.success("Policy saved to risk_policy.json")
                    else:
                        st.warning("Could not write to disk (Cloud is often read-only). Use Download instead.")
            with col_b:
                st.download_button(
                    "⬇️ Download updated policy JSON",
                    data=json.dumps(policy, indent=2).encode("utf-8"),
                    file_name="risk_policy.json",
                    mime="application/json"
                )

user_input = st.text_input("Model URL or repo ID", placeholder="e.g., meta-llama/Llama-3.1-8B")
col1, col2 = st.columns([1,1])
with col1:
    generate = st.button("Generate Markdown + Score", type="primary")
with col2:
    st.write("")  # spacer

if generate:
    if not user_input.strip():
        st.error("Please paste a Hugging Face model URL or repo ID.")
    else:
        try:
            with st.spinner("Fetching model card and computing score…"):
                repo_id = extract_repo_id(user_input)
                built = build_markdown_and_meta(repo_id)
                md = built["markdown"]
                meta = built["meta"]

                # Load policy (use risk_policy.json if present, else default)
                policy_path = os.path.join(os.path.dirname(__file__), "risk_policy.json")
                policy = load_policy(policy_path)

                score = score_model(meta, policy)

                # Build a Markdown scorecard
                safe_name = repo_id.replace("/", "__")
                fname_md = f"{safe_name}.md"
                score_md = f"""# Risk Scorecard – {repo_id}

**Overall:** {score['overall']['band']}  ({score['overall']['score']} / {score['overall']['max']}, {score['overall']['percent']}%)

## Breakdown
"""
                for k, v in score["details"].items():
                    band = ["Green","Yellow","Red"][v["score"]]
                    score_md += f"- **{k.replace('_',' ').title()}**: {band} (w={v['weight']}; weighted={v['weighted']}) — {v['rationale']}\n"

                # UI outputs
                st.success("Generated!")
                st.download_button("⬇️ Download Model Summary (Markdown)", data=md.encode("utf-8"),
                                   file_name=fname_md, mime="text/markdown")

                st.subheader("Risk Score")
                emoji = {"Green":"🟢", "Yellow":"🟡", "Red":"🔴"}[score["overall"]["band"]]
                st.markdown(f"**Overall:** {emoji} **{score['overall']['band']}**  "
                            f"({score['overall']['score']} / {score['overall']['max']}, {score['overall']['percent']}%)")

                st.download_button("⬇️ Download Risk Scorecard (Markdown)",
                                   data=score_md.encode("utf-8"),
                                   file_name=f"{safe_name}_risk.md",
                                   mime="text/markdown")

                st.download_button("⬇️ Download Risk Score (JSON)",
                                   data=json.dumps(score, indent=2).encode("utf-8"),
                                   file_name=f"{safe_name}_risk.json",
                                   mime="application/json")

                with st.expander("Preview: Model Summary (Markdown)"):
                    st.code(md, language="markdown")

                with st.expander("Preview: Risk Scorecard (Markdown)"):
                    st.code(score_md, language="markdown")

        except Exception as e:
            st.error(f"Failed to build Markdown or score model: {e}")
            st.stop()

st.caption("Tips: For private/gated repos, set HF_TOKEN via environment or Streamlit Secrets. "
           "Policy overrides: add a risk_policy.json next to this file to customize scoring.")
