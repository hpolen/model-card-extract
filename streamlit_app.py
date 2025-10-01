# streamlit_app.py
# HF Model Card → Markdown + Risk Score + Threat Modeling (STRIDE)
# Python 3.9+ compatible

import os
import re
import io
import json
import time
import hashlib
import zipfile
import pathlib
from typing import Dict, Any, Optional

import streamlit as st
from huggingface_hub import ModelCard, model_info

APP_VERSION = "1.4.0-STRIDE"

# ---------- Environment & Secrets ----------
# Keep HF cache local so we don't depend on ~/.cache
os.environ.setdefault("HF_HOME", str(pathlib.Path(__file__).with_name("hf_cache")))

def _safe_secrets() -> Dict[str, str]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}

SECRETS = _safe_secrets()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or SECRETS.get("OPENAI_API_KEY", "")
ADMIN_PIN = os.getenv("ADMIN_PIN") or SECRETS.get("ADMIN_PIN")

# ---------- STRIDE System Prompt ----------
# This prompt instructs GPT to produce a STRIDE-aligned threat model + a JSON register.
STRIDE_SYSTEM_PROMPT = """
You are a seasoned security architect. Perform **threat modeling using the STRIDE framework** for the system described by the user.
Follow this structure and be concise, practical, and enterprise-ready.

### Output Contract (Markdown)
0. **Executive Summary**
   - Business goal, major risks, recommended disposition (Go / Go with controls / No-Go)

1. **Scope & Assets**
   - High-value assets and data classes (PII/PCI/PHI/etc.)
   - Trust boundaries and actors

2. **Context & Assumptions**
   - Key technical assumptions (model hosting, APIs, data flows, controls in place)

3. **ASCII Data Flow Diagram (DFD)**
   - Simple text diagram with trust boundaries annotated

4. **STRIDE Threats & Controls**
   For each STRIDE category (S, T, R, I, D, E), list relevant threats for this system.
   For every threat, include:
   - **Threat**: one-line name
   - **Description**: how/why it could occur, affected assets
   - **Preconditions**: what must be true for exploitation
   - **Impact**: business impact summary
   - **Likelihood**: Low/Medium/High with short rationale
   - **Risk (5×5)**: provide a matrix cell (e.g., 3x4=12) and band (Green/Yellow/Red)
   - **Mitigations**: concrete controls, prioritized
   - **Detection/Monitoring**: signals, alerts, logs
   - **Residual Risk**: after mitigations

5. **Compliance & Legal**
   - Note any applicable PCI/HIPAA/GDPR/etc. implications and gaps

6. **Red-Team / Test Plan**
   - Short list of probes or scenarios to validate mitigations

7. **Operations & Monitoring**
   - Runbooks, model/infra metrics, anomaly signals

8. **Action Plan**
   - A numbered, prioritized list of next steps with owners (generic) and time horizon

### JSON Threat Register (append at the end)
After the markdown, output a fenced JSON code block (```json ... ```) named `threat_register`
with this shape:
{
  "repo_id": "<owner/name>",
  "generated_at": "<ISO8601 UTC>",
  "stride": [
    {
      "category": "S|T|R|I|D|E",
      "threat": "short name",
      "description": "string",
      "assets": ["..."],
      "preconditions": ["..."],
      "likelihood": "Low|Medium|High",
      "impact": "string",
      "risk_5x5": {"likelihood": 1-5, "impact": 1-5, "score": 1-25, "band": "Green|Yellow|Red"},
      "mitigations": ["..."],
      "detection": ["..."],
      "residual_risk": "string"
    }
  ]
}
Make sure the JSON is valid and compact, and put it in a **fenced code block labeled `json`**.
"""

# ---------- Page ----------
st.set_page_config(page_title="HF Model → Markdown + Risk + STRIDE Threat Model", page_icon="🛡️", layout="wide")
st.title("🛡️ Model Due Diligence: HF Model → Markdown + Risk Score + STRIDE Threat Modeling")
st.caption(
    f"App v{APP_VERSION}. Paste a Hugging Face URL or repo ID to generate a Markdown summary and risk score. "
    "Optionally, create a STRIDE threat model using GPT."
)

# ---------- Helpers: Extract / Normalize ----------
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
    m = re.search(r'(\d+(?:\.\d+)?)\s*B\b', card_text, re.I)
    return float(m.group(1)) if m else None

# ---------- Policy (default + loader) ----------
DEFAULT_POLICY: Dict[str, Any] = {
    "policy_version": "1.0.0",
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
        "warn_params_b": 20,
        "max_params_b": 70
    }
}

def load_policy(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_POLICY

# ---------- Risk Scoring ----------
def evaluate_dimension(name: str, meta: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    if name == "license":
        lic = normalize_license(meta.get("license"))
        if lic in policy["license"]["allow"]:
            return {"score": 0, "rationale": f"License {lic} is allowed"}
        if lic in policy["license"]["warn"]:
            return {"score": 1, "rationale": f"License {lic} requires caution"}
        if lic in policy["license"]["deny"]:
            return {"score": 2, "rationale": f"License {lic} is not permitted"}
        return {"score": 2, "rationale": "License unknown or missing"}

    if name == "data_transparency":
        fields = policy["data_transparency"]["require_any_of"]
        has_any = any(bool(meta.get(f)) for f in fields)
        return {"score": 0 if has_any else 1, "rationale": "Datasets/training data disclosed" if has_any else "Data sources unclear"}

    if name == "security_provenance":
        owner = (meta.get("repo_id", "")).split("/")[0].lower()
        downloads = meta.get("downloads_30d") or 0
        trusted = owner in [o.lower() for o in policy["security_provenance"]["trusted_owners"]]
        if trusted:
            return {"score": 0, "rationale": f"Trusted owner {owner}"}
        if downloads >= policy["security_provenance"]["min_downloads_30d"]:
            return {"score": 1, "rationale": f"Community model with healthy adoption ({downloads} downloads/30d)"}
        return {"score": 2, "rationale": "Low-signal provenance (owner not trusted, low adoption)"}

    if name == "maturity_support":
        last_mod = meta.get("last_modified")
        likes = meta.get("likes") or 0
        if last_mod or likes > 200:
            return {"score": 0, "rationale": "Active or well-liked repository"}
        return {"score": 1, "rationale": "Limited maturity signals"}

    if name == "compliance_alignment":
        txt = (meta.get("card_text") or "").lower()
        bad = any(k in txt for k in policy["compliance_alignment"]["keywords_bad"])
        ok = any(k in txt for k in policy["compliance_alignment"]["keywords_ok"])
        if bad:
            return {"score": 2, "rationale": "Problematic compliance language in card"}
        if ok:
            return {"score": 0, "rationale": "Mentions compliance considerations"}
        return {"score": 1, "rationale": "No explicit compliance guidance in card"}

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

# ---------- Model Summary ----------
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
        "pipeline_tag": pipeline,
        "library_name": library_name,
        "model_type": model_type,
        "tags": tags,
        "base_model": base_model,
        "sha": sha
    }
    return {"markdown": md, "meta": meta}

# ---------- OpenAI Helpers ----------
def call_openai(system_prompt: str, user_prompt: str, model_name: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    # Try new SDK first
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception:
        # Legacy SDK fallback
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e2:
            raise RuntimeError(f"OpenAI call failed: {e2}")

def try_extract_json_block(txt: str) -> Optional[str]:
    # Prefer fenced ```json ... ``` blocks
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, re.S | re.I)
    if m:
        return m.group(1)
    # Fallback: brace-balance the first JSON object
    start = txt.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(txt)):
        ch = txt[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return txt[start : i + 1]
    return None

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

# ---------- UI ----------
with st.sidebar:
    st.header("Input")
    user_input = st.text_input("Hugging Face URL or repo ID", placeholder="e.g., meta-llama/Llama-3.1-8B")

    st.header("Threat Modeling (STRIDE) Settings")
    if OPENAI_API_KEY:
        st.success("OpenAI key detected")
    else:
        st.info("Provide OPENAI_API_KEY in env or Streamlit secrets to enable Threat Modeling.")
    tm_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"], index=0)
    tm_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    tm_max_tokens = st.slider("Max tokens", 512, 4000, 2000, 128)

policy_path = os.path.join(os.path.dirname(__file__), "risk_policy.json")
policy = load_policy(policy_path)
policy_version = policy.get("policy_version", "1.0.0")

tab1, tab2 = st.tabs(["📄 Summary + Risk Score", "🧠 STRIDE Threat Modeling"])

# ===== TAB 1 =====
with tab1:
    colA, colB = st.columns([1,1])
    with colA:
        generate = st.button("Generate Markdown + Score", type="primary", use_container_width=True)
    with colB:
        st.write("")

    if generate:
        if not user_input.strip():
            st.error("Please paste a Hugging Face model URL or repo ID.")
        else:
            try:
                with st.spinner("Fetching model card and computing score…"):
                    repo_id = extract_repo_id(user_input.strip())
                    built = build_markdown_and_meta(repo_id)
                    md = built["markdown"]
                    meta = built["meta"]

                    score = score_model(meta, policy)

                    safe_name = repo_id.replace("/", "__")
                    fname_md = f"{safe_name}.md"

                    score_md = f"""# Risk Scorecard – {repo_id}

**Overall:** {score['overall']['band']}  ({score['overall']['score']} / {score['overall']['max']}, {score['overall']['percent']}%)

**Policy version:** {policy_version}

## Breakdown
"""
                    for k, v in score["details"].items():
                        band = ["Green","Yellow","Red"][v["score"]]
                        score_md += f"- **{k.replace('_',' ').title()}**: {band} (w={v['weight']}; weighted={v['weighted']}) — {v['rationale']}\n"

                    st.success("Generated!")
                    st.download_button("⬇️ Download Model Summary (Markdown)", data=md.encode("utf-8"),
                                       file_name=fname_md, mime="text/markdown", use_container_width=True)

                    st.subheader("Risk Score")
                    emoji = {"Green":"🟢", "Yellow":"🟡", "Red":"🔴"}[score["overall"]["band"]]
                    st.markdown(f"**Overall:** {emoji} **{score['overall']['band']}**  "
                                f"({score['overall']['score']} / {score['overall']['max']}, {score['overall']['percent']}%)")
                    st.markdown(f"**Policy version:** `{policy_version}`")

                    score_json = {
                        "app_version": APP_VERSION,
                        "policy_version": policy_version,
                        "repo_id": repo_id,
                        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "overall": score["overall"],
                        "details": score["details"],
                        "facts": {
                            "license": normalize_license(meta.get("license")),
                            "downloads_30d": meta.get("downloads_30d"),
                            "params_b": meta.get("params_b"),
                            "last_modified": meta.get("last_modified"),
                            "owner": repo_id.split("/")[0] if "/" in repo_id else None,
                        },
                    }

                    st.download_button("⬇️ Download Risk Scorecard (Markdown)",
                                       data=score_md.encode("utf-8"),
                                       file_name=f"{safe_name}_risk.md",
                                       mime="text/markdown", use_container_width=True)

                    st.download_button("⬇️ Download Risk Score (JSON)",
                                       data=json.dumps(score_json, indent=2).encode("utf-8"),
                                       file_name=f"{safe_name}_risk.json",
                                       mime="application/json", use_container_width=True)

                    with st.expander("Preview: Model Summary (Markdown)"):
                        st.code(md, language="markdown")
                    with st.expander("Preview: Risk Scorecard (Markdown)"):
                        st.code(score_md, language="markdown")

                    bundle = io.BytesIO()
                    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
                        z.writestr(fname_md, md)
                        z.writestr(f"{safe_name}_risk.md", score_md)
                        z.writestr(f"{safe_name}_risk.json", json.dumps(score_json, indent=2))
                        z.writestr("policy_version.txt", policy_version)
                        z.writestr("app_version.txt", APP_VERSION)
                    st.download_button("⬇️ Download Bundle (ZIP)",
                                       data=bundle.getvalue(),
                                       file_name=f"{safe_name}_bundle.zip",
                                       mime="application/zip", use_container_width=True)

            except Exception as e:
                st.error(f"Failed to build Markdown or score model: {e}")

# ===== TAB 2: STRIDE =====
with tab2:
    if not OPENAI_API_KEY:
        st.info("Provide `OPENAI_API_KEY` via environment or Streamlit secrets to enable STRIDE Threat Modeling.")
    else:
        st.success("Threat Modeling enabled")

    disabled = not (user_input.strip())
    repo_id_preview = extract_repo_id(user_input.strip()) if user_input.strip() else ""
    st.caption(f"Repo to analyze: `{repo_id_preview}`" if repo_id_preview else "Paste a repo ID/URL in the sidebar.")

    purpose = st.text_area("Business Purpose (what are you building & why?)", height=100, disabled=disabled)
    data_classes = st.text_area("Data Classes (PII/PCI/PHI/None; list specific fields if applicable)", height=80, disabled=disabled)
    connectors = st.text_area("External Connectors / Tools (RAG, web, DBs, APIs, plugins)", height=80, disabled=disabled)
    compliance = st.text_input("Compliance overlays (PCI, GDPR, HIPAA, GLBA, SOX, etc.)", disabled=disabled)
    constraints = st.text_area("Operational Constraints (SLA, latency, cost caps, isolation needs)", height=60, disabled=disabled)

    colTM1, colTM2 = st.columns([1,1])
    with colTM1:
        build_tm = st.button("Build STRIDE Prompt (Preview)", disabled=disabled, use_container_width=True)
    with colTM2:
        run_tm = st.button("Generate STRIDE Threat Model (GPT)", disabled=(disabled or not OPENAI_API_KEY), type="primary", use_container_width=True)

    # Build prompt content from model facts
    session = st.session_state

    if build_tm and user_input.strip():
        try:
            repo_id = extract_repo_id(user_input.strip())
            built = build_markdown_and_meta(repo_id)
            meta = built["meta"]
            session["__last_md__"] = built["markdown"]
            session["__last_meta__"] = meta

            # Auto-filled user message sent under STRIDE system prompt
            owner = repo_id.split("/")[0] if "/" in repo_id else ""
            user_prompt = f"""# STRIDE Threat Modeling Input

## System Name
{repo_id}

## Business Purpose
{(purpose or 'N/A').strip()}

## Architecture Summary (derived from model facts)
- Owner: {owner or 'N/A'}
- Library: {meta.get('library_name') or 'N/A'}
- Pipeline/model_type: {meta.get('pipeline_tag') or meta.get('model_type') or 'N/A'}
- Base model: {meta.get('base_model') or 'N/A'}
- Model size (approx): {str(meta.get('params_b')) + 'B' if meta.get('params_b') else 'Unknown'}
- Downloads (30d): {meta.get('downloads_30d') or 0}
- Last modified: {meta.get('last_modified') or 'N/A'}
- License: {normalize_license(meta.get('license'))}
- Tags: {", ".join([t for t in meta.get('tags', [])]) or 'N/A'}

## Trust Zones (assumed / to refine)
- Internet → SaaS Vendor (Hugging Face) → Enterprise VPC
- Client (browser) ↔ App (Streamlit/API) ↔ Model Runtime (self-hosted or API)

## Data Classes
{(data_classes or 'N/A').strip()}

## External Connectors / Tools
{(connectors or 'None').strip()}

## Compliance Overlays
{(compliance or 'None').strip()}

## Operational Constraints
{(constraints or 'None').strip()}

## Output Requirements
- Use the STRIDE framework for threats.
- Include risk scoring with 5×5 matrix (likelihood 1–5 × impact 1–5) and band.
- Append a fenced ```json threat_register``` block as described in the contract.
"""
            session["__tm_user_prompt__"] = user_prompt
            st.success("STRIDE prompt built.")
            with st.expander("Preview: User Prompt sent to GPT"):
                st.code(user_prompt, language="markdown")
        except Exception as e:
            st.error(f"Could not build prompt: {e}")

    if run_tm and OPENAI_API_KEY:
        if "__tm_user_prompt__" not in session or "__last_meta__" not in session:
            st.warning("Click 'Build STRIDE Prompt (Preview)' first.")
        else:
            repo_id = extract_repo_id(user_input.strip())
            meta = session["__last_meta__"]
            user_prompt = session["__tm_user_prompt__"]
            system_prompt = STRIDE_SYSTEM_PROMPT
            try:
                with st.spinner("Calling GPT to generate your STRIDE threat model…"):
                    tm_md = call_openai(system_prompt, user_prompt, model_name=tm_model, temperature=tm_temp, max_tokens=tm_max_tokens)

                # Try to extract the JSON threat register
                tm_json_block = try_extract_json_block(tm_md)
                tm_register_json = None
                if tm_json_block:
                    try:
                        tm_register_json = json.loads(tm_json_block)
                    except Exception:
                        tm_register_json = None

                # Header stamp
                header = (
                    f"# STRIDE Threat Model – {repo_id}\n\n"
                    f"- Generated at (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                    f"- App version: {APP_VERSION}\n"
                    f"- Policy version: {policy_version}\n"
                    f"- OpenAI model: {tm_model}\n\n"
                )
                full_tm_md = header + tm_md

                safe_name = repo_id.replace("/", "__")
                st.success("Threat model generated.")
                st.download_button("⬇️ Download STRIDE Threat Model (Markdown)",
                                   data=full_tm_md.encode("utf-8"),
                                   file_name=f"{safe_name}_stride_threat_model.md",
                                   mime="text/markdown", use_container_width=True)

                if tm_register_json is not None:
                    st.download_button("⬇️ Download Threat Register (JSON)",
                                       data=json.dumps(tm_register_json, indent=2).encode("utf-8"),
                                       file_name=f"{safe_name}_threat_register.json",
                                       mime="application/json", use_container_width=True)
                else:
                    st.info("No embedded JSON register detected. The model output may not have included a fenced JSON block.")

                with st.expander("Preview: STRIDE Threat Model (Markdown)"):
                    st.code(full_tm_md, language="markdown")

                bundle = io.BytesIO()
                with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
                    if "__last_md__" in st.session_state:
                        z.writestr(f"{safe_name}.md", st.session_state["__last_md__"])
                    z.writestr(f"{safe_name}_stride_threat_model.md", full_tm_md)
                    if tm_register_json is not None:
                        z.writestr(f"{safe_name}_threat_register.json", json.dumps(tm_register_json, indent=2))
                    z.writestr("policy_version.txt", policy_version)
                    z.writestr("app_version.txt", APP_VERSION)
                st.download_button("⬇️ Download Bundle (ZIP)",
                                   data=bundle.getvalue(),
                                   file_name=f"{safe_name}_stride_bundle.zip",
                                   mime="application/zip", use_container_width=True)

            except Exception as e:
                st.error(f"Threat model generation failed: {e}")

# Footer
st.caption(
    "Tips: For private/gated Hugging Face repos, set an HF token via huggingface-cli login (locally) or secrets. "
    "For STRIDE Threat Modeling, set OPENAI_API_KEY. Policy can be edited via risk_policy.json."
)
