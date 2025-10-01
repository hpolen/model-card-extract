import json
from typing import Dict, Any

DEFAULT_POLICY: Dict[str, Any] = { ... }  # same as policies/risk_policy.json structure

def load_policy(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_POLICY

def normalize_license(lic: str) -> str:
    if not lic: return "unknown"
    s = lic.strip().lower().replace(" ", "-")
    aliases = {"apache2":"apache-2.0","apache-2":"apache-2.0","gpl":"gpl-3.0","cc-by-nc":"cc-by-nc-4.0"}
    return aliases.get(s, s)

def evaluate_dimension(name: str, meta: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    # same logic we added in the app (license, data_transparency, etc.)
    ...

def score_model(meta: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    # compute details + overall band
    ...
