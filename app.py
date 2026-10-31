"""
PhishGuard - Offline Phishing URL Detection Web Application
------------------------------------------------------------
A clean, unified Streamlit interface powered by the Domain-Aware Hybrid Engine:
- Lexical ML Model: L2-regularized Logistic Regression trained with Mini-Batch SGD
  on character 3-to-5-gram subword TF-IDF representations (models/experimental/model_char_ngram.json).
- Domain Reputation Layer: Deterministic, Public Suffix List (PSL)-aware Tranco Top-100k
  reputation scoring with tenant-boundary isolation and anti-evasion protections.

Privacy & Safety:
- 100% offline lexical and reputation analysis.
- Submitted URLs are NEVER visited, resolved via DNS, or contacted over the network.
- Zero runtime dependencies on NumPy, pandas, or scikit-learn.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure src/ is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import streamlit as st
except ImportError:
    print(
        "ERROR: Streamlit is not installed in the current Python environment.\n"
        "To run this app, install streamlit:\n"
        "    python -m pip install streamlit\n"
        "Then launch the app with:\n"
        "    python -m streamlit run app.py\n"
    )
    sys.exit(1)

from src.char_ngram_engine import CharNgramEngine
from src.domain_aware_hybrid_engine import DomainAwareHybridEngine
from src.phishguard_engine import PhishGuardEngine
from src.url_feature_extractor import extract_url_features


def _resolve_existing_path(*relative_candidates: str) -> str:
    """Resolve the first existing file path among relative candidates."""
    for rel in relative_candidates:
        full_path = os.path.join(BASE_DIR, rel)
        if os.path.isfile(full_path):
            return full_path
    return os.path.join(BASE_DIR, relative_candidates[0])


# Page configuration
st.set_page_config(
    page_title="PhishGuard - Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1. MODEL LOADERS (Cached in memory to prevent unnecessary reloads)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_hybrid_engine() -> DomainAwareHybridEngine:
    """Load the primary Domain-Aware Hybrid Engine.
    
    Combines character n-gram subword logistic regression with deterministic
    Tranco Top-100k domain reputation and PSL tenant-boundary isolation.
    """
    model_path = _resolve_existing_path(
        os.path.join("models", "experimental", "model_char_ngram.json"),
        os.path.join("models", "model_char_ngram.json"),
    )
    tranco_path = _resolve_existing_path(
        os.path.join("data", "reputation", "tranco_top100k.json"),
        os.path.join("data", "tranco_top100k.json"),
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model 4 artifact missing: {model_path}")
    if not os.path.isfile(tranco_path):
        raise FileNotFoundError(f"Tranco snapshot missing: {tranco_path}")

    return DomainAwareHybridEngine(
        char_model_path=model_path,
        tranco_snapshot_path=tranco_path,
        alpha=1.3,
        threshold=0.55,
        enable_http_bare_discount=True,
    )


@st.cache_resource
def load_url_engine() -> PhishGuardEngine:
    """Backward-compatible loader for the 19-feature Random Forest model."""
    model_path = _resolve_existing_path(
        os.path.join("models", "phishguard_url_trees.json"),
        os.path.join("models", "phishguard_trees.json"),
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Production URL model artifact missing: {model_path}")
    return PhishGuardEngine(model_path)


@st.cache_resource
def load_engine() -> PhishGuardEngine:
    """Backward-compatible loader for the 49-feature webpage DOM model."""
    model_path = _resolve_existing_path(
        os.path.join("models", "phishguard_trees.json"),
        os.path.join("models", "phishguard_url_trees.json"),
    )
    return PhishGuardEngine(model_path)


# Safely initialize the primary hybrid detector on startup
try:
    engine = load_hybrid_engine()
except Exception as exc:
    st.error(f"Failed to initialize PhishGuard Engine: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# 2. APPLICATION HEADER & DESCRIPTION
# -----------------------------------------------------------------------------
st.title("🛡️ PhishGuard - Phishing URL Detector")
st.markdown(
    """
    **Offline Phishing URL Detection** powered by subword pattern analysis and domain reputation.
    
    * **Lexical Model:** Evaluates character 3-to-5-gram subword patterns using an L2-regularized 
      Logistic Regression model trained with Mini-Batch SGD (`model_char_ngram.json`).
    * **Reputation Adjustment:** Adjusts the model's risk score using an offline **Tranco Top-100k** 
      research ranking list. *(Note: Domain reputation is a deterministic adjustment policy, not 
      Random Forest or a separate ML model.)*
    * **Safety Protections:** Public Suffix List (PSL)-aware tenant isolation blocks reputation 
      discounts on user-controlled subdomains (`godaddysites.com`, `pages.dev`, `firebaseapp.com`), 
      and anti-evasion gates restrict discounts on aggressive credential-harvesting lures.
    * **Privacy Guarantee:** 100% offline analysis. PhishGuard **never visits the URL or downloads 
      webpage content** over the network.
    """
)

st.write("---")

# -----------------------------------------------------------------------------
# 3. URL INPUT & ANALYSIS
# -----------------------------------------------------------------------------
user_url = st.text_input(
    "Enter a web URL to analyze:",
    placeholder="e.g. https://www.google.com, https://github.com/login, or http://metmask-wallet-io.godaddysites.com",
    help="Enter any complete URL or domain name. Analysis runs 100% offline without network requests.",
    key="phishguard_url_input",
)

analyze_button = st.button("🔍 Check URL", type="primary", use_container_width=True)

if analyze_button:
    stripped_url = user_url.strip() if user_url else ""
    if not stripped_url:
        st.warning("⚠️ Please enter a URL before clicking 'Check URL'.")
    else:
        try:
            # 1. Deterministic Hybrid Inference (Pure-Python offline evaluation)
            probs = engine.predict_proba_sample(stripped_url)
            prob_phish = float(probs[0])
            prob_legit = float(probs[1])
            pred_class = engine.predict_sample(stripped_url)

            # 2. Detailed explanation and telemetry
            exp_details = engine.explain_sample(stripped_url)
            rep_data = exp_details["reputation_telemetry"]
            host_data = rep_data["host_info"]

        except ValueError as val_err:
            st.error(f"⚠️ **Invalid URL Format:** {val_err}")
        except Exception as exc:
            st.error(f"❌ **Analysis Error:** Unexpected failure during analysis: {exc}")
        else:
            st.write("---")
            st.subheader("Analysis Results")

            # A. Verdict Banner
            if pred_class == 1:
                st.success("Verdict: **Likely Legitimate**")
            else:
                st.error("Verdict: **Likely Phishing**")

            # B. Probability Metrics Cards
            col_phish, col_legit = st.columns(2)
            with col_phish:
                st.metric(
                    "Estimated Phishing Probability",
                    f"{prob_phish * 100:.2f}%",
                    help="Confidence that this URL exhibits deceptive or malicious phishing patterns.",
                )
            with col_legit:
                st.metric(
                    "Estimated Legitimate Probability",
                    f"{prob_legit * 100:.2f}%",
                    help="Confidence that this URL exhibits authentic, standard web patterns.",
                )

            # C. Visual Risk Progress Indicator
            st.progress(prob_phish)
            st.caption(
                f"Risk Indicator: **{prob_phish * 100:.1f}%** phishing risk "
                f"(Decision threshold: **{engine.threshold * 100:.1f}%**). "
                f"URLs with P(Phishing) ≥ {engine.threshold * 100:.1f}% are classified as Phishing."
            )

            # D. Concise Explanation Callout
            raw_p = exp_details["raw_model4_prob_phishing"]
            if pred_class == 1 and raw_p >= engine.threshold:
                tranco_rank = host_data.get("tranco_rank")
                rank_str = (
                    f"Rank #{tranco_rank:,}"
                    if tranco_rank is not None
                    else "Not ranked in the Tranco list"
                )
                st.info(
                    "💡 **Reputation Adjustment Applied:** The subword machine learning model initially assigned an "
                    f"elevated risk ({raw_p * 100:.1f}%) due to lure-like keywords (e.g. `/login` or `.html`). "
                    f"However, verified Tranco Top-100k reputation for `{host_data['registered_domain']}` "
                    f"({rank_str}) safely adjusted the score to {prob_phish * 100:.1f}%, "
                    "mitigating a false alarm."
                )
            elif host_data["is_multi_tenant"]:
                st.warning(
                    f"🛡️ **Tenant Isolation Enforced:** This URL is hosted on a multi-tenant platform "
                    f"(`{host_data['registered_domain']}`). Customer subdomains on shared site-builders "
                    "do not inherit platform reputation, ensuring malicious user pages cannot evade detection."
                )
            elif host_data["is_shared_platform"]:
                st.warning(
                    f"⚠️ **Shared Gateway/Redirector:** `{host_data['registered_domain']}` is a shared file host "
                    "or URL redirector. Platform reputation is strictly disabled to prevent abuse."
                )
            elif host_data["is_ip"]:
                st.warning(
                    "⚠️ **Raw IP Address:** The URL uses a raw IP address instead of a domain name. "
                    "Reputation discount is not applicable."
                )
            elif pred_class == 0:
                st.warning(
                    "🚨 **Phishing Indicators Detected:** The URL contains high-risk lexical subword patterns "
                    "and either lacks authoritative domain reputation or triggered anti-evasion protections."
                )
            else:
                st.success(
                    "✅ **Standard Legitimate Pattern:** The URL exhibits authentic subword structure "
                    "consistent with legitimate web traffic."
                )

            # E. Domain Telemetry Expander
            with st.expander("🌐 Domain Reputation & Safety Rules", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"**Registered Domain:**\n`{host_data['registered_domain']}`")
                with c2:
                    r_val = host_data["tranco_rank"]
                    r_txt = f"#{r_val:,}" if r_val is not None else "Unranked"
                    st.markdown(f"**Tranco Rank:**\n`{r_txt}`")
                with c3:
                    if host_data["is_multi_tenant"]:
                        st.markdown("**Safety Policy:**\n`🛡️ Multi-Tenant Isolated (Zero Reputation)`")
                    elif host_data["is_shared_platform"]:
                        st.markdown("**Safety Policy:**\n`⚠️ Shared Gateway (Zero Reputation)`")
                    elif host_data["is_ip"]:
                        st.markdown("**Safety Policy:**\n`⚠️ Raw IP Address (Zero Reputation)`")
                    elif host_data.get("is_verified_host"):
                        st.markdown("**Safety Policy:**\n`✅ Verified Host Policy (Unranked in Tranco)`")
                    elif host_data["reputation_applicable"]:
                        st.markdown("**Safety Policy:**\n`✅ Verified Domain Reputation Applied`")
                    else:
                        st.markdown("**Safety Policy:**\n`ℹ️ Unranked Domain (Standard Scoring)`")
                with c4:
                    dz = rep_data["delta_z"]
                    st.markdown(f"**Log-Odds Adjustment:**\n`{dz:+.4f}`")

                if rep_data["adjustment_reasons"]:
                    st.caption(
                        "**Adjustments:** "
                        + ", ".join(f"`{r}`" for r in rep_data["adjustment_reasons"])
                    )
                else:
                    st.caption("**Adjustments:** None (Pure subword model scoring enforced).")

            # F. Subword Token Explainability Expander
            with st.expander("🔍 Subword Character Token Explainability (Model 4)"):
                st.markdown(
                    "Model 4 breaks the URL into 3-to-5 character subwords and scores their statistical association "
                    "with phishing versus legitimate web traffic:"
                )
                col_tok_phish, col_tok_legit = st.columns(2)
                with col_tok_phish:
                    st.write("**Top Phishing Subwords (increased risk):**")
                    if exp_details["top_phishing_subwords"]:
                        for tok, w in exp_details["top_phishing_subwords"]:
                            st.write(f"- `{tok}` : `+{w:.4f}`")
                    else:
                        st.caption("None detected.")
                with col_tok_legit:
                    st.write("**Top Legitimate Subwords (decreased risk):**")
                    if exp_details["top_legitimate_subwords"]:
                        for tok, w in exp_details["top_legitimate_subwords"]:
                            st.write(f"- `{tok}` : `{w:.4f}`")
                    else:
                        st.caption("None detected.")

            # G. Mandatory Security Disclaimer
            st.info(
                "ℹ️ **Security Disclaimer**: PhishGuard evaluates lexical subwords and domain reputation offline. "
                "Predictions are statistical estimates, not absolute guarantees of safety. "
                "The presence of HTTPS (`https://`) does not guarantee that a website is trustworthy, and popular "
                "domains do not guarantee that every URL hosted on them is safe. Never enter sensitive credentials "
                "on unfamiliar or unverified websites."
            )
