"""
PhishGuard - Local Streamlit Demo
---------------------------------
Demonstrates the pure-Python PhishGuardRandomForest engine using the
49 extracted numerical webpage features.

IMPORTANT NOTICE:
Demo uses manually entered 49-feature values. It does not analyze raw URLs.
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
        "To run this app once permitted, install streamlit:\n"
        "    python -m pip install streamlit\n"
        "Then launch the app with:\n"
        "    streamlit run app.py\n"
    )
    sys.exit(1)

from src.phishguard_engine import PhishGuardEngine


# Page configuration
st.set_page_config(
    page_title="PhishGuard Demo",
    page_icon="🛡️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1. LOAD THE ENGINE & BENCHMARK PRESETS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine() -> PhishGuardEngine:
    model_path = os.path.join(BASE_DIR, "models", "phishguard_trees.json")
    return PhishGuardEngine(model_path)


@st.cache_data
def load_benchmark_samples():
    bench_path = os.path.join(BASE_DIR, "models", "phishguard_benchmark.json")
    if os.path.isfile(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f).get("samples", [])
    return []


try:
    engine = load_engine()
    benchmark_samples = load_benchmark_samples()
except Exception as exc:
    st.error(f"Failed to initialize PhishGuard Engine: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# 2. HEADER & TRANSPARENCY NOTICE
# -----------------------------------------------------------------------------
st.title("🛡️ PhishGuard ML Demo")
st.warning(
    "⚠️ **Demo Notice:** Demo uses manually entered 49-feature values. "
    "It does not analyze raw URLs."
)

st.markdown(
    """
    This interface runs the **100-tree Random Forest** classifier using our offline,
    pure-Python inference engine without external C-extensions.
    
    * **Target Classes:** `0` = **Phishing** | `1` = **Legitimate**
    * **Required Inputs:** Exactly 49 numerical webpage and URL-metric features.
    """
)

# -----------------------------------------------------------------------------
# 3. PRESET SELECTION (For quick testing)
# -----------------------------------------------------------------------------
st.subheader("1. Select Demo Preset or Enter Custom Values")

preset_options = ["Neutral / Default Values"]
if len(benchmark_samples) >= 10:
    preset_options.append("Benchmark Sample #0 (Verified Legitimate)")
    preset_options.append("Benchmark Sample #5 (Verified Phishing)")

selected_preset = st.selectbox(
    "Choose a sample to populate the 49 inputs below for demonstration:",
    preset_options,
    help="These presets populate the 49 fields with verified reference data for easy demonstration.",
)

# Determine default values based on selected preset
default_values = {}
if selected_preset == "Benchmark Sample #0 (Verified Legitimate)" and len(benchmark_samples) > 0:
    default_values = benchmark_samples[0]["features_dict"]
elif selected_preset == "Benchmark Sample #5 (Verified Phishing)" and len(benchmark_samples) > 5:
    default_values = benchmark_samples[5]["features_dict"]
else:
    # Sensible neutral defaults for all 49 features
    for feat in engine.feature_names:
        default_values[feat] = 0.0

# -----------------------------------------------------------------------------
# 4. INPUT FORM (49 FEATURES)
# -----------------------------------------------------------------------------
st.subheader("2. 49 Numerical Webpage & URL Features")
st.caption(
    "All fields are demo numerical inputs. Adjust any value to test the decision boundaries."
)

with st.form("phishguard_form"):
    user_inputs = {}

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔗 URL Lexical (18)",
        "📊 Corpus & Statistics (3)",
        "📄 Page Structure & DOM (18)",
        "🔐 Content & Keywords (10)",
    ])

    # Tab 1: URL Lexical features
    with tab1:
        lexical_features = [
            "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
            "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
            "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
            "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
            "SpacialCharRatioInURL", "LongURL"
        ]
        cols = st.columns(3)
        for i, feat in enumerate(lexical_features):
            with cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    label=feat,
                    value=float(default_values.get(feat, 0.0)),
                    step=1.0 if "NoOf" in feat or "Length" in feat or feat in ("IsDomainIP", "HasObfuscation", "LongURL") else 0.01,
                    key=f"input_{feat}",
                )

    # Tab 2: Corpus & Statistical Probabilities
    with tab2:
        stat_features = ["CharContinuationRate", "TLDLegitimateProb", "URLCharProb"]
        cols = st.columns(3)
        for i, feat in enumerate(stat_features):
            with cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    label=feat,
                    value=float(default_values.get(feat, 0.0)),
                    step=0.01,
                    key=f"input_{feat}",
                )

    # Tab 3: Page Structure & DOM features
    with tab3:
        dom_features = [
            "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
            "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
            "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
            "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
            "HasHiddenFields", "HasPasswordField"
        ]
        cols = st.columns(3)
        for i, feat in enumerate(dom_features):
            with cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    label=feat,
                    value=float(default_values.get(feat, 0.0)),
                    step=1.0 if "Line" in feat or "Redirect" in feat or "NoOf" in feat or feat.startswith("Has") or feat in ("Robots", "IsResponsive") else 0.01,
                    key=f"input_{feat}",
                )

    # Tab 4: Content & Keywords
    with tab4:
        content_features = [
            "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage",
            "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef"
        ]
        cols = st.columns(3)
        for i, feat in enumerate(content_features):
            with cols[i % 3]:
                user_inputs[feat] = st.number_input(
                    label=feat,
                    value=float(default_values.get(feat, 0.0)),
                    step=1.0 if "NoOf" in feat or feat in ("Bank", "Pay", "Crypto", "HasCopyrightInfo") else 0.01,
                    key=f"input_{feat}",
                )

    submitted = st.form_submit_button("🔍 Run PhishGuard Classification", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. PREDICTION & RESULTS DISPLAY
# -----------------------------------------------------------------------------
if submitted:
    st.subheader("3. Model Prediction Results")
    try:
        # Validate and predict using engine interface
        prediction = engine.predict_sample(user_inputs)
        probabilities = engine.predict_proba_sample(user_inputs)

        prob_phishing = probabilities[0]
        prob_legitimate = probabilities[1]

        # Display result badge
        col_res, col_prob0, col_prob1 = st.columns(3)

        with col_res:
            if prediction == 1:
                st.success("### Result: **Legitimate** (Safe)")
            else:
                st.error("### Result: **Phishing** (Malicious)")

        with col_prob0:
            st.metric(
                label="Probability Phishing (Class 0)",
                value=f"{prob_phishing * 100:.2f}%",
            )
            st.progress(prob_phishing)

        with col_prob1:
            st.metric(
                label="Probability Legitimate (Class 1)",
                value=f"{prob_legitimate * 100:.2f}%",
            )
            st.progress(prob_legitimate)

        # Informative summary
        st.info(
            f"**Decision Summary:** The model ensemble classified this 49-feature vector as "
            f"**{'Legitimate' if prediction == 1 else 'Phishing'}** based on an average "
            f"forest consensus of {prob_legitimate * 100:.1f}% confidence for legitimate and "
            f"{prob_phishing * 100:.1f}% confidence for phishing."
        )

    except Exception as err:
        st.error(f"Validation or Prediction Error: {err}")
