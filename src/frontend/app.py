"""NeuroBridge Enterprise — Streamlit B2B dashboard.

Three tabs (Molecule / Signal / Image), each fires a POST request against the
sibling FastAPI service and renders a result card with row counts, runtime,
and a deep link to the corresponding MLflow run.

Design: Trust & Authority — navy + sky CTA + cool-white background, Plus
Jakarta Sans, generous whitespace. Avoids emoji icons, AI gradients, and
playful flourishes (per design-system guidance for clinical-ML B2B).

Launch: `streamlit run src/frontend/app.py`
"""
from __future__ import annotations

import html as _html
import os

import httpx
import streamlit as st


_API_URL = os.environ.get("NEUROBRIDGE_API_URL", "http://localhost:8000")
_MLFLOW_URL = os.environ.get(
    "NEUROBRIDGE_MLFLOW_URL",
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
)
_MLFLOW_DISABLED = os.environ.get("NEUROBRIDGE_DISABLE_MLFLOW") == "1"


# Trust & Authority custom CSS — overrides Streamlit defaults to lock the
# design-system tokens. Loaded once at app start via st.markdown.
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stTabs, .stButton, .stTextInput {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background-color: #F8FAFC;
}

/* Brand header band */
.brand-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #F8FAFC;
    padding: 1.75rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #1E293B;
}
.brand-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #FFFFFF;
}
.brand-header p {
    color: #94A3B8;
    margin: 0.25rem 0 0 0;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Status pills */
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 0.5rem;
}
.status-ok    { background: #DCFCE7; color: #166534; }
.status-warn  { background: #FEF3C7; color: #92400E; }
.status-down  { background: #FEE2E2; color: #991B1B; }

/* Cards / metric containers */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
[data-testid="stMetricLabel"] {
    color: #64748B;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #0F172A;
    font-weight: 700;
    font-size: 1.85rem;
}

/* Primary action button */
.stButton > button[kind="primary"] {
    background: #0369A1;
    color: #FFFFFF;
    border: 0;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: background 180ms ease, transform 120ms ease;
}
.stButton > button[kind="primary"]:hover {
    background: #075985;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"]:focus {
    outline: 3px solid rgba(3, 105, 161, 0.35);
    outline-offset: 2px;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B;
    font-weight: 600;
    padding: 0.65rem 1.25rem;
    border-bottom: 2px solid transparent;
    transition: color 150ms ease, border-color 150ms ease;
}
.stTabs [aria-selected="true"] {
    color: #0F172A !important;
    border-bottom-color: #0369A1 !important;
}

/* Section headers inside tabs */
.section-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    color: #0369A1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0.15rem 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.section-desc {
    color: #475569;
    font-size: 0.95rem;
    margin: 0 0 1.5rem 0;
    line-height: 1.6;
}

/* Result card link styling */
.mlflow-link a {
    color: #0369A1;
    text-decoration: none;
    font-weight: 600;
    border-bottom: 1px solid rgba(3, 105, 161, 0.25);
    transition: border-color 150ms ease;
}
.mlflow-link a:hover {
    border-bottom-color: #0369A1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
"""


def _check_api_health() -> tuple[bool, str]:
    """Ping FastAPI /health endpoint; return (ok, status_text)."""
    try:
        resp = httpx.get(f"{_API_URL}/health", timeout=2.0)
        if resp.status_code == 200:
            return True, "ok"
        return False, f"http {resp.status_code}"
    except httpx.RequestError as e:
        return False, str(type(e).__name__)


def _post(endpoint: str, payload: dict) -> dict:
    """POST to the FastAPI surface; let httpx raise on non-2xx."""
    resp = httpx.post(f"{_API_URL}{endpoint}", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def _render_brand_header() -> None:
    st.markdown(
        """
        <div class="brand-header">
            <h1>NeuroBridge Enterprise</h1>
            <p>Three-modality clinical ML — Data Drift, Missing Modalities, Artifacts</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section(eyebrow: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <p class="section-eyebrow">{eyebrow}</p>
        <h2 class="section-title">{title}</h2>
        <p class="section-desc">{desc}</p>
        """,
        unsafe_allow_html=True,
    )


def _render_result(body: dict) -> None:
    """Render a 3-metric result card + MLflow deep link."""
    cols = st.columns(3)
    cols[0].metric("Rows", f"{body['rows']:,}")
    cols[1].metric("Columns", f"{body['columns']:,}")
    cols[2].metric("Runtime", f"{body['duration_sec']:.2f} s")

    safe_output_path = _html.escape(str(body["output_path"]))
    st.markdown(
        f"<p style='color:#475569;margin:1rem 0 0.5rem 0;font-size:0.9rem;'>"
        f"Output written to <code style='background:#E8ECF1;padding:2px 6px;border-radius:4px;'>"
        f"{safe_output_path}</code></p>",
        unsafe_allow_html=True,
    )

    run_id = body.get("mlflow_run_id")
    if run_id and not _MLFLOW_DISABLED:
        safe_run_id = _html.escape(str(run_id))
        safe_url = _html.escape(_MLFLOW_URL, quote=True)
        st.markdown(
            f"<p class='mlflow-link'>MLflow run: "
            f"<a href='{safe_url}/#/experiments/0/runs/{safe_run_id}' "
            f"target='_blank' rel='noopener noreferrer'>{safe_run_id[:12]}…</a></p>",
            unsafe_allow_html=True,
        )
    elif _MLFLOW_DISABLED:
        st.markdown(
            "<p style='color:#92400E;font-size:0.85rem;'>"
            "MLflow tracking is disabled (NEUROBRIDGE_DISABLE_MLFLOW=1).</p>",
            unsafe_allow_html=True,
        )


def _render_sidebar(api_ok: bool, api_status: str) -> None:
    with st.sidebar:
        st.markdown("### System Status")
        safe_api_status = _html.escape(api_status)
        api_pill = (
            f"<span class='status-pill status-ok'>API · {safe_api_status}</span>"
            if api_ok
            else f"<span class='status-pill status-down'>API · {safe_api_status}</span>"
        )
        mlflow_pill = (
            "<span class='status-pill status-warn'>MLflow · disabled</span>"
            if _MLFLOW_DISABLED
            else "<span class='status-pill status-ok'>MLflow · tracking</span>"
        )
        st.markdown(api_pill + mlflow_pill, unsafe_allow_html=True)

        st.markdown("### Endpoints")
        st.markdown(
            f"<p style='font-size:0.8rem;color:#475569;line-height:1.7;'>"
            f"FastAPI · <code>{_API_URL}</code><br/>"
            f"MLflow · <code>{_MLFLOW_URL}</code></p>",
            unsafe_allow_html=True,
        )

        st.markdown("### About")
        st.markdown(
            "<p style='font-size:0.85rem;color:#475569;line-height:1.6;'>"
            "Solving Data Drift, Missing Modalities, and Artifacts in clinical "
            "biosignal pipelines. Three production modalities behind one FastAPI "
            "surface, all runs tracked to MLflow.</p>",
            unsafe_allow_html=True,
        )


def _render_bbb_tab() -> None:
    _render_section(
        "MOLECULE — BBBP",
        "Blood-Brain-Barrier permeability decision",
        "Enter a SMILES string. The system computes a 2,048-bit Morgan "
        "fingerprint, runs it through a trained Random Forest classifier, "
        "and returns the predicted permeability label, the model's "
        "self-rated confidence, and the top SHAP feature attributions "
        "explaining the decision.",
    )

    EDGE_CASES = {
        "Custom input (default)": {
            "smiles": "CCO",
            "label": "Ethanol — small, drug-like, BBB-permeable",
            "expectation": "High confidence, label = permeable",
        },
        "Invalid SMILES (parse-error path)": {
            "smiles": "this_is_not_a_valid_molecule_at_all_!!",
            "label": "Garbage string — should not parse",
            "expectation": "API returns HTTP 400 with parse error; UI shows recoverable warning",
        },
        "Empty string (boundary)": {
            "smiles": "",
            "label": "Empty input — boundary condition",
            "expectation": "Pydantic accepts empty; API returns 400 (RDKit cannot parse)",
        },
        "Massive OOD: cyclosporine-like macrocycle": {
            "smiles": (
                "CC[C@H](C)[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)"
                "[C@@H]2CCCN2C(=O)[C@H](C(C)C)NC(=O)[C@H]([C@@H](C)CC)N(C)C(=O)"
                "[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)"
                "[C@H](CC(C)C)N(C)C(=O)CN(C)C1=O)C(C)C"
            ),
            "label": "Cyclosporine — 11-residue macrocycle (~1.2 kDa)",
            "expectation": (
                "Far outside training distribution; model should hedge "
                "with low confidence (well-calibrated systems don't "
                "pretend to know)."
            ),
        },
        "OOD: heavy halogenated aromatic": {
            "smiles": "Fc1c(F)c(F)c(c(F)c1F)c2c(F)c(F)c(F)c(F)c2F",
            "label": "Decafluorobiphenyl — extreme halogen density",
            "expectation": "Rare scaffold; expect lowered confidence vs ethanol",
        },
    }

    case_name = st.selectbox(
        "Test Edge Cases",
        options=list(EDGE_CASES.keys()),
        index=0,
        key="bbb_case",
        help=(
            "Pick a robustness probe. Each case demonstrates how the "
            "system handles a real-world failure mode — invalid input, "
            "out-of-distribution molecules, or boundary conditions."
        ),
    )
    case = EDGE_CASES[case_name]
    st.caption(f"**Probe:** {case['label']}  ·  **Expected:** {case['expectation']}")

    smiles = st.text_input(
        "SMILES string",
        value=case["smiles"],
        key="bbb_smiles",
        help="Examples: CCO (ethanol), CC(=O)Nc1ccc(O)cc1 (paracetamol)",
    )
    top_k = st.slider(
        "SHAP features to display", min_value=3, max_value=10, value=5, key="bbb_topk",
    )

    if st.button("Predict BBB permeability", type="primary", key="bbb_predict"):
        with st.spinner("Computing fingerprint, predicting, and explaining…"):
            try:
                result = _post("/predict/bbb", {"smiles": smiles, "top_k": top_k})
                _render_prediction_card(result)
                st.toast("Prediction complete", icon="✅")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    st.error(
                        "Model artifact not loaded yet. Run "
                        "`python -m src.models.bbb_model` to train it, "
                        "then retry."
                    )
                elif e.response.status_code == 400:
                    # Robustness story: show the WARNING instead of an ERROR
                    # — invalid input is a recoverable path, not a crash.
                    st.warning(
                        f"Robustness check passed: API rejected the input "
                        f"with HTTP 400 (no crash). Detail: "
                        f"{e.response.json().get('detail', e.response.text)}"
                    )
                else:
                    st.error(
                        f"Prediction failed (HTTP {e.response.status_code}): "
                        f"{e.response.text}"
                    )
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")


def _render_eeg_tab() -> None:
    _render_section(
        "SIGNAL — EEG",
        "Electroencephalogram artifact removal",
        "Bandpass-filters raw FIF/EDF recordings, removes EOG artifacts via "
        "ICA decomposition, and extracts per-band PSD + statistical features "
        "across fixed-duration epochs.",
    )
    eeg_in = st.text_input("Input FIF/EDF path", "data/raw/eeg.fif", key="eeg_in")
    eeg_out = st.text_input("Output Parquet path", "data/processed/eeg_features.parquet", key="eeg_out")
    if st.button("Run EEG pipeline", type="primary", key="eeg_run"):
        with st.spinner("Filtering and running ICA…"):
            try:
                _render_result(_post("/pipeline/eeg", {
                    "input_path": eeg_in, "output_path": eeg_out,
                }))
                st.toast("EEG pipeline complete", icon="✅")
            except httpx.HTTPStatusError as e:
                st.error(f"Pipeline failed (HTTP {e.response.status_code}): {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")


def _render_mri_tab() -> None:
    _render_section(
        "IMAGE — MRI",
        "Multi-site harmonization via ComBat",
        "Loads NIfTI volumes, masks brain tissue, computes per-ROI summary "
        "statistics, then harmonizes across acquisition sites with neuroHarmonize "
        "to remove scanner-driven domain shift. The diagnostic plot below "
        "compares per-site feature distributions before and after harmonization."
    )
    mri_dir = st.text_input(
        "Input NIfTI directory", "tests/fixtures/mri_sample", key="mri_dir",
        help="Path to a directory of .nii(.gz) files + sites.csv",
    )
    sites_csv = st.text_input(
        "Sites CSV", "tests/fixtures/mri_sample/sites.csv", key="mri_sites",
    )

    if st.button("Run ComBat diagnostics", type="primary", key="mri_diag"):
        with st.spinner("Running pre + post ComBat (×2 the work)…"):
            try:
                result = _post(
                    "/pipeline/mri/diagnostics",
                    {"input_dir": mri_dir, "sites_csv": sites_csv},
                )
                _render_combat_diagnostics(result)
                st.toast("Diagnostics complete", icon="✅")
            except httpx.HTTPStatusError as e:
                st.error(
                    f"Diagnostics failed (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                )
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")


def _render_prediction_card(result: dict) -> None:
    """Render a B2B-styled decision card: label badge + confidence + SHAP bars."""
    st.session_state["last_bbb_prediction"] = result
    provenance = result.get("provenance")
    if provenance is not None:
        run_id = provenance.get("mlflow_run_id")
        run_label = run_id[:8] if run_id else "—"
        train_date = provenance.get("train_date") or "—"
        n_examples = provenance.get("n_examples")
        n_label = f"n={n_examples}" if n_examples else "n=—"
        st.caption(
            f"🔎 MLflow run **{run_label}** · "
            f"Model **{provenance.get('model_version', 'v1')}** · "
            f"trained {train_date} · {n_label}"
        )
    label_text = _html.escape(str(result["label_text"]))
    badge_color = "#166534" if result["label"] == 1 else "#991B1B"
    badge_bg    = "#DCFCE7" if result["label"] == 1 else "#FEE2E2"
    confidence_pct = result["confidence"] * 100

    st.markdown(
        f"""
        <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;
                    padding:1.5rem;margin:1rem 0;box-shadow:0 1px 2px rgba(15,23,42,0.04);'>
            <p style='font-size:0.72rem;font-weight:700;color:#64748B;
                      letter-spacing:0.08em;text-transform:uppercase;margin:0;'>Prediction</p>
            <div style='display:flex;align-items:center;gap:0.75rem;margin-top:0.4rem;'>
                <span style='background:{badge_bg};color:{badge_color};
                             padding:0.4rem 0.9rem;border-radius:999px;
                             font-size:1rem;font-weight:700;letter-spacing:0.01em;'>
                    {label_text.upper()}
                </span>
                <span style='color:#475569;font-size:0.95rem;'>
                    Model confidence: <strong style='color:#0F172A;'>{confidence_pct:.1f}%</strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.markdown(
        "<p style='font-size:0.72rem;font-weight:700;color:#64748B;"
        "letter-spacing:0.08em;text-transform:uppercase;margin:1rem 0 0.4rem 0;'>"
        "Confidence</p>",
        unsafe_allow_html=True,
    )
    st.progress(float(result["confidence"]))

    # Trust caption — precision-at-confidence from held-out 20% test split.
    # Silent skip when the API response has no calibration field (legacy models).
    calibration = result.get("calibration")
    if calibration is not None:
        threshold_pct = round(calibration["threshold"] * 100)
        precision_pct = round(calibration["precision"] * 100)
        support = calibration["support"]
        if support == 0:
            st.caption(
                "📊 Bu güven aralığında held-out test örneği yok — "
                "kalibrasyon bilgisi mevcut değil."
            )
        else:
            st.caption(
                f"📊 Test set'te ≥{threshold_pct}% güven üreten tahminlerin "
                f"precision'ı **{precision_pct}%** (n={support})."
            )

    drift_z = result.get("drift_z")
    rolling_n = result.get("rolling_n", 0)
    if drift_z is None and rolling_n < 10:
        st.caption(
            f"📈 Drift: warming up ({rolling_n}/10 predictions buffered)."
        )
    elif drift_z is None:
        st.caption(
            "📈 Drift: unavailable (model lacks train-time confidence stats)."
        )
    else:
        # Sign + magnitude: |z| < 1 in-band, 1–2 mild, >=2 significant.
        if abs(drift_z) < 1.0:
            tag = "within expected range"
        elif abs(drift_z) < 2.0:
            tag = "mild distribution shift"
        else:
            tag = "significant shift — retrain recommended"
        st.caption(
            f"📈 Drift: trailing-{rolling_n} confidence median is "
            f"**{drift_z:+.2f}σ** from train-time distribution ({tag})."
        )

    # SHAP attributions chart
    n_features = len(result["top_features"])
    st.markdown(
        f"<p style='font-size:0.72rem;font-weight:700;color:#64748B;"
        f"letter-spacing:0.08em;text-transform:uppercase;margin:1.5rem 0 0.4rem 0;'>"
        f"Top {n_features} SHAP attributions</p>",
        unsafe_allow_html=True,
    )
    import pandas as pd
    shap_df = pd.DataFrame(result["top_features"]).set_index("feature")
    st.bar_chart(shap_df, height=240, color="#0369A1")

    st.caption(
        "Positive SHAP values pushed the model toward the predicted class; "
        "negative values pushed it away. Feature names are 2,048-bit Morgan "
        "fingerprint indices (`fp_<bit>`)."
    )


def _render_combat_diagnostics(result: dict) -> None:
    """Render the Pre/Post-ComBat KDE comparison + site-gap KPI strip."""
    import altair as alt
    import pandas as pd

    rows = result.get("rows", [])
    if not rows:
        st.info(
            "No data returned. Check that the input directory contains "
            ".nii(.gz) files and a sites.csv with subject_id/site columns."
        )
        return

    cols = st.columns(3)
    cols[0].metric("Site-gap (Pre-ComBat)", f"{result['site_gap_pre']:.4f}")
    cols[1].metric("Site-gap (Post-ComBat)", f"{result['site_gap_post']:.4f}")
    cols[2].metric(
        "Reduction factor",
        f"{result['reduction_factor']:.0f}×",
        help=(
            "Pre-gap / Post-gap. A 100× reduction means ComBat "
            "removed two orders of magnitude of site-driven domain shift."
        ),
    )

    df = pd.DataFrame(rows)
    # Pin the chart to the first feature (most recognizable for the audience).
    feat = df["feature"].iloc[0]
    feat_df = df[df["feature"] == feat]

    # Layered KDE: x = feature_value, color = site, faceted by harmonization_state.
    chart = (
        alt.Chart(feat_df)
        .transform_density(
            density="feature_value",
            groupby=["site", "harmonization_state"],
            as_=["feature_value", "density"],
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("feature_value:Q", title=f"{feat} (intensity)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(
                "site:N",
                title="Site",
                scale=alt.Scale(scheme="tableau10"),
            ),
            tooltip=[
                alt.Tooltip("site:N"),
                alt.Tooltip("feature_value:Q", format=".4f"),
                alt.Tooltip("density:Q", format=".3f"),
            ],
        )
        .properties(width=380, height=260)
        .facet(
            column=alt.Column(
                "harmonization_state:N",
                title=None,
                sort=["Pre-ComBat", "Post-ComBat"],
                header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
            )
        )
        .resolve_scale(x="shared", y="shared")
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"Per-site density of `{feat}` before and after ComBat. Each "
        f"colored region is one acquisition site. **Convergence of the "
        f"colored regions in the Post-ComBat panel is the visual proof "
        f"of harmonization** — the same property the {result['reduction_factor']:.0f}× "
        f"site-gap reduction quantifies."
    )


def _render_ai_assistant_tab() -> None:
    """Day-7 T3C: chat-style explainer for the most recent BBB prediction."""
    _render_section(
        "AI Assistant",
        "Natural-language rationale (LLM or deterministic template)",
        "Pulls the most recent BBB prediction from this session and asks "
        "the explainer to justify it. Falls back to a deterministic, "
        "auditable template when no LLM is configured."
    )

    last = st.session_state.get("last_bbb_prediction")
    if last is None:
        st.info(
            "Run a BBB prediction first (BBB tab → Predict button), "
            "then come back here to ask the assistant about it."
        )
        return

    # Snapshot card so the user knows which prediction is being explained
    st.caption(
        f"Latest prediction: **{last['label_text']}** "
        f"({float(last['confidence']) * 100:.0f}% confident)  ·  "
        f"Top SHAP: {', '.join(f['feature'] for f in last.get('top_features', [])[:3])}"
    )

    PRESETS = [
        "Why was this molecule predicted as permeable?",
        "Which features pushed the verdict the most?",
        "Is this prediction trustworthy given the drift signal?",
    ]
    preset = st.selectbox("Preset question", options=PRESETS, key="ai_preset")
    custom = st.text_input(
        "Or type your own question (optional)",
        value="",
        key="ai_custom",
        help="Custom questions only affect the LLM path; the template gives a generic SHAP-driven rationale either way.",
    )
    question = custom.strip() or preset

    if st.button("Ask the AI Assistant", type="primary", key="ai_ask"):
        with st.spinner("Composing rationale…"):
            try:
                body = {
                    "smiles": last.get("smiles", ""),
                    "label": last["label"],
                    "label_text": last["label_text"],
                    "confidence": last["confidence"],
                    "top_features": last.get("top_features", []),
                    "calibration": last.get("calibration"),
                    "drift_z": last.get("drift_z"),
                    "user_question": question,
                }
                # The /predict/bbb response payload doesn't include the
                # user-supplied SMILES (only label/confidence/etc.), so
                # pull it from the input widget for paper-trail accuracy.
                # Streamlit text inputs persist via st.session_state.
                if not body["smiles"]:
                    body["smiles"] = st.session_state.get("bbb_smiles", "")
                resp = _post("/explain/bbb", body)
            except httpx.HTTPStatusError as e:
                st.error(
                    f"Explainer failed (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                )
                return
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
                return

        history = st.session_state.setdefault("explain_history", [])
        history.insert(0, (question, resp))

    # Render history (most recent first)
    history = st.session_state.get("explain_history", [])
    if history:
        st.markdown("### Conversation")
        for q, r in history[:10]:  # cap at 10 most recent
            with st.container():
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {r['rationale']}")
                source = r.get("source", "?")
                model = r.get("model") or "—"
                st.caption(f"Source: `{source}`  ·  Model: `{model}`")
                st.divider()


def main() -> None:
    """Streamlit entrypoint. Idempotent — Streamlit re-runs on every interaction."""
    st.set_page_config(
        page_title="NeuroBridge Enterprise",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    api_ok, api_status = _check_api_health()
    _render_brand_header()
    _render_sidebar(api_ok, api_status)

    if not api_ok:
        st.warning(
            f"⚠️ FastAPI surface is not reachable at `{_API_URL}` ({api_status}). "
            "Pipeline runs will fail until the API service is up. "
            "Run `uvicorn src.api.main:app --port 8000` or `docker compose up`."
        )

    bbb_tab, eeg_tab, mri_tab, assistant_tab = st.tabs([
        "Molecule (BBB)",
        "Signal (EEG)",
        "Image (MRI)",
        "AI Assistant",
    ])

    with bbb_tab:
        _render_bbb_tab()
    with eeg_tab:
        _render_eeg_tab()
    with mri_tab:
        _render_mri_tab()
    with assistant_tab:
        _render_ai_assistant_tab()


if __name__ == "__main__":
    main()
