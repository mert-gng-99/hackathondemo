"""NeuroBridge Enterprise — Streamlit B2B dashboard (Editorial redesign).

Five tabs (Molecule / Signal / Image / AI Assistant / Experiments) sitting on
top of one FastAPI surface. Every interaction returns an auditable decision
artefact: label + confidence + calibration + drift + provenance + SHAP.

Visual language (post-redesign):
- Dark theme = editorial Netflix-style — deep neutral grays + sand accent
- Light theme = warm paper + charcoal type — Apple HIG / NYT-Cooking energy
- Single sand brand-mark across both themes (#D2C4B1)
- Inter (display + body) + JetBrains Mono (data / code)

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
_LLM_DISABLED = os.environ.get("NEUROBRIDGE_DISABLE_LLM") == "1"


# --------------------------------------------------------------------------- #
# Design tokens — single source of truth for both themes.                     #
# Tokens are exposed as CSS custom properties at the :root level; every       #
# component reads from them so a theme swap is just a value swap.             #
# --------------------------------------------------------------------------- #

_TOKENS_DARK = {
    # Surfaces (deepest → most elevated)
    "bg-base":       "#0e0e10",
    "bg-elevated":   "#161618",
    "bg-elevated-2": "#1e1e21",
    "bg-elevated-3": "#2a2a2e",
    # Brand accent
    "accent":        "#D2C4B1",
    "accent-strong": "#E8DCC6",
    "accent-soft":   "rgba(210, 196, 177, 0.12)",
    "accent-ring":   "rgba(210, 196, 177, 0.35)",
    # Text
    "text-primary":   "#F5F2ED",
    "text-secondary": "#A8A29A",
    "text-tertiary":  "#6B6660",
    "text-on-accent": "#161618",
    # Lines
    "border":        "#2a2a2e",
    "border-strong": "#3a3a3e",
    # Semantic (keep cool — never red/green dominant in editorial)
    "success":       "#7FB069",
    "warning":       "#E0B469",
    "danger":        "#D97A6C",
    # Effects
    "shadow-sm": "0 1px 2px rgba(0, 0, 0, 0.4)",
    "shadow-md": "0 8px 24px rgba(0, 0, 0, 0.45)",
    "shadow-lg": "0 16px 48px rgba(0, 0, 0, 0.55)",
}

_TOKENS_LIGHT = {
    "bg-base":       "#FAF7F2",
    "bg-elevated":   "#FFFFFF",
    "bg-elevated-2": "#F5F0E8",
    "bg-elevated-3": "#EDE5D5",
    "accent":        "#1e1e21",
    "accent-strong": "#0e0e10",
    "accent-soft":   "rgba(30, 30, 33, 0.06)",
    "accent-ring":   "rgba(30, 30, 33, 0.18)",
    "text-primary":   "#161618",
    "text-secondary": "#4A4540",
    "text-tertiary":  "#8A857E",
    "text-on-accent": "#FAF7F2",
    "border":        "#E5DDC9",
    "border-strong": "#D2C4B1",
    "success":       "#3F7D45",
    "warning":       "#A06D1F",
    "danger":        "#A1483D",
    "shadow-sm": "0 1px 2px rgba(40, 30, 20, 0.04)",
    "shadow-md": "0 4px 16px rgba(40, 30, 20, 0.08)",
    "shadow-lg": "0 12px 40px rgba(40, 30, 20, 0.12)",
}


def _build_css(theme: str) -> str:
    """Return the full <style> block for the active theme.

    All tokens are emitted as CSS variables so the rest of the stylesheet
    is theme-agnostic. Re-runs cheaply since Streamlit caches markdown.
    """
    tokens = _TOKENS_DARK if theme == "dark" else _TOKENS_LIGHT
    css_vars = "\n".join(f"        --ng-{k}: {v};" for k, v in tokens.items())

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
{css_vars}
    --ng-radius-sm: 8px;
    --ng-radius-md: 12px;
    --ng-radius-lg: 16px;
    --ng-radius-xl: 24px;
    --ng-font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --ng-font-mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
    /* Motion tokens — Apple HIG fluid-physics + Material standard mix */
    --ng-ease-out:      cubic-bezier(0.16, 1, 0.3, 1);          /* expo-out, hero entrances */
    --ng-ease-spring:   cubic-bezier(0.34, 1.56, 0.64, 1);      /* gentle overshoot, verdict reveal */
    --ng-ease-standard: cubic-bezier(0.4, 0, 0.2, 1);           /* MD standard, micro-interactions */
    --ng-ease-decel:    cubic-bezier(0, 0, 0.2, 1);             /* enter from off-screen */
    --ng-ease-accel:    cubic-bezier(0.4, 0, 1, 1);             /* exit to off-screen */
    --ng-dur-instant:   80ms;
    --ng-dur-fast:      180ms;
    --ng-dur-base:      240ms;
    --ng-dur-slow:      360ms;
    --ng-dur-hero:      640ms;
}}

html {{
    scroll-behavior: smooth;
}}

/* --- Global typography + canvas ----------------------------------------- */

html, body, [class*="css"], .stApp, .stMarkdown, .stTabs, .stButton,
.stTextInput, .stSelectbox, .stSlider, .stDataFrame, .stMetric, .stExpander {{
    font-family: var(--ng-font-sans) !important;
    color: var(--ng-text-primary);
}}

.stApp {{
    background: var(--ng-bg-base) !important;
    color: var(--ng-text-primary);
}}

main .block-container {{
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}}

/* --- Premium motion keyframes ------------------------------------------ */

@keyframes ng-fade-up {{
    from {{ opacity: 0; transform: translate3d(0, 14px, 0); }}
    to   {{ opacity: 1; transform: translate3d(0, 0, 0); }}
}}
@keyframes ng-fade-in {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes ng-scale-in {{
    from {{ opacity: 0; transform: scale(0.94); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}
@keyframes ng-pulse-dot {{
    0%   {{ box-shadow: 0 0 0 0 var(--ng-success), 0 0 8px var(--ng-success); }}
    70%  {{ box-shadow: 0 0 0 6px transparent, 0 0 8px var(--ng-success); }}
    100% {{ box-shadow: 0 0 0 0 transparent, 0 0 8px var(--ng-success); }}
}}
@keyframes ng-shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}
@keyframes ng-bar-grow {{
    from {{ transform: scaleX(0); }}
    to   {{ transform: scaleX(1); }}
}}
@keyframes ng-rise {{
    from {{ opacity: 0; transform: translate3d(0, 6px, 0); }}
    to   {{ opacity: 1; transform: translate3d(0, 0, 0); }}
}}

/* Page first-paint — soft fade-in over the whole canvas */
.stApp > div {{
    animation: ng-fade-in var(--ng-dur-base) var(--ng-ease-out) backwards;
}}

/* --- Hero / brand strip ------------------------------------------------- */

.hero {{
    position: relative;
    padding: 3rem 2.25rem 2.5rem 2.25rem;
    margin: -1rem 0 2rem 0;
    border-radius: var(--ng-radius-lg);
    background: linear-gradient(180deg,
        var(--ng-bg-elevated) 0%,
        var(--ng-bg-elevated-2) 100%);
    border: 1px solid var(--ng-border);
    box-shadow: var(--ng-shadow-md);
    overflow: hidden;
    /* Premium entrance: subtle drop with expo-out easing (Apple-style) */
    animation: ng-fade-up var(--ng-dur-hero) var(--ng-ease-out) backwards;
    will-change: transform, opacity;
}}

.hero::before {{
    /* Diffuse warm glow that never fully resolves — adds depth without noise */
    content: "";
    position: absolute;
    top: -40%; right: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse at center,
        var(--ng-accent-soft) 0%,
        transparent 60%);
    pointer-events: none;
    opacity: 0.7;
}}

.hero-eyebrow,
.hero-title,
.hero-tagline,
.hero-status-row {{
    position: relative; /* sit above ::before */
}}

.hero-eyebrow {{ animation: ng-rise var(--ng-dur-slow) var(--ng-ease-out) 80ms backwards; }}
.hero-title    {{ animation: ng-rise var(--ng-dur-slow) var(--ng-ease-out) 140ms backwards; }}
.hero-tagline  {{ animation: ng-rise var(--ng-dur-slow) var(--ng-ease-out) 220ms backwards; }}
.hero-status-row {{ animation: ng-rise var(--ng-dur-slow) var(--ng-ease-out) 300ms backwards; }}

.hero::after {{
    content: "";
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 1px;
    background: linear-gradient(180deg,
        transparent 0%,
        var(--ng-accent) 50%,
        transparent 100%);
}}

.hero-eyebrow {{
    font-family: var(--ng-font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--ng-accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 0 0 0.85rem 0;
}}

.hero-title {{
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--ng-text-primary);
    letter-spacing: -0.025em;
    line-height: 1.05;
    margin: 0 0 0.6rem 0;
}}

.hero-title .accent {{
    color: var(--ng-accent);
    font-weight: 800;
}}

.hero-tagline {{
    color: var(--ng-text-secondary);
    font-size: 1.05rem;
    line-height: 1.55;
    margin: 0 0 1.25rem 0;
    max-width: 60ch;
}}

.hero-status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin-top: 0.5rem;
}}

/* --- Status dots + pills ----------------------------------------------- */

.dot {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.32rem 0.72rem;
    border-radius: 999px;
    font-family: var(--ng-font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--ng-bg-elevated-3);
    color: var(--ng-text-secondary);
    border: 1px solid var(--ng-border);
    transition: border-color var(--ng-dur-fast) var(--ng-ease-standard),
                background var(--ng-dur-fast) var(--ng-ease-standard);
}}
.dot:hover {{ border-color: var(--ng-border-strong); }}

.dot::before {{
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--ng-text-tertiary);
    transition: background var(--ng-dur-base) var(--ng-ease-standard);
}}

.dot.is-ok::before    {{
    background: var(--ng-success);
    /* Subtle "alive" pulse — only on healthy state, says system is breathing */
    animation: ng-pulse-dot 2.4s var(--ng-ease-standard) infinite;
}}
.dot.is-warn::before  {{ background: var(--ng-warning); }}
.dot.is-down::before  {{ background: var(--ng-danger); }}
.dot.is-mute::before  {{ background: var(--ng-text-tertiary); }}

/* --- Section header ----------------------------------------------------- */

.section {{
    margin: 2rem 0 1.5rem 0;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--ng-border);
    animation: ng-rise var(--ng-dur-base) var(--ng-ease-out) backwards;
}}
.section-eyebrow {{
    font-family: var(--ng-font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--ng-accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 0 0 0.55rem 0;
}}
.section-title {{
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--ng-text-primary);
    letter-spacing: -0.02em;
    margin: 0 0 0.65rem 0;
    line-height: 1.2;
}}
.section-desc {{
    color: var(--ng-text-secondary);
    font-size: 0.97rem;
    line-height: 1.65;
    margin: 0;
    max-width: 70ch;
}}

/* --- Decision card (BBB) ----------------------------------------------- */

.card {{
    background: var(--ng-bg-elevated);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius-md);
    padding: 1.6rem 1.75rem;
    margin: 1.25rem 0;
    box-shadow: var(--ng-shadow-md);
    /* Drop-in reveal — starts slightly below + faded, settles with expo-out */
    animation: ng-fade-up var(--ng-dur-hero) var(--ng-ease-out) backwards;
    transition: border-color var(--ng-dur-base) var(--ng-ease-standard),
                box-shadow var(--ng-dur-base) var(--ng-ease-standard),
                transform var(--ng-dur-base) var(--ng-ease-standard);
    will-change: transform;
}}
.card:hover {{
    border-color: var(--ng-border-strong);
    box-shadow: var(--ng-shadow-lg);
    transform: translate3d(0, -2px, 0);
}}

.provenance-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    font-family: var(--ng-font-mono);
    font-size: 0.74rem;
    color: var(--ng-text-tertiary);
    letter-spacing: 0.04em;
    margin-bottom: 1.25rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--ng-border);
}}
.provenance-strip strong {{
    color: var(--ng-text-secondary);
    font-weight: 500;
}}

.verdict {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}}
.verdict-label {{
    font-family: var(--ng-font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ng-text-tertiary);
    margin: 0;
}}
.verdict-value {{
    font-size: 3rem;
    font-weight: 800;
    color: var(--ng-accent);
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
    font-feature-settings: "tnum" on, "lnum" on;
    /* Verdict reveal — gentle spring overshoot, the "moment of truth" */
    animation: ng-scale-in var(--ng-dur-hero) var(--ng-ease-spring) 120ms backwards;
}}
.verdict-confidence {{
    font-size: 1.1rem;
    color: var(--ng-text-secondary);
    margin: 0.25rem 0 0 0;
    font-weight: 400;
}}
.verdict-confidence strong {{
    color: var(--ng-text-primary);
    font-weight: 600;
    font-feature-settings: "tnum" on;
}}

.signals {{
    display: grid;
    gap: 0.65rem;
    padding: 1rem 0 1.25rem 0;
    border-top: 1px solid var(--ng-border);
    border-bottom: 1px solid var(--ng-border);
    margin-bottom: 1.25rem;
}}
.signal-row {{
    display: grid;
    grid-template-columns: 100px 1fr;
    gap: 0.85rem;
    align-items: baseline;
    font-size: 0.92rem;
    line-height: 1.55;
    /* Stagger entry — 50ms between rows, very Apple Settings */
    animation: ng-rise var(--ng-dur-base) var(--ng-ease-out) backwards;
}}
.signal-row:nth-child(1) {{ animation-delay: 280ms; }}
.signal-row:nth-child(2) {{ animation-delay: 330ms; }}
.signal-row:nth-child(3) {{ animation-delay: 380ms; }}
.signal-row:nth-child(4) {{ animation-delay: 430ms; }}
.signal-key {{
    font-family: var(--ng-font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--ng-text-tertiary);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.signal-value {{
    color: var(--ng-text-secondary);
    font-feature-settings: "tnum" on;
}}
.signal-value strong {{
    color: var(--ng-text-primary);
    font-weight: 600;
}}

/* --- Streamlit native overrides --------------------------------------- */

/* Buttons — primary CTA = sand block in dark, charcoal in light */
.stButton > button[kind="primary"],
.stButton > button[kind="primaryFormSubmit"] {{
    background: var(--ng-accent) !important;
    color: var(--ng-text-on-accent) !important;
    border: 0 !important;
    border-radius: var(--ng-radius-sm) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    font-size: 0.92rem !important;
    transition: background var(--ng-dur-fast) var(--ng-ease-standard),
                transform var(--ng-dur-fast) var(--ng-ease-out),
                box-shadow var(--ng-dur-fast) var(--ng-ease-standard) !important;
    box-shadow: 0 0 0 0 var(--ng-accent-ring), var(--ng-shadow-sm);
    will-change: transform;
}}
.stButton > button[kind="primary"]:hover {{
    background: var(--ng-accent-strong) !important;
    transform: translate3d(0, -1px, 0);
    box-shadow: 0 0 0 0 var(--ng-accent-ring), var(--ng-shadow-md) !important;
}}
.stButton > button[kind="primary"]:active {{
    /* Apple-style press: brief downward scale, no layout shift */
    transform: translate3d(0, 0, 0) scale(0.97) !important;
    transition-duration: var(--ng-dur-instant) !important;
}}
.stButton > button[kind="primary"]:focus-visible {{
    box-shadow: 0 0 0 3px var(--ng-accent-ring), var(--ng-shadow-sm) !important;
    outline: none !important;
}}

/* Buttons — secondary = transparent border */
.stButton > button:not([kind="primary"]):not([kind="primaryFormSubmit"]) {{
    background: transparent !important;
    color: var(--ng-text-primary) !important;
    border: 1px solid var(--ng-border-strong) !important;
    border-radius: var(--ng-radius-sm) !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.2rem !important;
    transition: border-color var(--ng-dur-fast) var(--ng-ease-standard),
                background var(--ng-dur-fast) var(--ng-ease-standard),
                transform var(--ng-dur-fast) var(--ng-ease-out) !important;
    will-change: transform;
}}
.stButton > button:not([kind="primary"]):not([kind="primaryFormSubmit"]):hover {{
    background: var(--ng-bg-elevated-3) !important;
    border-color: var(--ng-accent) !important;
    transform: translate3d(0, -1px, 0);
}}
.stButton > button:not([kind="primary"]):not([kind="primaryFormSubmit"]):active {{
    transform: translate3d(0, 0, 0) scale(0.97) !important;
    transition-duration: var(--ng-dur-instant) !important;
}}

/* Tabs — left-aligned underline indicator (Apple/Netflix tab strip) */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    border-bottom: 1px solid var(--ng-border);
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: var(--ng-text-tertiary) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.85rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: color var(--ng-dur-base) var(--ng-ease-standard),
                border-color var(--ng-dur-base) var(--ng-ease-out) !important;
    letter-spacing: -0.005em;
    position: relative;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--ng-text-secondary) !important;
}}
/* The hover "ghost" underline — only visible while hovering an inactive tab */
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover::after {{
    content: "";
    position: absolute;
    left: 1.4rem; right: 1.4rem; bottom: -1px;
    height: 2px;
    background: var(--ng-text-tertiary);
    opacity: 0.4;
    animation: ng-fade-in var(--ng-dur-fast) var(--ng-ease-out);
}}
.stTabs [aria-selected="true"] {{
    color: var(--ng-accent) !important;
    border-bottom-color: var(--ng-accent) !important;
    font-weight: 600 !important;
}}
/* Tab content cross-fades on switch */
.stTabs [data-baseweb="tab-panel"] {{
    animation: ng-fade-in var(--ng-dur-base) var(--ng-ease-out) backwards;
}}

/* Inputs — flat with accent-on-focus border + smooth ring expansion */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: var(--ng-bg-elevated-2) !important;
    color: var(--ng-text-primary) !important;
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-sm) !important;
    padding: 0.7rem 0.85rem !important;
    font-family: var(--ng-font-sans) !important;
    font-size: 0.95rem !important;
    transition: border-color var(--ng-dur-fast) var(--ng-ease-standard),
                box-shadow var(--ng-dur-base) var(--ng-ease-out),
                background var(--ng-dur-fast) var(--ng-ease-standard) !important;
}}
.stTextInput > div > div > input:hover,
.stTextArea > div > div > textarea:hover {{
    border-color: var(--ng-border-strong) !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: var(--ng-accent) !important;
    box-shadow: 0 0 0 3px var(--ng-accent-ring) !important;
    outline: none !important;
}}

/* Selectbox */
[data-baseweb="select"] > div {{
    background: var(--ng-bg-elevated-2) !important;
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-sm) !important;
    color: var(--ng-text-primary) !important;
}}

/* Sliders */
.stSlider [role="slider"] {{
    background: var(--ng-accent) !important;
    border: 2px solid var(--ng-bg-base) !important;
}}
.stSlider > div > div > div > div {{
    background: var(--ng-accent) !important;
}}

/* Progress bar — fill animates from 0 with expo-out (the "filling up" beat) */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg,
        var(--ng-accent) 0%,
        var(--ng-accent-strong) 100%) !important;
    border-radius: 999px !important;
    transition: width var(--ng-dur-hero) var(--ng-ease-out) !important;
    box-shadow: 0 0 16px var(--ng-accent-ring);
}}
.stProgress > div > div > div {{
    background: var(--ng-bg-elevated-3) !important;
    border-radius: 999px !important;
}}

/* Metric cards (KPI strip) — drop in with subtle stagger */
[data-testid="stMetric"] {{
    background: var(--ng-bg-elevated) !important;
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-md) !important;
    padding: 1.4rem 1.5rem !important;
    box-shadow: var(--ng-shadow-sm);
    animation: ng-fade-up var(--ng-dur-slow) var(--ng-ease-out) backwards;
    transition: border-color var(--ng-dur-base) var(--ng-ease-standard),
                box-shadow var(--ng-dur-base) var(--ng-ease-standard),
                transform var(--ng-dur-base) var(--ng-ease-standard);
    will-change: transform;
}}
[data-testid="stMetric"]:hover {{
    border-color: var(--ng-border-strong) !important;
    box-shadow: var(--ng-shadow-md);
    transform: translate3d(0, -2px, 0);
}}
/* Stagger when 3 metrics sit side-by-side */
[data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(1) {{ animation-delay: 80ms; }}
[data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(2) {{ animation-delay: 140ms; }}
[data-testid="stHorizontalBlock"] [data-testid="stMetric"]:nth-child(3) {{ animation-delay: 200ms; }}
[data-testid="stMetricLabel"] > div {{
    color: var(--ng-text-tertiary) !important;
    font-family: var(--ng-font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}}
[data-testid="stMetricValue"] > div {{
    color: var(--ng-text-primary) !important;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.02em !important;
    font-feature-settings: "tnum" on, "lnum" on !important;
    line-height: 1.1 !important;
}}
[data-testid="stMetricDelta"] {{
    color: var(--ng-text-secondary) !important;
}}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {{
    color: var(--ng-text-tertiary) !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
}}

/* Expander — chevron rotates smoothly + body cross-fades */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {{
    background: var(--ng-bg-elevated-2) !important;
    color: var(--ng-text-primary) !important;
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-sm) !important;
    font-weight: 500 !important;
    transition: background var(--ng-dur-fast) var(--ng-ease-standard),
                border-color var(--ng-dur-fast) var(--ng-ease-standard) !important;
    cursor: pointer;
}}
[data-testid="stExpander"] details summary:hover {{
    background: var(--ng-bg-elevated-3) !important;
    border-color: var(--ng-border-strong) !important;
}}
[data-testid="stExpander"] details summary svg {{
    transition: transform var(--ng-dur-base) var(--ng-ease-out) !important;
}}
[data-testid="stExpander"] details[open] summary svg {{
    transform: rotate(90deg);
}}
[data-testid="stExpander"] details[open] > div {{
    animation: ng-rise var(--ng-dur-base) var(--ng-ease-out);
}}
[data-testid="stExpander"] {{
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-sm) !important;
    background: var(--ng-bg-elevated) !important;
}}

/* Code / inline code */
code, pre {{
    background: var(--ng-bg-elevated-3) !important;
    color: var(--ng-accent-strong) !important;
    padding: 0.12rem 0.42rem !important;
    border-radius: 4px !important;
    font-family: var(--ng-font-mono) !important;
    font-size: 0.86rem !important;
}}

/* Alerts (info / warning / error / success) — flat editorial banners */
[data-testid="stAlert"] {{
    background: var(--ng-bg-elevated) !important;
    border: 1px solid var(--ng-border) !important;
    border-left: 3px solid var(--ng-accent) !important;
    border-radius: var(--ng-radius-sm) !important;
    color: var(--ng-text-primary) !important;
    box-shadow: var(--ng-shadow-sm);
    /* Slide-down + fade — alerts feel like notifications dropping in */
    animation: ng-fade-up var(--ng-dur-slow) var(--ng-ease-out) backwards;
}}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {{ border-left-color: var(--ng-accent); }}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {{ border-left-color: var(--ng-warning); }}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {{ border-left-color: var(--ng-danger); }}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {{ border-left-color: var(--ng-success); }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: var(--ng-bg-elevated) !important;
    border-right: 1px solid var(--ng-border) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: var(--ng-text-primary) !important;
}}
section[data-testid="stSidebar"] h3 {{
    font-family: var(--ng-font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: var(--ng-text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
}}

/* Sidebar brand mark */
.sidebar-brand {{
    font-family: var(--ng-font-sans);
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--ng-text-primary);
    letter-spacing: -0.02em;
    margin: 0 0 0.15rem 0;
}}
.sidebar-brand .accent {{
    color: var(--ng-accent);
}}
.sidebar-tagline {{
    font-family: var(--ng-font-mono);
    font-size: 0.7rem;
    color: var(--ng-text-tertiary);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0 0 1.5rem 0;
}}

/* Toggle (theme switch) */
[data-baseweb="checkbox"] [aria-checked="true"] {{
    background: var(--ng-accent) !important;
    border-color: var(--ng-accent) !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: var(--ng-bg-elevated) !important;
    border: 1px solid var(--ng-border) !important;
    border-radius: var(--ng-radius-md) !important;
    overflow: hidden;
}}

/* Markdown headings inside tabs */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
    color: var(--ng-text-primary) !important;
    letter-spacing: -0.015em !important;
}}
.stMarkdown h3 {{
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
}}

/* Divider */
hr, [data-testid="stDivider"] {{
    border-color: var(--ng-border) !important;
    margin: 1.5rem 0 !important;
}}

/* Toast (st.toast) — slide in from bottom-right, expo-out */
.stToast {{
    background: var(--ng-bg-elevated) !important;
    color: var(--ng-text-primary) !important;
    border: 1px solid var(--ng-border) !important;
    box-shadow: var(--ng-shadow-lg) !important;
    animation: ng-fade-up var(--ng-dur-slow) var(--ng-ease-spring) backwards;
}}

/* Chart container — quiet frame */
[data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] {{
    background: var(--ng-bg-elevated);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius-md);
    padding: 1rem;
}}

/* Bar chart (st.bar_chart) inherits the same frame */
[data-testid="stBarChart"] {{
    background: var(--ng-bg-elevated);
    border: 1px solid var(--ng-border);
    border-radius: var(--ng-radius-md);
    padding: 1rem;
}}

/* Reduced motion — fully disable animations + cap transitions to a flash */
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }}
}}

/* Scrollbar — subtle */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--ng-bg-base); }}
::-webkit-scrollbar-thumb {{
    background: var(--ng-bg-elevated-3);
    border-radius: 999px;
    border: 2px solid var(--ng-bg-base);
}}
::-webkit-scrollbar-thumb:hover {{ background: var(--ng-border-strong); }}
</style>
"""


# --------------------------------------------------------------------------- #
# Theme management                                                            #
# --------------------------------------------------------------------------- #

def _init_theme() -> str:
    """Initialize and return the active theme ('dark' default)."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def _altair_theme(theme: str) -> dict:
    """Return an altair theme matching the active palette.

    Registered as 'neurobridge' on first call; subsequent calls just enable.
    """
    tokens = _TOKENS_DARK if theme == "dark" else _TOKENS_LIGHT
    return {
        "config": {
            "background": tokens["bg-elevated"],
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": tokens["text-secondary"],
                "titleColor": tokens["text-secondary"],
                "labelFont": "Inter",
                "titleFont": "Inter",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "gridColor": tokens["border"],
                "domainColor": tokens["border"],
                "tickColor": tokens["border"],
            },
            "header": {
                "labelColor": tokens["text-primary"],
                "labelFont": "Inter",
                "labelFontSize": 13,
                "labelFontWeight": 600,
                "titleColor": tokens["text-secondary"],
            },
            "legend": {
                "labelColor": tokens["text-secondary"],
                "titleColor": tokens["text-secondary"],
                "labelFont": "Inter",
                "titleFont": "Inter",
            },
            "title": {
                "color": tokens["text-primary"],
                "font": "Inter",
                "fontWeight": 600,
            },
            "range": {
                # Editorial palette: sand-led, then warm secondaries.
                "category": [
                    tokens["accent"], "#8FB3C9", "#C99B8F", "#9DAD86",
                    "#B8A4C9", "#D4B86A", "#7FB069", "#A6A2C2",
                ],
            },
        }
    }


def _register_altair_theme(theme: str) -> None:
    """Register + enable the neurobridge altair theme for the current run."""
    try:
        import altair as alt
        alt.themes.register("neurobridge", lambda: _altair_theme(theme))
        alt.themes.enable("neurobridge")
    except Exception:
        # altair may not be importable in some environments; chart calls
        # will simply use altair defaults — no functional impact.
        pass


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                #
# --------------------------------------------------------------------------- #

def _check_api_health() -> tuple[bool, str]:
    """Ping FastAPI /health endpoint; return (ok, status_text)."""
    try:
        resp = httpx.get(f"{_API_URL}/health", timeout=2.0)
        if resp.status_code == 200:
            return True, "operational"
        return False, f"http {resp.status_code}"
    except httpx.RequestError as e:
        return False, type(e).__name__.lower()


def _post(endpoint: str, payload: dict, timeout: float = 120.0) -> dict:
    """POST to the FastAPI surface; let httpx raise on non-2xx."""
    resp = httpx.post(f"{_API_URL}{endpoint}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    """GET helper symmetric with _post."""
    resp = httpx.get(f"{_API_URL}{path}", timeout=10.0)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Hero / sidebar / section primitives                                         #
# --------------------------------------------------------------------------- #

def _render_brand_header(api_ok: bool, api_status: str) -> None:
    """Editorial hero strip: word-mark + tagline + 3 status dots."""
    api_class = "is-ok" if api_ok else "is-down"
    mlflow_class = "is-mute" if _MLFLOW_DISABLED else "is-ok"
    mlflow_label = "tracking off" if _MLFLOW_DISABLED else "tracking"
    llm_class = "is-mute" if _LLM_DISABLED else "is-ok"
    llm_label = "template only" if _LLM_DISABLED else "llm online"

    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-eyebrow">Living decision system · clinical ML</p>
            <h1 class="hero-title">Neuro<span class="accent">Bridge</span> Enterprise</h1>
            <p class="hero-tagline">
                Three production pipelines — molecule, signal, image — behind one
                auditable surface. Every prediction returns label, calibration,
                drift, provenance and a natural-language rationale.
            </p>
            <div class="hero-status-row">
                <span class="dot {api_class}">api · {_html.escape(api_status)}</span>
                <span class="dot {mlflow_class}">mlflow · {mlflow_label}</span>
                <span class="dot {llm_class}">explainer · {llm_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section(eyebrow: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="section">
            <p class="section-eyebrow">{_html.escape(eyebrow)}</p>
            <h2 class="section-title">{_html.escape(title)}</h2>
            <p class="section-desc">{_html.escape(desc)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result(body: dict) -> None:
    """Render a 3-metric result card + (optional) MLflow deep link."""
    cols = st.columns(3)
    cols[0].metric("Rows", f"{body['rows']:,}")
    cols[1].metric("Columns", f"{body['columns']:,}")
    cols[2].metric("Runtime", f"{body['duration_sec']:.2f} s")

    safe_output_path = _html.escape(str(body["output_path"]))
    st.markdown(
        f"<p style='color:var(--ng-text-tertiary);"
        f"margin:1rem 0 0.5rem 0;font-size:0.85rem;'>"
        f"output → <code>{safe_output_path}</code></p>",
        unsafe_allow_html=True,
    )

    run_id = body.get("mlflow_run_id")
    if run_id and not _MLFLOW_DISABLED:
        safe_run_id = _html.escape(str(run_id))
        safe_url = _html.escape(_MLFLOW_URL, quote=True)
        st.markdown(
            f"<p style='color:var(--ng-text-tertiary);font-size:0.85rem;'>"
            f"mlflow run · <a href='{safe_url}/#/experiments/0/runs/{safe_run_id}' "
            f"target='_blank' rel='noopener noreferrer' "
            f"style='color:var(--ng-accent);text-decoration:none;"
            f"border-bottom:1px solid var(--ng-accent-ring);'>"
            f"{safe_run_id[:12]}…</a></p>",
            unsafe_allow_html=True,
        )
    elif _MLFLOW_DISABLED:
        st.caption("mlflow tracking disabled (NEUROBRIDGE_DISABLE_MLFLOW=1)")


def _render_sidebar(api_ok: bool, api_status: str) -> None:
    with st.sidebar:
        st.markdown(
            """
            <p class="sidebar-brand">Neuro<span class="accent">Bridge</span></p>
            <p class="sidebar-tagline">enterprise · v1</p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Theme")
        theme = st.session_state.get("theme", "dark")
        is_dark = st.toggle(
            "Dark mode",
            value=(theme == "dark"),
            key="theme_toggle",
            help="Switch between editorial dark (Netflix-style) and warm paper (Apple HIG-style).",
        )
        new_theme = "dark" if is_dark else "light"
        if new_theme != theme:
            st.session_state["theme"] = new_theme
            st.rerun()

        st.markdown("### System")
        api_class = "is-ok" if api_ok else "is-down"
        mlflow_class = "is-mute" if _MLFLOW_DISABLED else "is-ok"
        llm_class = "is-mute" if _LLM_DISABLED else "is-ok"
        st.markdown(
            f"""
            <div style='display:flex;flex-direction:column;gap:0.4rem;'>
                <span class='dot {api_class}'>api · {_html.escape(api_status)}</span>
                <span class='dot {mlflow_class}'>mlflow · {"off" if _MLFLOW_DISABLED else "on"}</span>
                <span class='dot {llm_class}'>llm · {"template" if _LLM_DISABLED else "online"}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Endpoints")
        st.markdown(
            f"<p style='font-family:var(--ng-font-mono);font-size:0.78rem;"
            f"color:var(--ng-text-tertiary);line-height:1.8;margin:0;'>"
            f"fastapi · <code>{_API_URL}</code><br/>"
            f"mlflow &nbsp;· <code>{_MLFLOW_URL}</code></p>",
            unsafe_allow_html=True,
        )

        if st.button("🔧 Diagnose LLM", key="diag_llm_btn", help="Probe OpenRouter from this container"):
            try:
                diag = httpx.get(f"{_API_URL}/diag/openrouter", timeout=15.0).json()
                st.json(diag)
            except Exception as e:
                st.error(f"diag failed: {e!r}")

        st.markdown("### About")
        st.markdown(
            "<p style='font-size:0.86rem;color:var(--ng-text-secondary);"
            "line-height:1.65;margin:0;'>"
            "Trust-engineered clinical-ML platform. Three modalities — BBB drug "
            "screening, EEG signal cleaning, MRI multi-site harmonization — "
            "behind one FastAPI surface. Every inference is auditable.</p>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Tabs                                                                        #
# --------------------------------------------------------------------------- #

def _render_bbb_tab() -> None:
    _render_section(
        "MOLECULE — BBBP",
        "Blood-Brain-Barrier permeability decision",
        "Enter a SMILES string. The system computes a 2,048-bit Morgan "
        "fingerprint, runs it through a Random Forest classifier, and returns "
        "a label, calibration-grounded confidence, drift signal, and the top "
        "SHAP attributions explaining the decision.",
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
        "Test edge cases",
        options=list(EDGE_CASES.keys()),
        index=0,
        key="bbb_case",
        help=(
            "Pick a robustness probe. Each case demonstrates how the system "
            "handles a real-world failure mode — invalid input, "
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
        with st.spinner("Computing fingerprint, predicting, explaining…"):
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
                    # Robustness story: WARNING (recoverable), not ERROR.
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
    eeg_in = st.text_input(
        "Input FIF/EDF path",
        "tests/fixtures/eeg_sample.fif",
        key="eeg_in",
        help=(
            "Defaults to the bundled EEG fixture so the demo runs out of "
            "the box. Replace with your own .fif/.edf path on a real run."
        ),
    )
    eeg_out = st.text_input(
        "Output Parquet path",
        "data/processed/eeg_features.parquet",
        key="eeg_out",
    )
    if st.button("Run EEG pipeline", type="primary", key="eeg_run"):
        with st.spinner("Filtering and running ICA…"):
            try:
                result = _post(
                    "/pipeline/eeg",
                    {"input_path": eeg_in, "output_path": eeg_out},
                )
                st.session_state["last_eeg_run"] = result
                _render_result(result)
                st.toast("EEG pipeline complete", icon="✅")
            except httpx.HTTPStatusError as e:
                st.error(
                    f"Pipeline failed (HTTP {e.response.status_code}): "
                    f"{e.response.text}"
                )
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")

    last_eeg = st.session_state.get("last_eeg_run")
    if last_eeg is not None:
        with st.expander("Ask the AI Assistant about this EEG run", expanded=False):
            eeg_q_presets = [
                "Why were certain ICA components dropped?",
                "What does the bandpass filter do?",
                "Is this run consistent with previous runs?",
            ]
            eeg_preset = st.selectbox(
                "Preset question", options=eeg_q_presets, key="eeg_ai_preset",
            )
            eeg_custom = st.text_input(
                "Or type your own question (optional)",
                value="", key="eeg_ai_custom",
            )
            eeg_question = eeg_custom.strip() or eeg_preset
            if st.button("Ask AI Assistant", key="eeg_ai_ask"):
                with st.spinner("Composing rationale…"):
                    try:
                        eeg_resp = _post(
                            "/explain/eeg",
                            {
                                "rows": int(last_eeg.get("rows", 0)),
                                "columns": int(last_eeg.get("columns", 0)),
                                "duration_sec": float(last_eeg.get("duration_sec", 0.0)),
                                "mlflow_run_id": last_eeg.get("mlflow_run_id"),
                                "user_question": eeg_question,
                            },
                        )
                        st.markdown(f"**A:** {eeg_resp['rationale']}")
                        st.caption(
                            f"Source: `{eeg_resp.get('source', '?')}` · "
                            f"Model: `{eeg_resp.get('model') or '—'}`"
                        )
                    except httpx.HTTPStatusError as e:
                        st.error(
                            f"Assistant failed (HTTP {e.response.status_code}): "
                            f"{e.response.text}"
                        )
                    except httpx.RequestError as e:
                        st.error(f"Cannot reach FastAPI: {e!r}")


def _render_mri_tab() -> None:
    _render_section(
        "IMAGE — MRI",
        "Multi-site harmonization via ComBat",
        "Loads NIfTI volumes, masks brain tissue, computes per-ROI summary "
        "statistics, then harmonizes across acquisition sites with "
        "neuroHarmonize to remove scanner-driven domain shift. The diagnostic "
        "plot below compares per-site feature distributions before and after "
        "harmonization.",
    )
    mri_dir = st.text_input(
        "Input NIfTI directory",
        "tests/fixtures/mri_sample",
        key="mri_dir",
        help="Path to a directory of .nii(.gz) files + sites.csv",
    )
    sites_csv = st.text_input(
        "Sites CSV",
        "tests/fixtures/mri_sample/sites.csv",
        key="mri_sites",
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

    st.markdown("#### MRI Image Model")
    mri_image = st.text_input(
        "NIfTI image",
        "tests/fixtures/mri_sample/subject_0.nii.gz",
        key="mri_predict_image",
    )
    mri_labels = st.text_input(
        "Class labels",
        "control,abnormal",
        key="mri_predict_labels",
    )
    shape_cols = st.columns(3)
    target_d = shape_cols[0].number_input(
        "Resize D", min_value=1, max_value=256, value=64, step=1, key="mri_predict_d"
    )
    target_h = shape_cols[1].number_input(
        "Resize H", min_value=1, max_value=256, value=64, step=1, key="mri_predict_h"
    )
    target_w = shape_cols[2].number_input(
        "Resize W", min_value=1, max_value=256, value=64, step=1, key="mri_predict_w"
    )
    st.caption(
        "Defaults to 64³ for production exports. Use 8³ when testing with the "
        "dummy ONNX fixture from `tests/fixtures/build_dummy_mri_onnx.py`."
    )
    if st.button("Predict MRI image", key="mri_predict"):
        labels = [x.strip() for x in mri_labels.split(",") if x.strip()]
        payload: dict = {
            "input_path": mri_image,
            "target_shape": [int(target_d), int(target_h), int(target_w)],
        }
        if labels:
            payload["label_names"] = labels
        with st.spinner("Running MRI image model..."):
            try:
                result = _post("/predict/mri", payload, timeout=120.0)
            except httpx.HTTPStatusError as e:
                detail = e.response.text
                if e.response.status_code == 503:
                    st.warning(
                        "MRI model artifact is not available yet. Export the trained "
                        "ONNX model to `data/processed/mri_model.onnx` or set `MRI_MODEL_PATH`."
                    )
                else:
                    st.error(f"MRI prediction failed (HTTP {e.response.status_code}): {detail}")
            except httpx.RequestError as e:
                st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
            else:
                st.metric(
                    label=result.get("label_text", "prediction"),
                    value=f"{float(result.get('confidence', 0.0)) * 100:.1f}%",
                )
                probs = result.get("probabilities", [])
                if probs:
                    st.dataframe(probs, use_container_width=True, hide_index=True)


def _render_prediction_card(result: dict) -> None:
    """Editorial decision card: provenance · verdict · signals · SHAP."""
    st.session_state["last_bbb_prediction"] = result
    label_text = _html.escape(str(result["label_text"]))
    confidence_pct = float(result["confidence"]) * 100

    # 1) Provenance strip (auditable line)
    provenance = result.get("provenance") or {}
    run_id = provenance.get("mlflow_run_id")
    run_label = run_id[:8] if run_id else "—"
    train_date = provenance.get("train_date") or "—"
    model_version = provenance.get("model_version", "v1")
    n_examples = provenance.get("n_examples")
    n_label = f"n={n_examples}" if n_examples else "n=—"

    # 2) Build signal rows: calibration, drift
    signal_rows: list[tuple[str, str]] = []

    calibration = result.get("calibration")
    if calibration is not None:
        threshold_pct = round(float(calibration["threshold"]) * 100)
        precision_pct = round(float(calibration["precision"]) * 100)
        support = int(calibration["support"])
        if support == 0:
            cal_str = "no held-out support in this band"
        else:
            cal_str = (
                f"≥{threshold_pct}% confident → "
                f"<strong>{precision_pct}%</strong> precision · n={support}"
            )
        signal_rows.append(("calibration", cal_str))

    drift_z = result.get("drift_z")
    rolling_n = int(result.get("rolling_n", 0))
    if drift_z is None and rolling_n < 10:
        drift_str = f"warming up · {rolling_n}/10 buffered"
    elif drift_z is None:
        drift_str = "unavailable · model lacks train-time stats"
    else:
        if abs(drift_z) < 1.0:
            tag = "within expected range"
        elif abs(drift_z) < 2.0:
            tag = "mild distribution shift"
        else:
            tag = "significant shift — retrain recommended"
        drift_str = (
            f"trailing-{rolling_n} median <strong>{drift_z:+.2f}σ</strong> · {tag}"
        )
    signal_rows.append(("drift", drift_str))

    signals_html = "".join(
        f'<div class="signal-row"><span class="signal-key">{k}</span>'
        f'<span class="signal-value">{v}</span></div>'
        for k, v in signal_rows
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="provenance-strip">
                <span>mlflow · <strong>{_html.escape(run_label)}</strong></span>
                <span>model · <strong>{_html.escape(model_version)}</strong></span>
                <span>trained · <strong>{_html.escape(train_date)}</strong></span>
                <span><strong>{_html.escape(n_label)}</strong></span>
            </div>
            <div class="verdict">
                <p class="verdict-label">verdict</p>
                <p class="verdict-value">{label_text.lower()}</p>
                <p class="verdict-confidence">
                    Model confidence · <strong>{confidence_pct:.1f}%</strong>
                </p>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Native progress bar — themed via CSS variables
    st.progress(float(result["confidence"]))

    st.markdown(
        f"""
            <div class="signals">
                {signals_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SHAP attributions chart
    n_features = len(result["top_features"])
    st.markdown(
        f'<p class="section-eyebrow" style="margin-top:1.5rem;">'
        f'top {n_features} shap attributions</p>',
        unsafe_allow_html=True,
    )
    import pandas as pd
    shap_df = pd.DataFrame(result["top_features"]).set_index("feature")
    # Keep st.bar_chart for simplicity; the wrapper now sits in a themed frame.
    st.bar_chart(shap_df, height=240, color=_TOKENS_DARK["accent"]
                 if st.session_state.get("theme", "dark") == "dark"
                 else _TOKENS_LIGHT["accent"])

    st.caption(
        "Positive SHAP values pushed the model toward the predicted class; "
        "negative values pushed it away. Features are 2,048-bit Morgan "
        "fingerprint indices (`fp_<bit>`)."
    )


def _render_combat_diagnostics(result: dict) -> None:
    """Pre/Post-ComBat KDE comparison + 3-metric site-gap KPI strip."""
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
            "Pre-gap / Post-gap. A 100× reduction means ComBat removed "
            "two orders of magnitude of site-driven domain shift."
        ),
    )

    df = pd.DataFrame(rows)
    feat = df["feature"].iloc[0]
    feat_df = df[df["feature"] == feat]

    chart = (
        alt.Chart(feat_df)
        .transform_density(
            density="feature_value",
            groupby=["site", "harmonization_state"],
            as_=["feature_value", "density"],
        )
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("feature_value:Q", title=f"{feat} (intensity)"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(
                "site:N",
                title="Site",
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
            )
        )
        .resolve_scale(x="shared", y="shared")
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"Per-site density of `{feat}` before and after ComBat. Each colored "
        f"region is one acquisition site. **Convergence of the colored "
        f"regions in the Post-ComBat panel is the visual proof of "
        f"harmonization** — the same property the "
        f"{result['reduction_factor']:.0f}× site-gap reduction quantifies."
    )

    n_subjects = len({r["subject_id"] for r in result.get("rows", [])})
    with st.expander("Ask the AI Assistant about this ComBat run", expanded=False):
        mri_q_presets = [
            "Why does ComBat matter for multi-site MRI?",
            "How significant is this reduction factor?",
            "What would I lose without harmonization?",
        ]
        mri_preset = st.selectbox(
            "Preset question", options=mri_q_presets, key="mri_ai_preset",
        )
        mri_custom = st.text_input(
            "Or type your own question (optional)",
            value="", key="mri_ai_custom",
        )
        mri_question = mri_custom.strip() or mri_preset
        if st.button("Ask AI Assistant", key="mri_ai_ask"):
            with st.spinner("Composing rationale…"):
                try:
                    mri_resp = _post(
                        "/explain/mri",
                        {
                            "site_gap_pre": float(result["site_gap_pre"]),
                            "site_gap_post": float(result["site_gap_post"]),
                            "reduction_factor": float(result["reduction_factor"]),
                            "n_subjects": n_subjects,
                            "user_question": mri_question,
                        },
                    )
                    st.markdown(f"**A:** {mri_resp['rationale']}")
                    st.caption(
                        f"Source: `{mri_resp.get('source', '?')}` · "
                        f"Model: `{mri_resp.get('model') or '—'}`"
                    )
                except httpx.HTTPStatusError as e:
                    st.error(
                        f"Assistant failed (HTTP {e.response.status_code}): "
                        f"{e.response.text}"
                    )
                except httpx.RequestError as e:
                    st.error(f"Cannot reach FastAPI: {e!r}")


def _render_ai_assistant_tab() -> None:
    """Chat-style explainer for the most recent BBB prediction."""
    _render_section(
        "AI Assistant",
        "Natural-language rationale (LLM or deterministic template)",
        "Pulls the most recent BBB prediction from this session and asks the "
        "explainer to justify it. Falls back to a deterministic, auditable "
        "template when no LLM is configured.",
    )

    last = st.session_state.get("last_bbb_prediction")
    if last is None:
        st.info(
            "Run a BBB prediction first (Molecule tab → Predict button), "
            "then come back here to ask the assistant about it."
        )
        return

    top_features_preview = ", ".join(
        f["feature"] for f in last.get("top_features", [])[:3]
    )
    st.caption(
        f"Latest prediction: **{last['label_text']}** "
        f"({float(last['confidence']) * 100:.0f}% confident)  ·  "
        f"Top SHAP: {top_features_preview}"
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
        help=(
            "Custom questions only affect the LLM path; the template gives a "
            "generic SHAP-driven rationale either way."
        ),
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

    history = st.session_state.get("explain_history", [])
    if history:
        st.markdown("### Conversation")
        for q, r in history[:10]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {r['rationale']}")
            source = r.get("source", "?")
            model = r.get("model") or "—"
            st.caption(f"Source: `{source}`  ·  Model: `{model}`")
            st.divider()


def _render_experiments_tab() -> None:
    """MLflow runs table + two-run diff (Track 5)."""
    _render_section(
        "Experiments — MLOps Audit",
        "MLflow runs across BBB / EEG / MRI experiments",
        "Lists every recorded training run; pick any two to see a side-by-side "
        "metric + parameter diff. Foundation for auditable, reproducible "
        "model lineage.",
    )

    if st.button("Refresh runs", key="exp_refresh"):
        st.session_state.pop("experiments_runs_cache", None)

    runs = st.session_state.get("experiments_runs_cache")
    if runs is None:
        try:
            data = _get("/experiments/runs")
            runs = data.get("runs", [])
            st.session_state["experiments_runs_cache"] = runs
        except httpx.HTTPStatusError as e:
            st.error(
                f"Failed to load runs (HTTP {e.response.status_code}): "
                f"{e.response.text}"
            )
            return
        except httpx.RequestError as e:
            st.error(f"Cannot reach FastAPI at {_API_URL}: {e!r}")
            return

    if not runs:
        st.info(
            "No MLflow runs found. Trigger a pipeline first (Molecule / "
            "Signal / Image), then refresh this tab. (Under "
            "NEUROBRIDGE_DISABLE_MLFLOW=1 the list will stay empty.)"
        )
        return

    rows_preview = [
        {
            "run_id": run["run_id"][:8],
            "experiment": run["experiment_name"],
            "start_time": run["start_time"][:19],
            "status": run["status"],
            "n_metrics": len(run["metrics"]),
            "n_params": len(run["params"]),
        }
        for run in runs
    ]
    st.dataframe(rows_preview, use_container_width=True, hide_index=True)

    st.markdown("### Compare two runs")
    run_ids = [r["run_id"] for r in runs]
    if len(run_ids) < 2:
        st.caption("Need at least 2 runs to compare. Trigger another pipeline.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        sel_a = st.selectbox(
            "Run A", options=run_ids,
            format_func=lambda x: x[:8], key="diff_a",
        )
    with col_b:
        sel_b = st.selectbox(
            "Run B", options=run_ids,
            index=min(1, len(run_ids) - 1),
            format_func=lambda x: x[:8], key="diff_b",
        )

    if st.button("Show diff", type="primary", key="exp_diff_go"):
        try:
            diff = _post(
                "/experiments/diff",
                {"run_id_a": sel_a, "run_id_b": sel_b},
            )
        except httpx.HTTPStatusError as e:
            st.error(
                f"Diff failed (HTTP {e.response.status_code}): "
                f"{e.response.text}"
            )
            return
        rows = diff.get("rows", [])
        if not rows:
            st.info("Both runs have identical metrics and params (or are empty).")
            return
        diff_table = [
            {
                "key": r["key"],
                "kind": r["kind"],
                "A": r["value_a"] or "—",
                "B": r["value_b"] or "—",
                "differs": "✓" if r["differs"] else "",
            }
            for r in rows
        ]
        st.dataframe(diff_table, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Streamlit entrypoint. Idempotent — Streamlit re-runs on every interaction."""
    st.set_page_config(
        page_title="NeuroBridge Enterprise",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    theme = _init_theme()
    st.markdown(_build_css(theme), unsafe_allow_html=True)
    _register_altair_theme(theme)

    api_ok, api_status = _check_api_health()
    _render_brand_header(api_ok, api_status)
    _render_sidebar(api_ok, api_status)

    if not api_ok:
        st.warning(
            f"FastAPI surface is not reachable at `{_API_URL}` ({api_status}). "
            "Pipeline runs will fail until the API service is up. "
            "Run `uvicorn src.api.main:app --port 8000` or `docker compose up`."
        )

    bbb_tab, eeg_tab, mri_tab, assistant_tab, experiments_tab, agent_tab = st.tabs([
        "Molecule",
        "Signal",
        "Image",
        "AI Assistant",
        "Experiments",
        "🤖 Agent",
    ])

    with bbb_tab:
        _render_bbb_tab()
    with eeg_tab:
        _render_eeg_tab()
    with mri_tab:
        _render_mri_tab()
    with assistant_tab:
        _render_ai_assistant_tab()
    with experiments_tab:
        _render_experiments_tab()

    with agent_tab:
        st.markdown("### Orchestrator Agent")
        st.caption(
            "Pick the pipeline automatically, run it, then ground the response "
            "in curated reference docs (RAG)."
        )

        with st.form("agent_form"):
            agent_input = st.text_input(
                "Input",
                value="CCO",
                help="SMILES (e.g., CCO), .fif/.edf path, or NIfTI directory path",
            )
            agent_question = st.text_input(
                "Question (optional)",
                value="",
                help="Ask in any language — the agent will mirror it in the response",
            )
            agent_sites_csv = st.text_input(
                "MRI sites CSV (optional)",
                value="",
                help="Defaults to <MRI input directory>/sites.csv",
            )
            submitted = st.form_submit_button("Run agent")

        if submitted and agent_input:
            with st.spinner("Agent is reasoning..."):
                try:
                    payload: dict = {"user_input": agent_input}
                    if agent_question:
                        payload["user_question"] = agent_question
                    if agent_sites_csv:
                        payload["sites_csv"] = agent_sites_csv
                    response = _post("/agent/run", payload, timeout=120.0)
                except Exception as e:
                    st.error(f"Agent run failed: {e}")
                else:
                    st.markdown("#### Response")
                    st.write(response.get("text", ""))
                    st.caption(
                        f"model: `{response.get('model', '?')}` · "
                        f"finish: `{response.get('finish_reason', '?')}`"
                    )
                    trace = response.get("trace", [])
                    expander_title = f"🧠 Decision trace ({len(trace)} step{'s' if len(trace) != 1 else ''})"
                    with st.expander(expander_title, expanded=True):
                        if not trace:
                            st.write("_(no tool calls)_")
                        for i, step in enumerate(trace, start=1):
                            st.markdown(f"**{i}. `{step['name']}`**")
                            if step.get("error"):
                                st.error(step["error"])
                            else:
                                st.json(step.get("args", {}))
                                st.json(step.get("result", {}))


if __name__ == "__main__":
    main()
