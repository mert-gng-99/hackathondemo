# HF Space Manual Smoke Checklist — 2026-04-30

After the deploy completes (~3-5 min after `git push hf main`), open
https://mekosotto-hackathon.hf.space/ and walk this list. Anything
that fails is a regression vs the audit fix in
`docs/superpowers/plans/2026-04-30-hf-space-live-audit-fixes.md`.

## Hero strip (top of page)

- [ ] Hero title `NeuroBridge Enterprise` fades up smoothly (not instant)
- [ ] Status row shows three dots:
  - [ ] `api · operational` (green)
  - [ ] `mlflow · tracking` (green) — **was muted before this fix**
  - [ ] `explainer · template only` (muted, expected — LLM stays disabled)

## Molecule tab (BBB)

- [ ] Default edge case "Custom input (default)" is selected
- [ ] Input box shows `CCO`
- [ ] Click "Predict BBB permeability"
- [ ] Decision card animates in with spring scale-in on the verdict
- [ ] Provenance strip shows real values:
  - [ ] `mlflow · <8-char run id>` (NOT `—`) — **was `—` before this fix**
  - [ ] `model · v1`
  - [ ] `trained · <ISO timestamp>` (NOT `—`)
  - [ ] `n=<integer>` (NOT `n=—`)
- [ ] Verdict reads `permeable` with confidence ~80-100%
- [ ] SHAP bar chart renders with sand-colored bars
- [ ] Switch dropdown to "Invalid SMILES" → click Predict → see yellow warning, NOT red error

## Signal tab (EEG)

- [ ] Default input field shows `tests/fixtures/eeg_sample.fif` — **was `data/raw/eeg.fif` before this fix**
- [ ] Click "Run EEG pipeline"
- [ ] Result card shows rows / columns / duration_sec / mlflow_run_id
- [ ] Expand "Ask the AI Assistant about this EEG run"
- [ ] Click "Ask AI Assistant" → see deterministic-template rationale (no error)

## Image tab (MRI)

- [ ] Defaults: `tests/fixtures/mri_sample` and `tests/fixtures/mri_sample/sites.csv`
- [ ] Click "Run ComBat diagnostics"
- [ ] Three KPI cards render: Site-gap (Pre), Site-gap (Post), Reduction factor
- [ ] Pre/Post KDE altair chart renders
- [ ] Expand "Ask the AI Assistant about this ComBat run" → click → see rationale

## AI Assistant tab

- [ ] After running a BBB prediction in the Molecule tab, this tab shows
      "Latest prediction: ..." caption
- [ ] Click "Ask the AI Assistant" → conversation appears with source =
      `template` and model = `—` (LLM intentionally disabled on HF)

## Experiments tab

- [ ] Table loads with **at least 3 rows** — one each for `bbb_pipeline`,
      `eeg_pipeline`, `mri_pipeline` — **was empty before this fix**
- [ ] Compare-two-runs section is visible (≥2 rows are present)
- [ ] Pick two run IDs → click "Show diff" → diff table renders

## Sidebar

- [ ] Toggle "Dark mode" off → page rebuilds with cream paper theme
- [ ] Toggle back on → page rebuilds with editorial dark theme
- [ ] Both themes preserve the sand accent on the hero word-mark

## Reduced-motion respect

- [ ] (Optional) Open DevTools → Rendering → enable "prefers-reduced-motion"
- [ ] Reload — animations are near-instant (< 1ms duration), but layout
      and content are unchanged
