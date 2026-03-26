# Thesis Figure Standard and Figure Plan

This document defines one unified figure standard for the thesis and a concrete figure production plan derived from the current thesis draft, experiment outputs, and repository structure.

## 1. Unified Thesis Figure Standard

All thesis figures must be designed as part of one coherent academic figure family.
The goal is consistency across the whole thesis rather than isolated visual impact.
Every figure should look as if it belongs to the same paper, uses the same design system, and follows the same layout logic.

### 1.1 Core Design Goal

Optimize every figure for:
- academic clarity
- technical precision
- thesis print readability
- visual consistency
- publication-ready quality

Do not optimize for visual flashiness.
Do not use poster, slide, infographic, or marketing aesthetics.

### 1.2 Global Style Rules

Use the following style across all figures:
- white background only
- vector-style output preferred
- minimal and professional composition
- simple geometric shapes only
- no decorative textures
- no 3D effects
- no glossy effects
- no cartoon styling
- no presentation-slide styling
- no gradients unless they are extremely subtle and still print-safe

### 1.3 Color System

Use one restrained palette across the entire thesis:
- primary dark blue: `#1f3a5f`
- secondary teal: `#4e8da3`
- accent orange: `#e67e22`
- neutral gray: `#bfc5cc`
- dark text gray: `#333333`

Color semantics must remain stable:
- dark blue: baseline method, main structure, primary pipeline
- teal: improved or adaptive component, refined mechanism, proposed enhancement
- orange: privacy, injected noise, warning, trade-off emphasis, key delta
- gray: background context, inactive items, outliers, low-priority objects

Rules:
- never use rainbow palettes
- never add extra saturated colors unless absolutely necessary
- emphasize contrast with restraint, not with visual excess

### 1.4 Typography Rules

Use one clean academic sans-serif family throughout all figures, similar to:
- Arial
- Helvetica
- Source Sans

Do not use:
- handwritten fonts
- playful rounded fonts
- decorative fonts
- stylized display fonts

Hierarchy:
- title > panel label > axis/box label > annotation
- titles should be only slightly larger than labels
- labels should remain readable when scaled down for thesis printing
- avoid oversized text
- avoid compressed text
- align text horizontally whenever possible
- keep spacing even and predictable

### 1.5 Layout System

All figures must follow a strict layout system:
- align elements to a visible or implied grid
- keep consistent margins
- keep consistent inner padding
- keep consistent panel gaps
- keep consistent line widths
- keep consistent text placement rules

Anti-overlap rules are mandatory:
- no overlap between text, arrows, shapes, legends, or plotted objects
- legends must not cover data
- titles must not collide with plot content
- arrows should avoid crossing text
- labels should be moved outside dense regions whenever possible
- simplify wording instead of shrinking text into crowded spaces

### 1.6 Figure Family Rules

#### A. Method / pipeline figures

Use:
- left-to-right or top-to-bottom logic
- simple boxes, arrows, separators, and short labels
- subtle highlighting for adaptive estimation and differential privacy modules
- minimal text inside boxes

Avoid:
- decorative icons
- unnecessary symbols
- excessive branching

#### B. Comparison figures

Use:
- identical panel sizes
- identical object placement across methods
- matched scales
- visually explicit differences
- only essential content changed across panels

These figures should make comparison immediate.
The reader should not need to search for what changed.

#### C. Experimental charts

Use:
- clean axes
- light or minimal gridlines
- simple legends
- balanced line thickness
- restrained annotation
- the same color system and typography as diagram figures

Prefer one main message per chart.
Do not overload a chart with too many metrics unless they share the same reading task.

#### D. Qualitative data illustrations

Use:
- sparse labeling
- simple point/line overlays
- matched viewpoints or matched object layouts
- clean separation between inliers, outliers, and model structures

### 1.7 Export Rules

Every final thesis figure should be exported in:
- `SVG` for editable vector output
- `PDF` for thesis insertion and print-safe vector export
- `PNG` for quick preview

Recommended output policy:
- diagrams and framework figures: vector first
- comparison panels: vector first
- charts: vector first whenever possible
- raster preview only as a convenience copy

### 1.8 Naming Convention

Use consistent output names:
- `thesis_fig_<topic>.svg`
- `thesis_fig_<topic>.pdf`
- `thesis_fig_<topic>.png`

Examples:
- `thesis_fig_method_comparison.svg`
- `thesis_fig_dp_tradeoff_overview.pdf`
- `thesis_fig_mixed_calibration_overview.png`

### 1.9 Final Quality Checklist

Before accepting any figure, confirm:
- the main message is obvious within a few seconds
- the typography matches the thesis family
- the palette follows the fixed color semantics
- there is no overlap anywhere
- the layout remains readable after downscaling
- the figure could sit next to other thesis figures without looking stylistically inconsistent

## 2. Figure Production Plan From This Repository

This plan is derived from:
- thesis chapter structure in `THESIS_DRAFT_CN.md`
- method claims in `THESIS_TEMPLATE_CONTENT.md`
- current experiment outputs in `outputs/`
- current plotting support in `visualization/plots.py`
- current method comparison script in `scripts/generate_method_comparison_figure.py`

The list below focuses on thesis figures rather than every possible debug plot.

## 3. Must-Have Thesis Figures

### Figure 1. Overall method comparison: HVF vs Auto-HVF vs DP-HVF

Purpose:
- present the thesis storyline in one figure
- show the transition from fixed-threshold HVF to adaptive Auto-HVF to privacy-aware DP-HVF
- emphasize structure, noise adaptation, and privacy trade-off

Chapter fit:
- Chapter 1 overview or Chapter 3 opening

Status:
- already has a base artifact via `scripts/generate_method_comparison_figure.py`

Recommended final filename:
- `thesis_fig_method_comparison`

### Figure 2. Original HVF pipeline diagram

Purpose:
- explain the baseline processing flow clearly
- show hypothesis generation, residual computation, preference construction, hierarchical voting, clustering, and extraction

Chapter fit:
- Section 3.1

Status:
- not yet formalized as a thesis figure in the repo
- should be generated as a clean baseline pipeline diagram

Recommended final filename:
- `thesis_fig_hvf_pipeline`

### Figure 3. Auto-HVF pipeline diagram

Purpose:
- show where AIKOSE-based adaptive scale estimation enters the HVF pipeline
- visually highlight what changes relative to baseline HVF

Chapter fit:
- Section 3.2

Status:
- not yet formalized as a dedicated thesis figure

Recommended final filename:
- `thesis_fig_auto_hvf_pipeline`

### Figure 4. DP-HVF pipeline diagram with privacy injection points

Purpose:
- show where differential privacy is injected
- highlight model scoring and model selection as privacy-sensitive stages
- show that evaluation is performed on post-DP final output

Chapter fit:
- Section 3.3

Status:
- not yet formalized as a dedicated thesis figure

Recommended final filename:
- `thesis_fig_dp_hvf_pipeline`

### Figure 5. Mixed-scene enhancement and residual calibration diagram

Purpose:
- explain why mixed `F/H` scenes are harder
- show same-type vs cross-type handling and residual calibration logic
- provide the conceptual bridge to the mixed calibration experiment

Chapter fit:
- Section 3.4

Status:
- not yet formalized as a dedicated thesis method figure

Recommended final filename:
- `thesis_fig_mixed_geometry_calibration`

### Figure 6. Privacy budget allocation strategy diagram

Purpose:
- explain `equal` vs `manual` allocation
- show total budget split across injection points
- make the engineering contribution around budget accounting visible

Chapter fit:
- Section 3.5

Status:
- not yet formalized as a dedicated thesis figure

Recommended final filename:
- `thesis_fig_dp_budget_allocation`

### Figure 7. AdelaideRMF dataset / task illustration

Purpose:
- visually introduce the real-data problem setting
- show representative correspondence structures or example mixed scene intuition
- help readers understand what fundamental / homography / mixed mean in the thesis context

Chapter fit:
- Section 4.1

Status:
- currently missing as a thesis-ready figure

Recommended final filename:
- `thesis_fig_adelaide_task_setup`

### Figure 8. Evaluation metric family summary figure

Purpose:
- summarize how the thesis evaluates performance
- group metrics into fitting quality, clustering quality, matching quality, runtime, and privacy cost
- reduce explanation burden in Section 4.2

Chapter fit:
- Section 4.2

Status:
- currently missing as a dedicated figure
- useful even if the thesis also contains metric definitions in text or tables

Recommended final filename:
- `thesis_fig_metric_system`

### Figure 9. Method overview chart across model families

Purpose:
- summarize quality / runtime / model-count trade-off across families
- support the claim that the pipeline runs stably on real data

Chapter fit:
- Section 4.3 or early Chapter 4 results overview

Status:
- already supported by `plot_adelaide_thesis_method_overview(...)`
- current outputs include `*_thesis_method_overview.png`

Recommended final filename:
- `thesis_fig_method_overview`

### Figure 10. DP trade-off overview figure

Purpose:
- show epsilon versus performance and epsilon versus noise scale in one unified view
- support the main DP claim of measurable privacy-performance trade-off

Chapter fit:
- Section 4.4

Status:
- already supported by `plot_dp_tradeoff_overview(...)`
- current outputs include `*_thesis_dp_tradeoff_overview.png`

Recommended final filename:
- `thesis_fig_dp_tradeoff_overview`

### Figure 11. Mixed calibration overview figure

Purpose:
- show that mixed residual calibration preserves fitting quality while reducing scale estimation bias
- support the mixed-scene enhancement argument without overstating theory

Chapter fit:
- Section 4.5

Status:
- already supported by `plot_mixed_calibration_overview(...)`
- current recommended output already exists in `outputs/adelaide_thesis_visual_pub_v7_thesis_calibration_overview.png`

Recommended final filename:
- `thesis_fig_mixed_calibration_overview`

## 4. Recommended Secondary Figures

These are useful if space allows, but they are not as essential as the figures above.

### Figure 12. Sample-level qualitative comparison figure

Purpose:
- show one representative case under HVF / Auto-HVF / DP-HVF using the same sample layout
- make structural differences intuitive beyond aggregate metrics

Why useful:
- helps readers connect abstract metrics to visible fitting behavior

### Figure 13. Single-metric supplementary charts

Candidates:
- `Inlier F1` by family
- `ARI Inlier` by family
- `Match IoU` by family
- `Runtime` by family

Why useful:
- good for appendix or supplementary result pages
- less suitable as the main thesis body figures if a combined overview already exists

### Figure 14. Epsilon sweep single-metric curves

Candidates already exist in `outputs/`:
- `tradeoff_inlier_f1_vs_epsilon`
- `tradeoff_ari_inlier_vs_epsilon`
- `tradeoff_match_iou_vs_epsilon`
- `tradeoff_noise_scale_vs_epsilon`

Why useful:
- can be used in appendix or if the thesis wants to discuss each metric separately after the overview figure

### Figure 15. Budget allocation sensitivity comparison

Purpose:
- compare `equal` and `manual` epsilon allocation
- useful only if you decide to run the additional experiments discussed in the repo

Status:
- concept supported by the codebase, but not yet a standard thesis figure

## 5. Minimal Final Figure Set

If you want the smallest complete thesis figure set with strong coverage, generate these 11 figures:
- Figure 1: method comparison
- Figure 2: HVF pipeline
- Figure 3: Auto-HVF pipeline
- Figure 4: DP-HVF pipeline
- Figure 5: mixed geometry and calibration concept
- Figure 6: privacy budget allocation
- Figure 7: Adelaide task setup
- Figure 8: metric system
- Figure 9: method overview chart
- Figure 10: DP trade-off overview
- Figure 11: mixed calibration overview

## 6. What Already Exists vs What Still Needs To Be Created

Already has a usable base in the repo:
- method comparison figure
- thesis method overview chart
- DP trade-off overview chart
- mixed calibration overview chart

Still needs new thesis-ready figure generation:
- HVF baseline pipeline diagram
- Auto-HVF pipeline diagram
- DP-HVF pipeline diagram
- mixed-scene enhancement / calibration concept diagram
- privacy budget allocation diagram
- Adelaide task setup illustration
- evaluation metric system figure
- optional qualitative case comparison

## 7. Practical Generation Order

Recommended order:
1. finalize the thesis-wide figure standard first
2. finalize the chapter-3 method diagrams
3. finalize the three chapter-4 result overviews
4. add dataset/task and metric-system supporting figures
5. add optional supplementary figures only if needed

This order keeps the core thesis narrative complete as early as possible.
