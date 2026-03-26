# Thesis Figure System Prompt

Use this prompt whenever generating thesis figures so that all diagrams, comparison figures, and charts follow one unified academic visual system.

## Master Prompt

Generate the requested figure under a single unified thesis visual system.
Do not design it as a standalone flashy illustration.
Design it as one member of a consistent figure family for a computer vision thesis on robust multi-model fitting, adaptive estimation, and differential privacy.

### Primary Goal

Optimize for:
- clarity
- consistency
- professional academic presentation
- print readability
- publication-ready output

The figure must look appropriate for a thesis or paper in computer vision, algorithms, or experimental evaluation.
Every generated figure must look like it belongs to the same thesis.

### Global Style Constraints

- Use a clean, minimal, technical academic style.
- Use vector-style graphics whenever possible.
- Use a white background only.
- Do not use decorative textures.
- Do not use 3D rendering.
- Do not use glossy effects.
- Do not use cartoon or presentation-slide aesthetics.
- Do not use gradients unless they are extremely subtle and still publication-safe.
- Prefer simple geometric shapes, crisp lines, and restrained visual emphasis.

### Unified Visual Language

Apply one strict visual system across all figures:
- consistent margins
- consistent spacing
- consistent line widths
- consistent font family
- consistent text hierarchy
- consistent alignment
- consistent panel structure
- consistent color usage

All panels should align to a visible or implied grid.
All comparisons should use matched scales and matched layouts whenever possible.
The figure should remain legible after being scaled down for thesis printing.

### Typography System

Use a clean academic sans-serif font similar to:
- Arial
- Helvetica
- Source Sans

Do not use:
- handwritten fonts
- rounded playful fonts
- decorative fonts
- stylized display fonts

Typography rules:
- keep text crisp and readable
- titles slightly larger than labels
- labels slightly larger than annotations
- keep font sizes consistent within the same hierarchy
- avoid oversized titles
- avoid compressed text
- keep text horizontally aligned when possible
- maintain even spacing between text elements

### Anti-Overlap and Spacing Rules

These are hard constraints:
- absolutely no overlap between text, arrows, shapes, legends, or plotted objects
- leave sufficient padding around labels
- leave enough whitespace between panels
- do not crowd the composition
- simplify wording instead of shrinking text into dense areas
- legends must never cover data
- titles must not collide with figure content
- arrows must not cross text unless unavoidable
- place labels outside dense regions whenever possible
- give each panel a clear boundary or clear spatial separation

### Color System

Use only this restrained academic palette:
- primary dark blue: #1f3a5f
- secondary teal: #4e8da3
- accent orange: #e67e22
- neutral gray: #bfc5cc
- dark text gray: #333333

Color usage rules:
- dark blue for the main method, baseline structure, or primary geometry
- teal for improved, adaptive, or refined components
- orange only for emphasis, privacy noise, warnings, or highlighted differences
- gray for background elements, outliers, inactive components, or low-priority context
- never use rainbow palettes
- never introduce many saturated colors
- maintain strong contrast with professional restraint

### Figure Composition Rules

Each figure must communicate one main message immediately.
Establish a strong visual hierarchy so the reader can identify the comparison focus at a glance.
Make comparisons explicit rather than implied.

For all figures:
- prioritize one central message
- avoid visual clutter
- maximize interpretability before aesthetics
- keep wording minimal and technical
- use repeated structure to improve comparability

### Rules for Comparison Figures

- keep panel sizes identical
- keep object positions consistent across methods
- only change the essential content required for comparison
- make differences visually obvious
- use the same scale and arrangement whenever possible
- emphasize contrast in fitting quality, clustering quality, robustness, or privacy trade-off

### Rules for Pipeline or Framework Figures

- use a clear left-to-right or top-to-bottom flow
- use simple boxes, arrows, separators, and short labels
- highlight adaptive estimation or differential privacy modules subtly but clearly
- avoid unnecessary iconography
- avoid decorative symbols

### Rules for Charts and Experimental Plots

- use clean axes
- use light or minimal gridlines
- use simple legends
- use balanced line thickness
- keep the plot uncluttered
- keep the same typography and color system as the diagrams
- preserve comparability across charts by keeping scales and ordering stable when appropriate

### Output Quality Requirements

- publication-ready
- high resolution
- sharp edges
- suitable for thesis PDF export
- suitable for print
- visually consistent with the rest of the thesis figure family

### Non-Negotiable Constraint

Do not optimize for visual flashiness.
Do not try to make the figure look like a poster, slide, infographic, or marketing graphic.
Optimize for clarity, consistency, technical credibility, and academic print readability.
Every figure must look like part of the same thesis figure family.

## Recommended Usage

Append the specific figure task after the master prompt. For example:

"Using the thesis figure system above, generate a comparison figure showing HVF, Auto-HVF, and DP-HVF under the same layout. Emphasize structure preservation, noise adaptation, and privacy trade-off. Keep labels minimal and publication-ready."
