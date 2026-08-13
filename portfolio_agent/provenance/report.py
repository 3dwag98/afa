"""Render a manifest as a standalone research note.

Why HTML, and why standalone
----------------------------
A result that lives only in terminal scrollback is a result nobody re-reads.
The point of the note is that a run can be opened six weeks later, in a
browser, by someone who was not there — so it has to carry its own caveats, its
own provenance, and enough of the numbers that the reader does not have to go
looking for the rest.

Standalone means one file with no external requests: no CDN, no font, no chart
library. That is partly so it works offline and partly because a research note
that silently fails to render when a CDN is unreachable is worse than a plain
table. The decay chart is inline SVG drawn from the numbers directly, which is
about thirty lines and has no version to pin.

What the note leads with
------------------------
The dirty-tree warning, in the header, before the metrics. A number produced
from uncommitted code is not reproducible and is otherwise indistinguishable
from one that is; burying that in a footer would make the note complicit in the
confusion it exists to prevent.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .manifest import RunManifest

logger = logging.getLogger(__name__)

_STYLE = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --rule: #e5e7eb;
  --accent: #1d4ed8; --warn-bg: #fef2f2; --warn-fg: #991b1b; --warn-rule: #fecaca;
  --ok: #047857; --code-bg: #f6f7f9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa3af; --rule: #262b33;
    --accent: #7aa2f7; --warn-bg: #2a1416; --warn-fg: #fca5a5; --warn-rule: #7f1d1d;
    --ok: #34d399; --code-bg: #161a20;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .6rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--rule); }
.sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: .875em; background: var(--code-bg); padding: .1em .35em;
              border-radius: 3px; }
.warn { background: var(--warn-bg); color: var(--warn-fg);
        border: 1px solid var(--warn-rule); border-radius: 6px;
        padding: .8rem 1rem; margin: 0 0 1.5rem; }
.warn strong { display: block; margin-bottom: .2rem; }
.ok { color: var(--ok); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
        gap: .75rem; margin: 0 0 1rem; }
.card { border: 1px solid var(--rule); border-radius: 6px; padding: .7rem .85rem; }
.card .label { color: var(--muted); font-size: .75rem; text-transform: uppercase;
               letter-spacing: .04em; }
.card .value { font-size: 1.1rem; margin-top: .15rem;
               font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .4rem .7rem; border-bottom: 1px solid var(--rule); }
th { color: var(--muted); font-weight: 600; font-size: .78rem;
     text-transform: uppercase; letter-spacing: .04em; }
td.num { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.note { border-left: 3px solid var(--accent); padding: .5rem .9rem; margin: .6rem 0;
        color: var(--muted); }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
         border-top: 1px solid var(--rule); padding-top: 1rem; }
svg { max-width: 100%; height: auto; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if abs(value) >= 1000 or (value and abs(value) < 1e-4):
            return f"{value:.4g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return _e(value)


def _cards(pairs: Sequence[tuple]) -> str:
    cells = "".join(
        f'<div class="card"><div class="label">{_e(label)}</div>'
        f'<div class="value">{_format(value)}</div></div>'
        for label, value in pairs
    )
    return f'<div class="grid">{cells}</div>'


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return '<p class="sub">Nothing recorded.</p>'
    head = "".join(f"<th>{_e(c)}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            cells.append(
                f'<td class="num">{_format(value)}</td>' if numeric
                else f"<td>{_format(value)}</td>"
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _decay_svg(points: Sequence[Mapping[str, Any]], width: int = 640, height: int = 220) -> str:
    """A line chart of IC against horizon, drawn from the numbers.

    Inline SVG rather than a charting library: the note has to render with no
    network, and a chart that silently disappears when a CDN is unreachable is
    worse than no chart. Thirty lines, nothing to pin.
    """
    usable = [
        p for p in points
        if isinstance(p.get("horizon"), (int, float))
        and isinstance(p.get("mean_ic"), (int, float))
    ]
    if len(usable) < 2:
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 52, 16, 16, 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    xs = [float(p["horizon"]) for p in usable]
    ys = [float(p["mean_ic"]) for p in usable]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(min(ys), 0.0), max(max(ys), 0.0)
    if y_max == y_min:
        y_max = y_min + 1e-6
    x_span = (x_max - x_min) or 1.0

    def sx(x: float) -> float:
        return pad_left + (x - x_min) / x_span * plot_w

    def sy(y: float) -> float:
        return pad_top + (y_max - y) / (y_max - y_min) * plot_h

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
        for i, (x, y) in enumerate(zip(xs, ys))
    )
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="currentColor"/>'
        for x, y in zip(xs, ys)
    )
    x_labels = "".join(
        f'<text x="{sx(x):.1f}" y="{height - 12}" text-anchor="middle" '
        f'font-size="11" fill="currentColor" opacity=".65">{int(x)}d</text>'
        for x in xs
    )
    y_ticks = "".join(
        f'<line x1="{pad_left}" y1="{sy(v):.1f}" x2="{width - pad_right}" '
        f'y2="{sy(v):.1f}" stroke="currentColor" stroke-width="1" opacity=".12"/>'
        f'<text x="{pad_left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="currentColor" opacity=".65">{v:+.3f}</text>'
        for v in (y_min, (y_min + y_max) / 2, y_max)
    )
    zero = (
        f'<line x1="{pad_left}" y1="{sy(0.0):.1f}" x2="{width - pad_right}" '
        f'y2="{sy(0.0):.1f}" stroke="currentColor" stroke-width="1" opacity=".35"/>'
        if y_min < 0 < y_max else ""
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Information coefficient against forecast horizon">'
        f"{y_ticks}{zero}"
        f'<path d="{path}" fill="none" stroke="currentColor" stroke-width="2"/>'
        f"{dots}{x_labels}</svg>"
    )


def render_note(manifest: RunManifest) -> str:
    """Render one manifest as a standalone HTML page."""
    git = manifest.git or {}
    dirty = git.get("dirty")

    if dirty is True:
        banner = (
            '<div class="warn"><strong>This result cannot be reproduced.</strong>'
            "The working tree had uncommitted changes when the run happened, so "
            "the code that produced these numbers is not recoverable from the "
            f"recorded commit <code>{_e((git.get('commit') or '?')[:12])}</code>."
            "</div>"
        )
    elif dirty is None:
        banner = (
            '<div class="warn"><strong>Provenance incomplete.</strong>'
            "Git state could not be determined, so it is not known whether this "
            "ran from committed code.</div>"
        )
    else:
        banner = ""

    header_cards = _cards([
        ("Kind", manifest.kind),
        ("Symbols", manifest.n_symbols),
        ("Universe", manifest.universe_fingerprint or "-"),
        ("Config", manifest.config_fingerprint or "-"),
        ("Commit", (git.get("commit") or "unknown")[:12]),
        ("Reproducible", manifest.reproducible),
    ])

    metric_rows = [
        {"metric": key, "value": value}
        for key, value in sorted(manifest.metrics.items())
        if isinstance(value, (int, float, bool))
    ]
    metrics_html = _table(metric_rows, ["metric", "value"])

    settings_rows = [
        {"setting": key, "value": value} for key, value in sorted(manifest.settings.items())
    ]
    settings_html = _table(settings_rows, ["setting", "value"])

    split_html = ""
    if manifest.split:
        split_html = "<h2>Split</h2>" + _table(
            [{"parameter": k, "value": v} for k, v in sorted(manifest.split.items())],
            ["parameter", "value"],
        )

    decay_html = ""
    decay = manifest.extras.get("decay") if manifest.extras else None
    if isinstance(decay, Mapping) and decay.get("points"):
        chart = _decay_svg(decay["points"])
        table = _table(
            decay["points"],
            [c for c in ("horizon", "mean_ic", "icir", "t_stat", "p_value")
             if any(c in p for p in decay["points"])],
        )
        shape = decay.get("shape") or ""
        decay_html = (
            "<h2>Signal decay</h2>"
            + (f'<div class="note">{_e(shape)}</div>' if shape else "")
            + chart + table
        )

    regime_html = ""
    regimes = manifest.extras.get("ic_by_regime") if manifest.extras else None
    if isinstance(regimes, Sequence) and regimes:
        regime_html = "<h2>IC by regime</h2>" + _table(
            list(regimes),
            [c for c in ("regime", "n_dates", "mean_ic", "t_stat")
             if any(c in r for r in regimes)],
        )

    folds_html = ""
    folds = manifest.extras.get("folds") if manifest.extras else None
    if isinstance(folds, Sequence) and folds:
        folds_html = "<h2>Walk-forward folds</h2>" + _table(
            list(folds),
            [c for c in ("fold", "test_start", "test_end", "n_dates",
                         "n_purged", "n_embargoed", "mean_ic", "t_stat")
             if any(c in f for f in folds)],
        )

    caveats = list(manifest.notes)
    caveats_html = ""
    if caveats:
        items = "".join(f'<div class="note">{_e(note)}</div>' for note in caveats)
        caveats_html = f"<h2>Caveats</h2>{items}"

    provenance_rows = [
        {"field": "commit", "value": git.get("commit") or "unknown"},
        {"field": "branch", "value": git.get("branch") or "unknown"},
        {"field": "dirty tree", "value": "unknown" if dirty is None else dirty},
        {"field": "config fingerprint", "value": manifest.config_fingerprint or "-"},
        {"field": "universe fingerprint", "value": manifest.universe_fingerprint or "-"},
        {"field": "universe name", "value": manifest.universe_name or "-"},
    ]
    for key, value in (manifest.data or {}).items():
        provenance_rows.append({"field": f"data.{key}", "value": value})
    for key, value in sorted((manifest.libraries or {}).items()):
        provenance_rows.append({"field": key, "value": value})
    for key, value in sorted((manifest.environment or {}).items()):
        provenance_rows.append({"field": key, "value": value})
    for key, value in (manifest.artifacts or {}).items():
        provenance_rows.append({"field": f"artifact.{key}", "value": value})

    title = f"Run {manifest.run_id}"
    subtitle_bits = [manifest.kind]
    if manifest.strategy:
        subtitle_bits.append(manifest.strategy)
    if manifest.trainer:
        subtitle_bits.append(f"via {manifest.trainer}")
    subtitle_bits.append(manifest.created_at)

    timings_html = ""
    if manifest.timings:
        timings_html = "<h2>Timings</h2>" + _table(
            [{"stage": k, "seconds": v} for k, v in sorted(manifest.timings.items())],
            ["stage", "seconds"],
        )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_STYLE}</style>
</head><body><main>
<h1>{_e(title)}</h1>
<p class="sub">{_e(" · ".join(subtitle_bits))}</p>
{banner}
{header_cards}
<h2>Metrics</h2>
{metrics_html}
{decay_html}
{regime_html}
{folds_html}
{split_html}
<h2>Settings</h2>
{settings_html}
{timings_html}
{caveats_html}
<h2>Provenance</h2>
{_table(provenance_rows, ["field", "value"])}
<footer>
Generated by portfolio-agent from <code>runs/{_e(manifest.run_id)}.json</code>.
Rendering a note re-reads the manifest and computes nothing, so this page and
the run it describes can never disagree.
</footer>
</main></body></html>
"""


def write_note(
    manifest: RunManifest, output: Optional[Path | str] = None
) -> Path:
    """Render a manifest to HTML beside the manifest itself by default."""
    from .manifest import DEFAULT_RUNS_DIR

    path = Path(output) if output is not None else DEFAULT_RUNS_DIR / f"{manifest.run_id}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_note(manifest), encoding="utf-8")
    logger.info("Wrote research note %s", path)
    return path


def render_index(manifests: Sequence[RunManifest]) -> str:
    """A table of runs, newest first."""
    rows = [m.summary() for m in manifests]
    if not rows:
        return "No runs recorded."
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {c: max(len(c), *(len(_format(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(_format(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)
