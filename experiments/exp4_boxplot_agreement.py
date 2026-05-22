"""Experiment 4 — Agreement visualizations.

Three figures per Exp2 results JSON:

  fig_exp2_agreement_<rubric_id>.png
      Box plot of mean pairwise agreement (%) per clinical domain, one panel
      per rubric. ALL configured domains always appear on the x-axis; an empty
      domain gets a greyed-out 'no data' marker.

  fig_exp4_class_stack_<rubric_id>.png
      Stacked bar of the agreement-class taxonomy (fully_agree /
      majority_agree / split / full_disagree / skipped) per clinical domain.
      All 5 classes are always represented in the legend, even at 0%.

  fig_exp4_boxplot_by_domain.png
      Combined 2x3 multi-panel box plot for the 5 rubrics.

  fig_exp4_class_summary.png
      Single bar plot summarising agreement-class distribution across all rows
      per rubric — gives a one-glance answer to RQ1/RQ2 at the panel level.

The two non-LLM JSON exports keep their original names so anything downstream
of `results/exp4_boxplot_data*.json` keeps working.

Run on HPC after Exp2 completes:
    python experiments/exp4_boxplot_agreement.py
or with the real-LLM config:
    EXP_CONFIG=config/configs/config_exp4_realllm.json \\
    EXP2_RESULTS_PATH=results/exp2_agreement_results_realllm.json \\
    python experiments/exp4_boxplot_agreement.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH       = Path(os.environ.get(
    'EXP_CONFIG',
    str(ROOT / 'config' / 'configs' / 'config_exp4_boxplots.json'),
))
EXP2_RESULTS_PATH = Path(os.environ.get(
    'EXP2_RESULTS_PATH',
    str(ROOT / 'results' / 'exp2_agreement_results.json'),
))


AGREEMENT_CLASSES = ['fully_agree', 'majority_agree', 'split',
                     'full_disagree', 'skipped']
CLASS_COLORS = {
    'fully_agree':    '#2A9D8F',
    'majority_agree': '#8AB17D',
    'split':          '#E9C46A',
    'full_disagree':  '#E76F51',
    'skipped':        '#999999',
}


def _safe_scores(cat_scores: Dict[str, List[float]], dom: str) -> List[float]:
    return cat_scores.get(dom, [])


def _plot_rubric_block(ax, data_to_plot, domains, box_colors, rubric_name,
                       has_data_flags):
    import numpy as np

    positions  = [i + 1 for i, d in enumerate(data_to_plot) if d]
    plot_data  = [d for d in data_to_plot if d]

    if plot_data:
        bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                        notch=False, widths=0.55)
        colors_used = [box_colors[i] for i, d in enumerate(data_to_plot) if d]
        for patch, color in zip(bp['boxes'], colors_used):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for i, (line, data) in enumerate(zip(bp['medians'], plot_data)):
            pos      = positions[i]
            median_v = float(np.median(data))
            mean_v   = float(np.mean(data))
            ax.text(pos, median_v + 1.5, f'med={median_v:.1f}',
                    ha='center', fontsize=7)
            ax.text(pos, mean_v - 4.0, f'\u03bc={mean_v:.1f}',
                    ha='center', fontsize=7, color='#555')

    for i, d in enumerate(data_to_plot):
        if not d:
            ax.text(i + 1, 50, 'no data', ha='center', va='center',
                    fontsize=7, color='#aaa',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#f5f5f5', ec='#ccc'))

    ax.set_xticks(range(1, len(domains) + 1))
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(80, color='red', linestyle='--', linewidth=0.8,
               alpha=0.6, label='80% threshold')
    ax.grid(axis='y', linestyle='--', alpha=0.45)


def _plot_class_stack(ax, class_by_domain: Dict[str, Dict[str, int]],
                      domains: List[str], rubric_name: str):
    """Stacked bar of agreement-class counts per domain. Always includes all
    five classes in the legend, even when counts are 0."""
    import numpy as np
    x  = np.arange(len(domains))
    bottoms = np.zeros(len(domains))
    for cls in AGREEMENT_CLASSES:
        heights = np.array([class_by_domain.get(dom, {}).get(cls, 0)
                            for dom in domains], dtype=float)
        ax.bar(x, heights, bottom=bottoms,
               color=CLASS_COLORS[cls], edgecolor='white',
               width=0.7, label=cls)
        for xi, h, b in zip(x, heights, bottoms):
            if h > 0:
                ax.text(xi, b + h / 2, f'{int(h)}',
                        ha='center', va='center', fontsize=8, color='white')
        bottoms += heights
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_ylabel('Row count', fontsize=9)
    ax.set_title(rubric_name, fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=7, ncol=2)


def _plot_global_class_summary(ax, exp2_data, domains):
    """Bar plot: % of each agreement class per rubric, aggregated across all
    domains. Always includes all 4 classes (skipped excluded from %)."""
    import numpy as np
    rubrics = [b['rubric_name'].split('\u2014')[0].strip()[:18]
               for b in exp2_data]
    x = np.arange(len(rubrics))
    width = 0.18
    bar_classes = ['fully_agree', 'majority_agree', 'split', 'full_disagree']
    for i, cls in enumerate(bar_classes):
        heights = []
        for blk in exp2_data:
            live = [r for r in blk['results']
                    if r.get('agreement_class') != 'skipped']
            tot = len(live) or 1
            n = sum(1 for r in live if r.get('agreement_class') == cls)
            heights.append(n / tot * 100)
        ax.bar(x + (i - 1.5) * width, heights, width,
               color=CLASS_COLORS[cls], label=cls, edgecolor='white')
        for xi, h in zip(x + (i - 1.5) * width, heights):
            if h > 0:
                ax.text(xi, h + 1, f'{h:.0f}%', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(rubrics, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('% of rows', fontsize=10)
    ax.set_title('Agreement-class distribution per rubric (RQ1 / RQ2)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if not EXP2_RESULTS_PATH.exists():
        print(f'Exp2 results not found at {EXP2_RESULTS_PATH}. Run exp2 first.')
        sys.exit(1)

    with open(EXP2_RESULTS_PATH) as f:
        exp2_data = json.load(f)

    fig_dir    = ROOT / config['output_files']['figure_dir']
    fig_dir.mkdir(parents=True, exist_ok=True)
    domains    = config.get('categories') or config.get('domains') or \
                 ['Cardiology', 'Pharmacology', 'Neurology',
                  'Pediatrics', 'Emergency']
    dpi        = config.get('figure_dpi', 300)
    box_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    boxplot_data: List[Dict] = []

    # ------------------------------------------------------------------
    # Per-rubric figures
    # ------------------------------------------------------------------
    for rubric_block in exp2_data:
        rubric_id   = rubric_block['rubric_id']
        rubric_name = rubric_block['rubric_name']
        results     = rubric_block['results']

        live = [r for r in results if r.get('agreement_class') != 'skipped']
        cat_scores: Dict[str, List[float]] = defaultdict(list)
        cat_classes: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {c: 0 for c in AGREEMENT_CLASSES})

        for r in live:
            cat = r.get('question_category', 'unknown')
            cat_scores[cat].append(r['mean_pairwise_agreement'])
        for r in results:
            cat = r.get('question_category', 'unknown')
            cat_classes[cat][r.get('agreement_class', 'skipped')] += 1

        data_to_plot = [_safe_scores(cat_scores, dom) for dom in domains]
        has_data     = [bool(d) for d in data_to_plot]

        # 1. Box plot
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_rubric_block(ax, data_to_plot, domains, box_colors,
                           rubric_name, has_data)
        ax.set_ylabel('Mean Pairwise Agreement (%)', fontsize=11)
        ax.set_title(f'Inter-Judge Agreement by Clinical Domain\n{rubric_name}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        plt.tight_layout()
        fp = fig_dir / f'fig_exp2_agreement_{rubric_id}.png'
        fig.savefig(fp, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fp}')

        # 2. Class stack per domain
        fig, ax = plt.subplots(figsize=(11, 5))
        _plot_class_stack(ax, cat_classes, domains, rubric_name)
        plt.tight_layout()
        fp = fig_dir / f'fig_exp4_class_stack_{rubric_id}.png'
        fig.savefig(fp, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fp}')

        boxplot_data.append({
            'rubric_id':       rubric_id,
            'rubric_name':     rubric_name,
            'domain_scores':   {dom: cat_scores.get(dom, []) for dom in domains},
            'domain_classes':  {dom: dict(cat_classes.get(dom, {}))
                                for dom in domains},
        })

    # ------------------------------------------------------------------
    # Combined per-rubric box plot multi-panel
    # ------------------------------------------------------------------
    n = max(1, len(exp2_data))
    cols = 3 if n > 2 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = (axes.flatten() if hasattr(axes, 'flatten')
                 else [axes])

    for idx, rubric_block in enumerate(exp2_data):
        ax          = axes_flat[idx]
        rubric_name = rubric_block['rubric_name']
        results     = rubric_block['results']
        live        = [r for r in results if r.get('agreement_class') != 'skipped']
        cat_scores  = defaultdict(list)
        for r in live:
            cat = r.get('question_category', 'unknown')
            cat_scores[cat].append(r['mean_pairwise_agreement'])
        data_to_plot = [_safe_scores(cat_scores, dom) for dom in domains]
        has_data     = [bool(d) for d in data_to_plot]
        _plot_rubric_block(ax, data_to_plot,
                           [d[:5] for d in domains],
                           box_colors, rubric_name, has_data)
        ax.set_ylabel('Mean PW (%)', fontsize=8)
        ax.set_title(rubric_name.split('\u2014')[0].strip(),
                     fontsize=9, fontweight='bold')

    for k in range(len(exp2_data), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        'Inter-Judge Agreement by Clinical Domain — All Rubrics',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    combined_path = fig_dir / 'fig_exp4_boxplot_by_domain.png'
    fig.savefig(combined_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined figure: {combined_path}')

    # ------------------------------------------------------------------
    # Cross-rubric agreement-class summary
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_global_class_summary(ax, exp2_data, domains)
    plt.tight_layout()
    fp = fig_dir / 'fig_exp4_class_summary.png'
    fig.savefig(fp, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fp}')

    out_path = ROOT / config['output_files']['results_json']
    with open(out_path, 'w') as f:
        json.dump(boxplot_data, f, indent=2)
    print(f'Boxplot data saved -> {out_path}')


if __name__ == '__main__':
    main()
