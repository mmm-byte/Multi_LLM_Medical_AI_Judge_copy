from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent

EXP2_RESULTS_PATH = ROOT / "results" / "exp2_agreement_results.json"
EXP3_SUMMARY_PATH = ROOT / "results" / "exp3_summary_table.csv"
OUTPUT_DIR = ROOT / "results"

RUBRIC_LABELS = {
    "rubric1_pemat": "PEMAT",
    "rubric2_healthbench": "HealthBench",
    "rubric3_clinical_eval": "Clinical Eval",
    "rubric4_prometheus": "Prometheus",
    "rubric5_pemat_likert": "PEMAT-Likert",
}

AGREEMENT_ORDER = ["full_disagree", "split", "majority_agree", "fully_agree"]
AGREEMENT_SCORE_MAP = {"full_disagree": 0, "split": 1, "majority_agree": 2, "fully_agree": 3}


def load_exp2_dataframe():
    with open(EXP2_RESULTS_PATH) as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data

    rows = []
    for r in results:
        rubric_id = r.get("rubric_id")
        domain = r.get("domain")
        agreement_class = r.get("panel_agreement_class") or r.get("agreement_class")
        pairwise = r.get("pairwise_agreement")

        mean_pw = None
        if isinstance(pairwise, list) and pairwise:
            if isinstance(pairwise[0], dict):
                vals = [p.get("agreement") for p in pairwise if p.get("agreement") is not None]
            else:
                vals = [p for p in pairwise if p is not None]
            if vals:
                mean_pw = sum(vals) / len(vals) * 100

        if agreement_class is None or rubric_id is None:
            continue

        rows.append({
            "rubric_id": rubric_id,
            "rubric_label": RUBRIC_LABELS.get(rubric_id, rubric_id),
            "domain": domain,
            "agreement_class": agreement_class,
            "mean_pairwise_pct": mean_pw,
            "agreement_score": AGREEMENT_SCORE_MAP.get(agreement_class),
        })

    return pd.DataFrame(rows)


def main():
    df = load_exp2_dataframe()
    print(f"Loaded {len(df)} question-rubric results from Exp2")

    df_valid = df.dropna(subset=["mean_pairwise_pct"])

    fig1 = px.box(
        df_valid,
        x="rubric_label",
        y="mean_pairwise_pct",
        color="rubric_label",
        points="outliers",
        title="Mean Pairwise Judge Agreement by Rubric (Full Dataset)<br><span style=font-size:18px;font-weight:normal>Source: Exp2 results | Prometheus and Clinical Eval show tighter agreement</span>",
    )
    fig1.update_xaxes(title_text="Rubric")
    fig1.update_yaxes(title_text="Agreement %")
    fig1.update_layout(showlegend=False)
    fig1.write_image(str(OUTPUT_DIR / "exp4_boxplot_by_rubric.png"))
    with open(str(OUTPUT_DIR / "exp4_boxplot_by_rubric.png.meta.json"), "w") as f:
        json.dump({"caption": "Pairwise Judge Agreement by Rubric",
                   "description": "Boxplot of mean pairwise agreement percentage across all judge pairs, grouped by rubric"}, f)

    class_counts = (
        df.groupby(["rubric_label", "agreement_class"])
        .size()
        .reset_index(name="count")
    )
    class_pct = class_counts.copy()
    totals = class_counts.groupby("rubric_label")["count"].transform("sum")
    class_pct["pct"] = 100 * class_pct["count"] / totals

    fig2 = px.bar(
        class_pct,
        x="rubric_label",
        y="pct",
        color="agreement_class",
        category_orders={"agreement_class": AGREEMENT_ORDER},
        title="Panel Agreement Class Distribution by Rubric (Full Dataset)<br><span style=font-size:18px;font-weight:normal>Source: Exp2 results | Split outcomes dominate across rubrics</span>",
    )
    fig2.update_xaxes(title_text="Rubric")
    fig2.update_yaxes(title_text="Share of Questions %")
    fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title=None))
    fig2.write_image(str(OUTPUT_DIR / "exp4_agreement_class_distribution.png"))
    with open(str(OUTPUT_DIR / "exp4_agreement_class_distribution.png.meta.json"), "w") as f:
        json.dump({"caption": "Agreement Class Distribution by Rubric",
                   "description": "Stacked bar chart of fully_agree/majority_agree/split/full_disagree proportions per rubric"}, f)

    if df_valid["domain"].notna().any():
        fig3 = px.box(
            df_valid,
            x="domain",
            y="mean_pairwise_pct",
            color="domain",
            points="outliers",
            title="Mean Pairwise Judge Agreement by Clinical Domain<br><span style=font-size:18px;font-weight:normal>Source: Exp2 results | Domain-level agreement comparison</span>",
        )
        fig3.update_xaxes(title_text="Domain")
        fig3.update_yaxes(title_text="Agreement %")
        fig3.update_layout(showlegend=False)
        fig3.write_image(str(OUTPUT_DIR / "exp4_boxplot_by_domain.png"))
        with open(str(OUTPUT_DIR / "exp4_boxplot_by_domain.png.meta.json"), "w") as f:
            json.dump({"caption": "Pairwise Judge Agreement by Clinical Domain",
                       "description": "Boxplot of mean pairwise agreement percentage grouped by clinical domain (RQ4)"}, f)
        print("Saved domain-level boxplot (RQ4)")

    plot_data_records = df.to_dict(orient="records")
    with open(str(OUTPUT_DIR / "exp4_boxplot_data.json"), "w") as f:
        json.dump(plot_data_records, f, indent=2)

    print("\nSaved figures and data to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
