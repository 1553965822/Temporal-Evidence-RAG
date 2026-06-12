from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import ensure_dir


def format_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def write_table(
    name: str,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    tex_path = out_dir / f"{name}.tex"
    json_path = out_dir / f"{name}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    rendered = [[format_value(cell) for cell in row] for row in rows]
    widths = [max(len(str(col)), *(len(row[i]) for row in rendered)) for i, col in enumerate(columns)]
    lines = [f"# {title}", ""]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    lines.extend([header, sep])
    for row in rendered:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    col_spec = "l" * len(columns)
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rendered:
        tex_lines.append(" & ".join(row) + " \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    json_path.write_text(
        json.dumps({"title": title, "columns": columns, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path), "json": str(json_path)}


def dataset_stats_from_config(config: dict) -> dict[str, Any]:
    rows = []
    for name, spec in config["datasets"].items():
        rows.append(
            [
                name,
                spec["total"],
                f"{spec['positive_label']} {spec['positive_count']}",
                f"{spec['negative_label']} {spec['negative_count']}",
                spec["split"],
            ]
        )
    return {
        "name": "dataset_stats",
        "title": "实验数据集统计",
        "columns": ["Dataset", "Samples", "Positive", "Negative", "Split"],
        "rows": rows,
    }


def export_dataset_stats_table(config: dict, output_dir: str | Path) -> list[dict[str, Any]]:
    table = dataset_stats_from_config(config)
    write_table(table["name"], table["title"], table["columns"], table["rows"], output_dir)
    return [table]
