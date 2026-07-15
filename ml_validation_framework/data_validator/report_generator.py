"""
report_generator.py
Generates a self-contained HTML validation report from a report dict.
"""
from typing import Dict
from pathlib import Path


class ReportGenerator:
    @staticmethod
    def save_html(report: Dict, path: str):
        html = ReportGenerator.build_html(report)
        with open(path, "w",encoding="utf-8") as f:
            f.write(html)

    @staticmethod
    def build_html(report: Dict) -> str:
        s = report["summary"]
        ts = report.get("timestamp", "")
        shape = report.get("data_shape", {})
        valid_badge = (
            '<span class="badge badge-pass">✓ VALID</span>'
            if s["valid"] else
            '<span class="badge badge-fail">✗ INVALID</span>'
        )

        errors_html = "".join(
            f'<div class="issue error"><b>ERROR</b> [{r["column"] or "global"}] {r["message"]}</div>'
            for r in report.get("errors", [])
        ) or "<p>No errors ✓</p>"

        warnings_html = "".join(
            f'<div class="issue warning"><b>WARNING</b> [{r["column"] or "global"}] {r["message"]}</div>'
            for r in report.get("warnings", [])
        ) or "<p>No warnings ✓</p>"

        cols_html = ""
        for col, stats in report.get("column_stats", {}).items():
            null_cls = "bad" if stats["null_pct"] > 5 else ""
            cols_html += f"""
            <tr>
              <td>{col}</td>
              <td>{stats["dtype"]}</td>
              <td class="{null_cls}">{stats["null_count"]} ({stats["null_pct"]}%)</td>
              <td>{stats.get("mean", "—")}</td>
              <td>{stats.get("std", "—")}</td>
              <td>{stats.get("min", "—")}</td>
              <td>{stats.get("max", "—")}</td>
            </tr>"""

        all_checks_html = ""
        for r in report.get("all_results", []):
            icon = "✓" if r["passed"] else ("✗" if r["severity"] == "error" else "⚠")
            cls = "pass" if r["passed"] else ("fail" if r["severity"] == "error" else "warn")
            all_checks_html += f"""
            <tr class="{cls}">
              <td>{icon}</td>
              <td>{r["check"]}</td>
              <td>{r["column"] or "—"}</td>
              <td>{r["message"]}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ML Data Validation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e2e8f0; }}
  .header {{ background: linear-gradient(135deg, #1a1f2e, #2d3748); padding: 2rem; border-bottom: 2px solid #4a5568; }}
  h1 {{ font-size: 1.8rem; color: #63b3ed; margin-bottom: .3rem; }}
  .meta {{ color: #718096; font-size: .9rem; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; padding: 1.2rem; text-align: center; }}
  .card .num {{ font-size: 2.4rem; font-weight: bold; margin: .4rem 0; }}
  .card .label {{ color: #718096; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; }}
  .num.green {{ color: #68d391; }} .num.red {{ color: #fc8181; }} .num.yellow {{ color: #f6e05e; }} .num.blue {{ color: #63b3ed; }}
  .badge {{ display: inline-block; padding: .3rem .9rem; border-radius: 20px; font-weight: bold; font-size: .9rem; margin-top: .5rem; }}
  .badge-pass {{ background: #22543d; color: #68d391; }}
  .badge-fail {{ background: #742a2a; color: #fc8181; }}
  section {{ background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; margin-bottom: 1.5rem; overflow: hidden; }}
  section h2 {{ padding: 1rem 1.5rem; background: #2d3748; font-size: 1rem; color: #a0aec0; text-transform: uppercase; letter-spacing: .05em; }}
  .issues {{ padding: 1rem 1.5rem; }}
  .issue {{ padding: .6rem .9rem; border-radius: 6px; margin-bottom: .5rem; font-size: .88rem; }}
  .issue.error {{ background: #2d1b1b; border-left: 4px solid #fc8181; }}
  .issue.warning {{ background: #2d2a1b; border-left: 4px solid #f6e05e; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #2d3748; padding: .7rem 1rem; text-align: left; color: #a0aec0; font-weight: 600; }}
  td {{ padding: .6rem 1rem; border-bottom: 1px solid #2d3748; }}
  tr.pass td {{ color: #68d391; }}
  tr.fail td {{ color: #fc8181; }}
  tr.warn td {{ color: #f6e05e; }}
  .bad {{ color: #fc8181 !important; }}
  tr:hover td {{ background: #2d3748; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ ML Data Validation Report</h1>
  <div class="meta">Generated: {ts} &nbsp;|&nbsp; Shape: {shape.get("rows","?")} rows × {shape.get("columns","?")} columns &nbsp;|&nbsp; {valid_badge}</div>
</div>
<div class="container">
  <div class="grid">
    <div class="card"><div class="num blue">{s["total_checks"]}</div><div class="label">Total Checks</div></div>
    <div class="card"><div class="num green">{s["passed"]}</div><div class="label">Passed</div></div>
    <div class="card"><div class="num red">{s["errors"]}</div><div class="label">Errors</div></div>
    <div class="card"><div class="num yellow">{s["warnings"]}</div><div class="label">Warnings</div></div>
  </div>

  <section>
    <h2>🔴 Errors</h2>
    <div class="issues">{errors_html}</div>
  </section>

  <section>
    <h2>⚠️ Warnings</h2>
    <div class="issues">{warnings_html}</div>
  </section>

  <section>
    <h2>📊 Column Statistics</h2>
    <table>
      <tr><th>Column</th><th>Type</th><th>Nulls</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
      {cols_html}
    </table>
  </section>

  <section>
    <h2>📋 All Checks</h2>
    <table>
      <tr><th></th><th>Check</th><th>Column</th><th>Message</th></tr>
      {all_checks_html}
    </table>
  </section>
</div>
</body>
</html>"""
