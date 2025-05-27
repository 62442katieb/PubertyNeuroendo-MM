# convert_report.py
import markdown
from pathlib import Path

# 1) Read your Markdown
md_path = Path("report.md")
md_txt  = md_path.read_text()

# 2) Convert to HTML
html = markdown.markdown(
    md_txt,
    extensions=["tables", "fenced_code", "md_in_html"]
)

# 3) Wrap in basic HTML (so you can drop in the <head> scripts)
full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DNN Report</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{startOnLoad:true}});</script>
</head>
<body>
{html}
</body>
</html>
"""

# 4) Write it out
out_path = Path("report.html")
out_path.write_text(full_html)

print(f"Written {out_path.resolve()}")