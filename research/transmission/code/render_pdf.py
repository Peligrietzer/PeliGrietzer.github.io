"""
render_pdf.py -- render ../transmission_note.md to ../transmission_note.pdf
(pandoc -> HTML -> headless Chromium print-to-PDF via Node Playwright).

Run:  python3 render_pdf.py [path/to/katex/dist]
  * with a KaTeX dist directory (npm install katex; .../node_modules/katex/dist) math is
    typeset by KaTeX (best quality);
  * without it, pandoc's MathML output is used.
Needs: pandoc on PATH; node with the 'playwright' package and a Chromium build.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MD = os.path.join(ROOT, "transmission_note.md")
HTML = os.path.join(ROOT, "transmission_note.html")
PDF = os.path.join(ROOT, "transmission_note.pdf")
CSS = os.path.join(HERE, "note.css")
JS = os.path.join(HERE, "render_pdf.js")


def main():
    katex = sys.argv[1] if len(sys.argv) > 1 else None
    if katex:
        url = "file://" + os.path.abspath(katex).rstrip("/") + "/"
        cmd = ["pandoc", MD, "-s", f"--katex={url}", "--css", "file://" + CSS,
               "--resource-path", ROOT, "-o", HTML]
    else:
        cmd = ["pandoc", MD, "-s", "--mathml", "--css", CSS, "--embed-resources",
               "--resource-path", ROOT, "-o", HTML]
    subprocess.run(cmd, check=True, cwd=ROOT)
    subprocess.run(["node", JS, HTML, PDF], check=True)
    print("wrote", PDF)


if __name__ == "__main__":
    sys.exit(main())
