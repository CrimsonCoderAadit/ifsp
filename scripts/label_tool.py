"""Local keyboard-driven route label review; Enter saves and advances."""
from __future__ import annotations

import csv
import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1] / "data/real"
LABELS = ROOT / "labels.csv"
FIELDS = ["source", "file", "route_number", "board_type", "view", "lighting", "notes", "suggested", "confirmed"]


def images():
    return sorted(path for path in ROOT.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def rows():
    if not LABELS.exists():
        return {}
    with LABELS.open(newline="", encoding="utf-8") as stream:
        return {row["file"]: row for row in csv.DictReader(stream)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/img/"):
            relative = unquote(self.path[5:])
            path = (ROOT / relative).resolve()
            if ROOT not in path.parents or not path.is_file():
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
            self.wfile.write(path.read_bytes()); return
        data = [{"source": path.relative_to(ROOT).parts[0], "file": str(path.relative_to(ROOT)),
                 **rows().get(str(path.relative_to(ROOT)), {})} for path in images()]
        page = """<!doctype html><meta name=viewport content='width=device-width'><title>Bus route labels</title>
<style>body{font:20px sans-serif;max-width:900px;margin:auto;padding:20px}img{max-width:100%;max-height:65vh}input,select,button{font:inherit;padding:8px;margin:6px}</style>
<h1>Route labels</h1><p id=count></p><img id=photo alt='Bus to label'><form id=form>
<label>Route number <input name=route_number autofocus placeholder='blank if unreadable'></label>
<label>Board <select name=board_type><option>none</option><option>led</option><option>printed</option><option>paper_placard</option></select></label>
<label>View <select name=view><option>front</option><option>three_quarter</option><option>side</option><option>rear</option></select></label>
<label>Lighting <select name=lighting><option>day</option><option>dusk</option><option>night</option></select></label>
<label>Notes <input name=notes></label><button>Save + next (Enter)</button></form>
<script>const items=DATA;let i=0;const f=document.querySelector('form');function show(){const x=items[i];if(!x){document.body.innerHTML='<h1>All images reviewed</h1>';return}document.querySelector('#count').textContent=(i+1)+' / '+items.length+' — '+x.file;document.querySelector('#photo').src='/img/'+encodeURIComponent(x.file);for(const e of f.elements)if(e.name)e.value=x[e.name]||e.options?.[0]?.value||'';f.route_number.focus()}f.onsubmit=async e=>{e.preventDefault();const x=items[i],row={source:x.source,file:x.file,...Object.fromEntries(new FormData(f)),suggested:'0',confirmed:'1'};await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(row)});i++;show()};show()</script>""".replace("DATA", json.dumps(data))
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(page.encode())

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404); return
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        existing = rows(); existing[row["file"]] = {key: row.get(key, "") for key in FIELDS}
        with LABELS.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(existing.values())
        self.send_response(204); self.end_headers()


if __name__ == "__main__":
    print("Open http://127.0.0.1:8765")
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
