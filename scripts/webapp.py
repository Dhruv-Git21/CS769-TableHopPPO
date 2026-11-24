"""TableHop demo portal: ask a question about a table, see the hop-by-hop path, the evidence and the answer.

usage: python scripts/webapp.py [--run results/runs/<h1 run>] [--port 8769] [--split dev]
then open http://127.0.0.1:8769 . Standard library HTTP server only (no extra dependencies). Binds to
localhost. Display only: never writes TEST_LOG, never used for model selection; questions come from DEV.

API: GET /api/info, GET /api/examples (results/demo/showcase.json), GET /api/table?id=<example id>,
     POST /api/ask {"id": <example id>, "question": <optional new question>, "compare": bool}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))

import demo

from tablehop.data.load import load_split
from tablehop.eval.metrics import exact_match, f1
from tablehop.eval.trace_viz import trace_example

ROOT = Path(__file__).resolve().parents[1]
STATE: dict = {}
LOCK = threading.Lock()  # one model, one request at a time


def table_json(ex) -> dict:
    return {"title": ex.table.title, "header": ex.table.header,
            "rows": [[{"text": c.text, "link": bool(c.link)} for c in row] for row in ex.table.rows],
            "n_passages": len(ex.passages)}


def logged(run: Path | None, qid: str) -> dict | None:
    """A baseline's answer from its DEV evaluation log (identical to what the reported numbers used)."""
    if run is None:
        return None
    cache = STATE.setdefault("logs", {})
    if run not in cache:
        f = run / "predictions_dev.jsonl"
        cache[run] = {r["id"]: r for r in map(json.loads, f.read_text().splitlines()) if r} if f.exists() else {}
    return cache[run].get(qid)


def ask(qid: str, question: str | None, compare: bool) -> dict:
    base = STATE["by_id"].get(qid)
    if base is None:
        raise KeyError(f"unknown example id {qid}")
    custom = bool(question and question.strip() and question.strip() != base.question)
    ex = demo.ask_example(base, question.strip(), int(time.time())) if custom else base
    t0 = time.time()
    with LOCK:
        tr = trace_example(STATE["pol"], ex, STATE["pool"], STATE["head"], STATE["rfn"])
        comps = []
        if compare and not custom:
            for name, run in (("B1 SFT (reads whole table)", STATE["b1"]),):
                r = logged(run, qid)
                if r:
                    comps.append({"method": name, "pred": r["pred"], "em": exact_match(r["pred"], base.answer),
                                  "source": f"dev evaluation log, {run.name}"})
            m0 = demo.compare_one(None, ex, STATE["head"])
            comps.append({"method": "M0 oracle (gold path)", "pred": m0["pred"], "em": m0["em"],
                          "source": "frozen answer head on the gold path"})
        elif compare and custom and STATE["b1"] is not None:
            b = demo.compare_one(STATE["b1"], ex)
            comps.append({"method": "B1 SFT (reads whole table)", "pred": b["pred"], "em": None,
                          "source": f"live, {STATE['b1'].name}"})
    tr["compare"], tr["custom"], tr["latency_s"] = comps, custom, round(time.time() - t0, 2)
    tr["table"] = table_json(ex)
    tr["em"] = None if custom else exact_match(tr["pred"], base.answer)
    tr["f1"] = None if custom else f1(tr["pred"], base.answer)
    tr["gold"] = None if custom else base.answer
    if custom:
        tr["gold_path"] = None
    return tr


PAGE = (Path(__file__).parent / "webapp.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[webapp] " + (fmt % args) + "\n")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/info":
            return self._json(STATE["info"])
        if u.path == "/api/examples":
            f = ROOT / "results/demo/showcase.json"
            out = json.loads(f.read_text()) if f.exists() else {}
            sq = ROOT / "examples/demo_questions.json"
            out["suggestions"] = {k: v for k, v in json.loads(sq.read_text()).items()
                                  if not k.startswith("_")} if sq.exists() else {}
            return self._json(out)
        if u.path == "/api/table":
            qid = parse_qs(u.query).get("id", [""])[0]
            ex = STATE["by_id"].get(qid)
            if ex is None:
                return self._json({"error": "unknown id"}, 404)
            return self._json({"id": qid, "question": ex.question, "gold": ex.answer, **table_json(ex)})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            return self._json(ask(req.get("id", ""), req.get("question"), bool(req.get("compare", True))))
        except KeyError as e:
            return self._json({"error": str(e)}, 404)
        except Exception as e:  # noqa: BLE001  show the error in the page instead of a dead request
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--b1")
    ap.add_argument("--split", default="dev", choices=["dev", "ood_dev"])
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()
    run = Path(a.run) if a.run else demo.newest_h1(Path(a.results_dir))
    b1s = sorted(Path(a.results_dir).glob("runs/b1_sft_hybridqa_s0_*"))
    b1 = Path(a.b1) if a.b1 else (b1s[0] if b1s else None)
    print(f"[webapp] loading model {run.name} ...", flush=True)
    cfg, pol, pool, head, rfn = demo.load_policy(run)
    exs = load_split(cfg["data"], a.split)
    meta = json.loads((run / "meta.json").read_text())
    STATE.update(pol=pol, pool=pool, head=head, rfn=rfn, b1=b1, by_id={e.id: e for e in exs},
                 info={"run": run.name, "method": meta.get("method"), "seed": meta.get("seed"),
                       "split": a.split, "n_examples": len(exs), "b1_run": b1.name if b1 else None,
                       "answer_head_sha256": getattr(head, "sha256", "")[:12],
                       "evidence_mode": getattr(head, "evidence_mode", "entity")})
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[webapp] ready: http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
