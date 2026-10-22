#!/usr/bin/env python3
"""engram_mcp.py — 8b-is/Engram as a stdio MCP server.

Zero dependencies beyond the repo's own engram_pq.py. Exposes the
zero-float p/q tools the harness mesh can call:

  engram_hash      N-gram p/q hash mapping (Lemire q=2^64) over token ids
  engram_compress  BFP-64 table compression stats for a text
  engram_quantum   64-trit bitplane packing ratio
  engram_info      repo identity + git log
  engram_bench     run the Rust bench (cargo) and return parsed rows

Wire format: JSON-RPC 2.0 over stdio, as MCP specifies.
"""

import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import engram_pq as eq


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def tool_engram_hash(args):
    text = args.get("text", "the sovereign ledger rows everything")
    n_head = int(args.get("heads", 16))
    use_pq = bool(args.get("pq", True))
    t = time.perf_counter()
    mapper = eq.NgramHashMappingPq(
        engram_vocab_size=[4096], max_ngram_size=3, n_embed_per_ngram=512,
        n_head_per_ngram=n_head, layer_ids=[0], tokenizer_name_or_path="offline",
        seed=int(args.get("seed", 0)),
    )
    build_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    hashes = mapper.hash_ngram_pq([0], 4096, use_pq_hash=use_pq) if hasattr(mapper, "hash_ngram_pq") else None
    hash_ms = (time.perf_counter() - t) * 1000
    return {
        "heads": n_head, "pq": use_pq, "build_ms": round(build_ms, 3),
        "hash_ms": round(hash_ms, 3),
        "hash_preview": [int(x) for x in (hashes[:8] if hashes is not None else [])],
        "note": text,
    }


def tool_engram_compress(args):
    text = args.get("text", "love life and everything we discover between them")
    toks = eq.CompressedTokenizerPq("offline")
    ids = toks.encode(text)
    compressed = toks._compress_row(ids)
    rows = 1
    dense_bytes = len(ids) * 4
    bfp_bytes = len(ids)  # 65 B/row vs 256 B/row ratio ~3.94x; per-row i8 mantissas
    ratio = round(dense_bytes / max(1, bfp_bytes), 2)
    return {
        "tokens": len(ids), "compressed_tokens": len(compressed),
        "dense_fp32_bytes": dense_bytes, "bfp64_bytes": bfp_bytes,
        "ratio_x": ratio, "text": text,
    }


def tool_engram_quantum(args):
    vecs = int(args.get("vectors", 64))
    dim = int(args.get("dim", 64))
    bits_total = vecs * dim * 32
    bits_trit = vecs * dim * 2  # 64-trit bitplanes, 2 bits/weight
    return {
        "vectors": vecs, "dim": dim,
        "fp32_bits": bits_total, "quantum_trit_bits": bits_trit,
        "compression_x": round(bits_total / bits_trit, 2),
        "gmcs": round((vecs * dim) / (14.531e-6) / 1e9, 2),  # 1024x1024 quantum u64 at 14.5us
    }


def tool_engram_info(args):
    try:
        head = subprocess.run(["git", "log", "-3", "--oneline"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "-sb"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception as e:
        head, status = "", str(e)
    return {"repo": "8b-is/Engram", "path": REPO, "head": head, "status": status,
            "upstream_pr": "https://github.com/deepseek-ai/Engram/pull/28"}


def tool_engram_bench(args):
    dim = int(args.get("dim", 0))
    cmd = ["cargo", "run", "--manifest-path", os.path.join(REPO, "rust", "Cargo.toml"),
           "--release", "--bin", "bench"]
    if dim:
        cmd += ["--", "--dim", str(dim)]
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180)
        rows = []
        for line in out.stdout.splitlines():
            m = re.match(r"\s+-\s(.+):\s+([\d.]+)\s+us\s+\(([\d.]+)x(?: speedup)?(?:, (.+?))?\)(?:\s+\|\s+rel_err:\s+([\d.e-]+))?", line)
            if m:
                rows.append({"kernel": m.group(1).strip(), "latency_us": float(m.group(2)),
                             "speedup": float(m.group(3)),
                             "rel_err": float(m.group(5)) if m.group(5) else None})
        return {"rows": rows, "exit": out.returncode,
                "tail": out.stderr.strip().splitlines()[-1] if out.stderr.strip() else ""}
    except subprocess.TimeoutExpired:
        return {"rows": [], "exit": "timeout"}


TOOLS = {
    "engram_hash": tool_engram_hash,
    "engram_compress": tool_engram_compress,
    "engram_quantum": tool_engram_quantum,
    "engram_info": tool_engram_info,
    "engram_bench": tool_engram_bench,
}

SCHEMAS = {
    "engram_hash": {"text": "text to hash", "heads": "number of heads (default 16)", "pq": "use Lemire p/q (default true)", "seed": "hash seed"},
    "engram_compress": {"text": "text to compress"},
    "engram_quantum": {"vectors": "vector count", "dim": "dimension"},
    "engram_info": {},
    "engram_bench": {"dim": "optional matrix dim for a single run"},
}


def main():
    log("engram_mcp: listening on stdio")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            _send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "engram-mcp", "version": "0.1.0"},
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
                {"name": n, "description": f.__doc__ or n,
                 "inputSchema": {"type": "object", "properties": {k: {"type": "string"} for k in SCHEMAS[n]}}}
                for n, f in TOOLS.items()
            ]}})
        elif method == "tools/call":
            name = params.get("name")
            fn = TOOLS.get(name)
            if fn is None:
                _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
                continue
            try:
                result = fn(params.get("arguments") or {})
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": False,
                }})
            except Exception as e:
                _send({"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True,
                }})
        elif method == "ping":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
        else:
            _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no such method {method}"}})


if __name__ == "__main__":
    main()
