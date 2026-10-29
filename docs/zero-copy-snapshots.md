# Zero-copy snapshots and `SIGBUS`

Status: recorded, no code change — Engram has no persistence layer yet.

## Context

`mmap` reserves address space eagerly and loads blocks on first touch. If the
backing inode shrinks (`ftruncate`, `O_TRUNC`) while a mapping is live, or a
page past EOF is touched, the kernel cannot back the virtual page and delivers
`SIGBUS`, which kills the process. User-space `sigaction` + `sigsetjmp` is not a
fix: it needs a jump buffer around every dereference, inhibits dead-store and
aliasing optimisation, and is unsound across thread migration.

## Finding in this repository

- `engram_pq.py` keeps `Bfp64HeadRow`, `Mem8AdaptiveHeadRow`, `Quantum64TritBlock`
  and the multi-head embeddings entirely in memory. There is no `save`, `load`,
  `dump`, or file read; `mem8_adaptive` is an embedding mode, not a MEM|8 file
  reader.
- `rust/` is `#![forbid(unsafe_code)]` with zero dependencies. `mmap` is
  unreachable there by construction: it needs a foreign call and would break
  both the forbid and the zero-dependency posture.

There is no file-backed snapshot to read, so no `SIGBUS` exposure exists today
and no code change is made.

## Invariant when node heads are persisted

Four layers, in this order:

1. **Immutable inode replacement** — new temporary inode, `fsync`, `0400`,
   then `rename`. Never truncate a live file.
2. **Unlinked-inode lifetime** — POSIX keeps a replaced inode's blocks alive
   until the last `munmap`; readers survive rotation.
3. **Pre-map `fstat` envelope** — physical `st_size` must equal the header's
   declared length before mapping.
4. **Prefault and verify** — `MADV_WILLNEED`, then footer digest verification
   at open.

Canonical, tested implementation: `8b-is/aurora` PR #5,
`tools/zero_copy_snapshot.py` (stdlib only, `AURSNAP1` format). A Python-side
adapter is the natural fit here; the Rust crate would need a separate opt-in
crate to keep `forbid(unsafe_code)` intact.
