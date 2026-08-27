# Third-party notice

The AV-latent decode and synchronized boundary-crossfade behavior in this
component is adapted from the `H3 Motion Context Clip Stitcher` contributed to
`ComfyUI-noEmbryo` by noEmbryo.

- Upstream repository: https://github.com/noembryo/ComfyUI-noEmbryo
- Reviewed revision: `e0f6fd1622cf1fe9390beef5543b1be0d222df05`
- Source file: `stitcher.py`
- Upstream license: MIT

Asteria replaces the upstream folder/glob interface with an operation-owned,
hash-bound manifest containing only repository-validated accepted clips. The
node name, manifest validation, per-clip lineage checks, exact-path handling,
and Asteria adapter contract are Asteria-specific.
