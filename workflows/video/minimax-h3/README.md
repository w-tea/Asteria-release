# MiniMax H3 local workflows

These files are Asteria-owned, API-format ComfyUI workflow templates. They are
not copies of the ComfyUI graphical template. The local video adapter resolves
the named bindings into request values and artifact inputs before submission.

Verified upstream contracts (2026-08-20):

- ComfyUI native H3 guide: <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- Native nodes: `MiniMaxH3ImageToVideo` and `MiniMaxH3ReferenceToVideo`
- Minimum ComfyUI: `0.30.0`; Asteria managed pin: `0.33.1`
- The managed pin supports sequence continuation through synchronized
  reference video/audio conditioning. The newer native multi-frame guide node
  exists only after the managed release, so `direct_context` is not advertised
  until an immutable supported runtime release contains it.
- Local output grid: 24 fps, dimensions divisible by 32, duration represented
  by the H3 `17k+5` frame grid
- Local open-weight profile: the 768p Base/FL2VA and REF2VA weights only. The
  hosted H3 Context/Regenerate 2K offerings are not advertised as local.

The versioned node IDs live only in these workflow assets. Product services
select a workflow ID and apply its binding map; they do not hard-code nodes.
