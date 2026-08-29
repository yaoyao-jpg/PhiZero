# Reconstruction examples

Each entry in `cases.json` contains only a case ID and a relative source-video
path. The six source clips are 896x512, 33 frames, and 8 FPS.

Reconstruction extracts the 256-symbol Physical Language from each source clip
and renders it using that clip's first frame as the appearance condition.
Generated symbols and videos are written under
`outputs/reconstruction/`.
