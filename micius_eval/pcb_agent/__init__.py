"""PCB Agent: end-to-end KiCad engineering tasks (30 tasks).

Generation runs the PCB engineering agent harness (Kimi Code CLI 1.50.0 plus
atomic KiCad/PCB tools) inside the pinned evaluation image, one container per
task. Grading is a separate, independent three-axis review (answer /
structure / completion) by a Codex agent inside an isolated KiCad 10.0.6
container; a task counts as solved only if all three axes PASS.
"""

import os

GENERATION_IMAGE = os.environ.get("MICIUS_PCB_IMAGE", "ghcr.io/zhiman-bj/pcb-agent-env:v34")
VERIFIER_IMAGE = os.environ.get("MICIUS_PCB_VERIFIER_IMAGE", "ghcr.io/zhiman-bj/pcb-agent-verifier:kicad-10.0.6")
