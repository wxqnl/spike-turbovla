# RoboTwin reference code

These files are inherited from [TurboVLA](https://github.com/H-EmbodVis/TurboVLA)
and use the shared model and the StarVLA-derived runtime under
`third_party/starvla_runtime/`.

The spike-turbovla experiments in this release only cover LIBERO. No RoboTwin
spiking-policy training, rollout score, or dependency compatibility is claimed.
This directory is retained as reference code, not as a validated spike recipe.

For the upstream ANN RoboTwin clean50 setup, data preparation, simulator, and
deployment instructions, use the upstream TurboVLA documentation. Its simulator
stack needs a separate environment from the pinned LIBERO environment.
