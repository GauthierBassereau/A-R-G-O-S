# A-R-G-O-S

## Overview  
A-R-G-O-S is a research project aimed at solving **physical intelligence** by bridging **planning** and **actioning** through generative models and optimization.  

The approach consists of two main steps:  
1. **Planning** – generating high-level trajectories of future states.
2. **Actioning** – translating those trajectories into executable actions optimized at inference time.

All images are treated as **states**, represented as embeddings from **DINOv3**

## Model Pipeline
The world model is an auto-regressive generative model, conditioned on either text or actions.
1. World Model predicts a trajectory of next states, conditioned on text, this becomes the **objective**.
2. Use **CEM** algorithm and World Model conditioned on actions to optimize a trajectory of actions to be as close as possible to the **objective**.

## Model Architecure
It is a DiT backbone with added tokens to the Attention layer. It can be:
- Text tokens
- Action tokens
- Context tokens

---

## Data to explore
- **Task centric Robot Data:**
    - AGIBOTWORLD
    - Open X Embodiment
    - Droid
    - RobotMind
    - SO100 Community
    - BridgeData V2, Egodex, RoboVQA, HoloAssist, Ego4D

- **Raw videos of people doing things:**
    - HowTo100M

## Current Version
- Take current state + instructions
- Predict next 2 seconds of states, auto-regressively, taking the last generated state as input (Teacher Forcing)
- Optimize actions to match the generated states
- Do a POC with AGIBOTWORLD dataset for planning predictor and will use SO-100 dataset for action predictor

## Improvements for next versions
- Add history/memory -> adding past frames to the predictors
    - Use relative position embedding.
    - Use some kind of memory bank that gets updated when a new token doesn't match anything in the memory bank

- Generate the plan on mutiple scale
    (1) Goal Image
    (2) Generate one intermediate step bewteen current state and Goal Image
    (3) Redo (2) X times
    (4) Optimize actions to reach the intermediates states

- Train the Action Predictor entirely in simulation with very varying background so that the model generelizes very well

- Replace Teacher Forcing by Self Forcing

---

## Code Structure
- `datasets/`: Data loaders and samplers.
  - Planning uses AGIBOTWORLD (videos + text instructions).
  - Actioning uses SO-100 (states + joint values + instructions).
- `embeddings/`: Encoders for image (DINOv3) and text (T5/CLIP); optional caching utilities.
- `models/`: Core networks.
  - `planning/`: Text-conditioned DiT for next-state planning and autoregressive rollout.
  - `actioning/`: Action predictor/dynamics model conditioned on joint values.
- `optim/`: Optimization algorithms (e.g., CEM) to select actions matching planned trajectories; schedulers lr and diffusion
- `trainers/`: Training loops, evaluation, checkpointing for planner and action models.
- `pipelines/`: End-to-end plan-and-act orchestration and evaluation routines.
- `scripts/`: CLIs for data prep, embedding precompute, training, inference, and demos.
- `utils/`: Transforms, IO helpers, seeding, device and logging utilities.
- `tests/`: tests.
- `logs/`: Experiment outputs and metrics.

---
## Git commands so I don't forget aha

### Create a new branch
git pull origin main
git checkout -b feature/my-new-feature

### Merge the branch
git checkout main
git pull origin main
git merge feature/my-new-feature
git push origin main

### Cleanup
git branch -d feature/my-new-feature
git push origin --delete feature/my-new-feature


