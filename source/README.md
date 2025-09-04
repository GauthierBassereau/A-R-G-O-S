# A-R-G-O-S

## Overview  
A-R-G-O-S is a research project aimed at solving **physical intelligence** by bridging **planning** and **actioning** through generative models and optimization.  

The approach consists of two main steps:  
1. **Planning** – generating high-level trajectories of future states.
2. **Actioning** – translating those trajectories into executable actions optimized at inference time.

All images are treated as **states**, represented as embeddings from **DINOv3**

## 1. Planning

- **Text Predictor:**
  - Implemented with a **diffusion model** DiT, trained autoregressively in the same way as video generation.
  - **Conditioning:** The model is conditioned on **text instructions** to guide the generated plan.
  - Timestep between predcictions: 0.25s (Could be interesting to make it bigger...)

- **Data**
    - Raw videos with text description of the instruction for the task being done in the video

## 2. Actioning  
- **Goal:**  
  - Optimize actions to align with the generated plan.

- **Action Predictor:**  
  - Fine-tuned version of the Text predictor
  - **Conditioning:** The model is conditioned on **joint values**.
  - Timestep between predcictions: 0.25s (Could be interesting to make it smaller...)

- **Optimization:**  
  - Use **CEM (Cross-Entropy Method)** or other optimization algorithms to select the best actions, ensuring that the generated future states match the planned trajectory.

- **Data**
    - Raw videos with joint values (Can be completly random trajectoris, no real need for special demonstrations)

---

## Data to explore
- **Robot Data with instructions:**
    - AGIBOTWORLD
    - Open X Embodiment
    - Droid
    - RobotMind
    - SO100 Community

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