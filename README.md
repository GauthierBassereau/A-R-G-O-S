# A-R-G-O-S (Work in progress...)

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
It is a DiT backbone with added cross-attention to:
- Text tokens
- Action tokens
- Context tokens
(Those are all optional)

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
- Do a POC with AGIBOTWORLD dataset for pre-training planning predictor and will use SO-100 dataset for action predictor and fine-tuning the planning predictor.

## Improvements for next versions
- Add history/memory -> adding past frames to the predictors
    - Use relative position embedding.
    - Use some kind of memory bank that gets updated when a new token doesn't match anything in the memory bank / Use PCA to get rid of uselss tokens (see DINO Foresight)
- Generate the plan on mutiple scale
    (1) Goal Image
    (2) Generate one intermediate step bewteen current state and Goal Image
    (3) Redo (2) X times
    (4) Optimize actions to reach the intermediates states
- Train the Action Predictor entirely in simulation with varying background -> showing that the action predictor can be train with almost no cost.
- Replace Teacher Forcing by Self Forcing