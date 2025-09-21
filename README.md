# A-R-G-O-S (Work in progress...)

## Overview  

Research labs worldwide are exploring different paths toward generalization in robotics and AI: **VLAs, LBHs, Physics Engines, World Models, Diffusion Policies**, and more.  

It is difficult to predict which approach will ultimately succeed, but a safe assumption is:  
**the winning method will be the one capable of learning from the widest variety of modalities, at scale.**

My project, **A-R-G-O-S**, aims to do exactly this. The goal is to build a system that can:  

1. **Learn from large-scale, multimodal data** (vision, demonstrations, text, proprioception, etc.).  
2. **Separate planning from control**, with a high-level world model for planning and a robust low-level controller for executing motor commands.  
3. **Leverage both web-scale pretraining and simulation**.  

## Core Idea: Planning vs Control  

The key design choice:  
- **Planning World Model (Low Frequency)**  
  - Operates on **latent representations** (e.g., DINOv3 embeddings).  
  - Predicts **future observations** (mainly vision + proprioception latents).  
  - Plans at a **lower frequency** (e.g., ~800 ms between states).  
  - Learns from **web-scale multimodal data**: human videos, robot demonstrations, text instructions, task-centric datasets.  
  - Output: a sequence of **future target states** in latent space.  

- **Low-Level Controller (High Frequency)**  
  - Operates directly on **current state**, a short **history of past states**, and the **next target state** predicted by the planner.  
  - Runs at **much higher frequency** (robot control loop, e.g., 50–500 Hz).  
  - Outputs **motor torques** directly.  
  - Training:  
    - **Imitation learning** from demonstrations.  
    - **Massive scale simulation** for robustness and adaptation.  
    - **Potential continual learning** (unsolved, but core ambition).  
  - Analogy: similar to controllers used for humanoid locomotion policies (robust low-level reflex-like behavior).  

This separation provides two major benefits:  
1. **Scalability** – Planning leverages abundant web-scale data, while the controller focuses on domain-specific motor control.  
2. **Robustness** – The controller can generalize in varied environments, while the planner generalizes across tasks and modalities.  

---

## Technical Plan  

### Planning World Model  
- **Encoder**  
  - Images: **DINOv3** (ViT-Large)  
  - Text: **DINOv3 text encoder**  
  - Proprioception: **MLP**  
- **World Model**  
  - **Flow Matching DiT backbone**  
  - Cross-attention on:  
    - Text tokens  
    - Action tokens  
    - Context tokens  
  - **Classifier-Free Guidance** training  
- **Outputs**  
  - Predicts next **latent observation** (vision + proprioception).  
  - Can decode latent vision for visualization.  

### Low-Level Controller  
- **Inputs**  
  - Current state (vision + proprioception)  
  - Small history window of past states  
  - Target next state (from planner)  
- **Outputs**  
  - Motor torques
- **Training**  
  - Phase 1: Imitation learning (robot demos).  
  - Phase 2: Large-scale simulated training with randomized backgrounds, environments, and dynamics.  
  - Phase 3: Continual learning (online adaptation).  

## Task List  

- [x] DINO encoder for image and text (Torch Hub).  
- [x] Image–text dataset (HuggingFace streaming).  
- [x] Train Image Decoder as Auto-Encoder (latent visualization).  
- [ ] Pre-train Flow Matching DiT on Image dataset (text-to-image style).  
- [ ] Create Video–Instruction dataset (HuggingFace).  
- [ ] Train Flow Matching DiT on video data.
- Design & train Low-Level Controller (first imitation, then sim, then online). -> will detail this later.

---

## Data to Explore  
(Need to do more research on this)
**Task-Centric Robot Data:**  
- AGIBOTWORLD  
- Open X Embodiment  
- DROID  
- RobotMind  
- SO100 Community  
- BridgeData V2, Egodex, RoboVQA, HoloAssist, Ego4D  

**Human / General Videos:**  
- HowTo100M
- Large scale video datasets, Kinematics etc...

## Proof of Concept (POC)  

- Input: Current state + text instructions.  
- Planner: Predict next 2 seconds of latent states, auto-regressively (teacher forcing).  
- Controller: Use planned next state to generate torques.  
- Data: Start with AGIBOTWORLD/DROID/Ego datasets for pretraining; specialize with SO-100 for fine-tuning.  

## Roadmap for Improvements  

- **History / Memory**  
  - Memory bank / PCA filtering of redundant tokens (see *DINO Foresight*).  

- **Hierarchical Planning**  
  - (1) Predict Goal Image.  
  - (2) Generate intermediate states bridging current → closest goal.  
  - (3) Redo (2) until next goal is at a certain threshold from the current state.
  -> This could enable more efficient and better planning generation

- **Self-Forcing**  
  - Replace teacher forcing with self-forcing for long-horizon rollouts.  