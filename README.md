# A-R-G-O-S (Work in progress...)

## Overview

There are so many different ways to approach this problem. Research labs around the world each have their own methods, all claiming to be scalable paths toward generalization: VLAs, LBHs, Physics Engines, World Models, and more...

**It is difficult to predict which approach will ultimately succeed, but we can safely assume that the winning method will be the one capable of learning from the most diverse and abundant data.**

With this in mind, my project will focus on building a model that can efficiently learn from the widest possible range of data. Off the top of my head, I’m thinking of:

- Images  
- Environment videos  
- Human videos  
  - Various camera angles  
  - Task demonstrations  
- Robot videos  
  - Various camera angles  
  - Robot task demonstrations  
- Other sensors  
  - Force/torque feedback  
  - Proprioception  
  - Audio  
  - Accelerometer  
  - Tactile feedback  
  - Environmental (temperature, wind speed, etc.)  
- Text  
  - Task instructions  
  - Planning and reasoning  
- Simulation and real environments for continual learning through trial and error  

I’m probably missing some modalities, but the goal remains simple: **learn the distribution of as much data and as many modalities as possible.**

---

**Having defined the objective, let’s move to the technical plan.**

Today, the most capable framework for learning from *multimodal* and *inherently stochastic* data is Diffusion / Flow Matching. Other methods exist, such as Conditional VAEs (used in ACT) or energy-based models, but diffusion currently seems best suited.

Of course, not all modalities contribute equally to modeling the world. For example, image data provides far more useful information for robotic automation than wind speed. To start, I will focus on:

- Images  
- Videos  
- Human demonstration videos  
- Robot demonstration videos (with proprioception)  
- Task instructions  

Here’s how each contributes:  
- **Images** → capture spatial information  
- **Videos** → capture dynamics over time  
- **Human demonstration videos** → capture behavior patterns  
- **Robot demonstration videos** → capture behavior + proprioception  
- **Task instructions** → map natural language to behaviors  

---

**Model at inference (current design):**

- Encode images using **DINOv3**  
- Encode text instructions using **DINOv3 text encoder**  
- Encode proprioception using a simple **MLP**
- Use a **Flow Matching DiT** to predict the next-step encoded data (image + proprioception)  
- Decode proprioception to obtain actions  
- Optionally decode images for visualization  

This is a simple idea, and I believe it is close to what Toyota Research Institute recently published ([LBMs](https://arxiv.org/pdf/2504.02792)), or the paper [Video Generation as Robot Policy](https://arxiv.org/pdf/2508.00795), as well as Unitree’s more recent project [Unifolm-World-Model-Action](https://github.com/unitreerobotics/unifolm-world-model-action/tree/main).  

---

## Model Architecure
Image Encoder: Dinov3, ViT Large
World Model: DiT backbone with added cross-attention layer to:
- Text tokens
- Action tokens
- Context tokens
*Trained with Classifier Free Guidance*

## Task List
- [x] DINO encoder for image and text loaded from torch hub.
- [x] Create the Image and text pairs dataset streamed from huggingface
- [x] Train Decoder on Image dataset as an Auto-Encoder to be able to have visual interpretaion of futur predictions.
- [ ] Pre-train the flow matching Dit using the Image dataset. Exactly like a text to image generation model.
- [ ] Create the video and instructions pairs dataset streamed from hugginface
- [ ] Train the flow matching Dit using the Video dataset.
- [ ] Create the video and actions pairs dataset, streamed from hugginface
- [ ] Continue flow matching DiT training by adding action modality too.

---

## Data to explore (need to do my research because there are much more than this)
- **Task centric Robot Data:**
    - AGIBOTWORLD
    - Open X Embodiment
    - Droid
    - RobotMind
    - SO100 Community
    - BridgeData V2, Egodex, RoboVQA, HoloAssist, Ego4D

- **Raw videos of people doing things:**
    - HowTo100M

## POC Version
- Take current state + instructions
- Predict next 2 seconds of states, auto-regressively, taking the last generated state as input (Teacher Forcing)
- Do a POC with AGIBOTWORLD/DROID/Ego datasets for pre-training and will add/use-only SO-100 dataset for post-training.

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

---
# Current Results
---

## DINO encoder

---

## Laion-coco dataset streamed from huggingface

---

## Train Decoder 

---