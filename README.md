# A-R-G-O-S (Work in progress...)

## Overview

So many different ways to approach this, each lab throughout the world has different methods that each believe capable of true generalization if scaled up with the right data VLAs, LBHs, Physics Engines, World Models and more...

**Now, it is quite hard to predict which approach will win, but we can already predict that the winner will be the one having most data that it can learn from.**

And knowing this, my project will aim at building a model, whatever it is, that is as efficient as possible to learn from the widest range possible of data.
Out of my head, I am thinking of:
- Images
- Environment Videos
- Human Videos
    - All kind of camera angle
    - Demonstration of task
- Robot Video
    - All kind of camera angle
    - Demonstration of task
- Other sensors
    - Force/torque feedback
    - Proprioception
    - Audio
    - Accelerometer
    - Tactile
    - Temperature, Wind speed
- Text
    - Task instructions
    - Planning and reasoning
- Simulation/Real Environment for continual learning, with trials and errors

And I am probably missing some, but the goal is simple, learn the distribution of as much data and modality as we can.

---

**I have defined the objective, let's now dive into the technical plan.**

Today, the most capable framework for learning from *multimodal* and *inherently stochastic* data is Diffusion/Flow matching, but other methods exist like Conditional VAE (used in the ACT) and energy-based models for example.

However, this long list of possible learned data doesn't have same impact on the modeling of the world, I would safely say that image sensors are giving more information than Wind Speed for robotic automation.
I will start my project by leveraging Images, Videos, Demonstrations Videos, Robot Demonstrations Videos, Task Instructions and Proprioception.

Images -> Learning spatial information
Videos -> Learning dynamic information
Demonstrations Videos -> Learning behaviors information
Robot Demonstrations Videos -> Learning behaviors information + proprioception information
Task Instructions -> Learning the mapping from text to behavior

Here is how I think about my model during inference:

Encode images using Dinov3
Encode text instructions using Dinov3 text encoder + CLIP + T5M
Encode proprioception using simple MLP
Flow matching DiT predict next step encoded data (image + proprioception)
Decoder of proprioception to get actions.
Decoder of imags (optional, for visualization purpose)

Extremely simple. And I think it is very close to what Toyota Research Institute has published recently, [LBMs](https://arxiv.org/pdf/2504.02792), or another paper called [Video Generation as Robot policy](https://arxiv.org/pdf/2508.00795).

---

## Model Architecure
It is a DiT backbone with added cross-attention to:
- Text tokens
- Action tokens
- Context tokens
(Those are all trained with Classifier Free Guidance)

## Timeline
- [x] DINO encoder for image and text loaded from torch hub.
- [x] Create the Image and text pairs dataset streamed from huggingface
- [ ] Train Decoder on Image dataset as an Auto-Encoder to be able to have visual interpretaion of futur predictions.
- [ ] Pre-train the flow matching Dit using the Image dataset. Exactly like a text to image generation model.
- [ ] Create the video and instructions pairs dataset streamed from hugginface
- [ ] Train the flow matching Dit using the Video dataset.
- [ ] Create the video and actions pairs dataset, streamed from hugginface
- [ ] Continue flow matching DiT training by adding action modality too.

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