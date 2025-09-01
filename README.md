# A-R-G-O-S  
**Solving Physical Intelligence**  

---

## Overview  
A-R-G-O-S is a research project aimed at solving **physical intelligence** by bridging **planning** and **actioning** through generative models and optimization.  

The approach consists of two main steps:  
1. **Planning** – generating high-level trajectories of future states.  
2. **Actioning** – translating those trajectories into executable actions optimized at inference time.  

---

## 1. Planning  
- **Representation:**  
  - All images are treated as **states**, represented as embeddings from **DINOv3**.  

- **Text Predictor:**  
  - Implemented with a **diffusion model**, trained autoregressively in the same way as video generation.  
  - **Conditioning:** The model is conditioned on **text instructions** to guide the generated plan.  

---

## 2. Actioning  
- **Goal:**  
  - Optimize actions to align with the generated plan.  

- **Action Predictor:**  
  - Implemented with a **diffusion model**, trained autoregressively similar to video generation.  
  - **Conditioning:** The model is conditioned on **joint values**.  

- **Optimization:**  
  - Use **CEM (Cross-Entropy Method)** or other optimization algorithms to select the best actions, ensuring that the generated future states match the planned trajectory.  

---

## To Be Determined  
- Length of each **planning horizon** and **actioning horizon**.  
