# Online Self-Learning For Smart HVAC Control
This is the upgrade version of our publication <a href="https://ieeexplore.ieee.org/document/8914027">Online Self-Learning For Smart HVAC Control</a>, named OSLN+.<br>
<div align="center"><img src="./png/System.png" width="500" height="250"></div><br>

Compared with our old version published in IEEE SMC 2019, there are three majior difference in OSLN+:<br>

1. To prevent the setpoint from changing extremely and frequently, OSLN+ additionally learns to predict the long-term temperature convergence tendency for representing the stability of the corresponding setpoint. Here, S represents the environment state (i.e., temperature in the office), E represents the uncontrable environment factors which may affect the environment state, and A represents the controllable action (i.e., the AHU output air temperature setpoint).<br>
<div align="center"><img src="./png/Network.png" width="500" height="375" alt="Paris" class="center"></div><br>

2. OSLN+ is designed as an auto-encoder form with a latent predictor. Considering there are different sensor combinations in different target offices, this design increase the extensive ability so that it is possible to share/transfer the latent predictor in future research. (i.e., the encoder/decoder can be viewed as the "domain converter" if we apply transfer learning or meta-learning for OSLN+ in future research; each domain has its specifical encoder/decoder but shares the latent predictor.)<br>
<div align="center"><img src="./png/Lrecon.png" width="250" height="75" alt="Paris" class="center"></div><br>
3. Moreover, to better regularize the network during learning with limited data samples, we design a novel constraint loss to guide the physical prior for the network. We penalize the network when it predicts the future temperature using the wrong assumption that violates the physical rule (e.g., when the network considers the AHU temperature setpoint to be negative related to the output predicted future temperature, it receives a loss value from such constraint.) <br>
<div align="center"><img src="./png/Lgradient.png" width="500" height="125" alt="Paris" class="center"></div><br>

## About the code
The code simulates the control process of OSLN+ in the real-world environment using an NN-based environment model trained with dataset collected from a large scale office. For executing the control simulation, please run AIplayground.py. However, due to the property issue, we are not yet allowed to release the dataset; nevertheless, we provide the dataset discription instead. Please refere to: ./data_log/15A_dataset_from_real_env.<br>
<div align="center"><img src="./png/architecture.jpg" width="750" height="375" alt="Paris" class="center"></div><br>

## Requirement
-pandas 1.3.5<br>
-torch 1.11.0<br>
