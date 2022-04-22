# Online Self-Learning For Smart HVAC Control
<img src="./png/System.png" width="500" height="250">
This is the upgrade version of our publication <a href="https://ieeexplore.ieee.org/document/8914027">Online Self-Learning For Smart HVAC Control</a>, named OSLN+.<br>
Compared with our old version published in IEEE SMC 2019, to prevent the setpoint from changing extremely and frequently, OSLN+ additionally learns to predict the long-term temperature convergence tendency for representing the stability of the corresponding setpoint. <br>
The code simulates the control process of OSLN+ in the real-world environment using an NN-based environment model trained with dataset collected from a large scale office.

## Requirement
-pandas 1.3.5<br>
-torch 1.11.0<br>

## About the code
<img src="./png/architecture.jpg" width="750" height="475">
### AI_playground.py 
For executing the control simulation, please run this file. With the default setting, the record file will appear in experimental_record.

### 





