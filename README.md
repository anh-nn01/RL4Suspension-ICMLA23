# RL4Suspension-ICMLA23

This is the official PyTorch implementation of the paper: https://ieeexplore.ieee.org/document/10459806. 

The arxiv version is available at https://arxiv.org/abs/2408.08425.

**Physics-Guided Reinforcement Learning System for Realistic Vehicle Active Suspension Control** 

Authors: ***Anh N. Nhu, Ngoc-Anh Le, Shihang Li, Thang D.V. Truong***

Proceedings of 22nd International Conference on Machine Learning and Applications (IEEE ICMLA) 2023.

Code 
----
<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=github,vscode,py,pytorch" />
  </a>
</p>

* `config.py`: code containing configurations of vehicle's mechanical system (body mass, wheel mass, spring stiffness, etc); Neural Network architecture (number of neurons); training hyperparameters; and road excitation.
* `model.py`: Neural Network implementation for Q-Network (Critic) and Policy Network (Actor).
* `road_generator.py`: Python implementation of ISO-8608 random road profiles and simple sine-shape road for training and evaluation.
* `trainer.py`: Trainer to
  * (1) initialize Actor-Critic weights ($\theta_{\mu}$ & $\theta_{Q}$);
  * (2) sample buffered experience from the environment using Quarter Model ODE (Ordinary Differential Equation);
  * (3) back-propagate and optimize weights of Q-Network and Policy Network ($\nabla_{\theta_{Q}}$ & $\nabla_{\theta_{\mu}}$)
  * (4) update weights & save checkpoints.
* `execute_training.py`: contains implementation of Ornstein-Uhlenbeck noise, buffer stack, and code to execute the DDPG training framework.

Training and Evaluation
----
To execute the training process, please use either `execute_training.py` or `RL_Training.ipynb`. 

If you use `execute_training.py` to train model, please change checkpoint names for Q-Network and Policy Network in `execute_training.py`, and run the following command:
```
  python3 execute_training.py
```

Once you train the model (or use our pretrained model), you may find the `evaluation.ipynb` notebook to be helpful. Specifically, it shows visualization of comparative performance between our proposed DDPG-based Active Suspension Control to Passive System as well as the dynamically controlled stiffness and damping coefficient.

Trained Models
----
To have a quick start, please use the pretrained Actor/Critic in folder `./checkpoints`. Since the final model performance is sensitive to initialized weights, you can use these weights as the initialization for experimenting new ideas.

📑 Citation
----
If by any chance you find our paper or our code source helpful for your research / project(s), please kindly support us by citing our work. Your endorsement is particularly valuable to us, and we deeply appreciate it!

```
  @INPROCEEDINGS{10459806,
  author={Nhu, Anh N. and Le, Ngoc-Anh and Li, Shihang and Truong, Thang D.V.},
  booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Physics-Guided Reinforcement Learning System for Realistic Vehicle Active Suspension Control}, 
  year={2023},
  volume={},
  number={},
  pages={422-429},
  keywords={Damping;Suspensions (mechanical systems);Roads;Stability criteria;Control systems;Real-time systems;Vehicle dynamics;Deep Reinforcement Learning;Mechanical En-gineering;Active Suspension System;Vehicle;Vertical Dynamics},
  doi={10.1109/ICMLA58977.2023.00065}}

```
