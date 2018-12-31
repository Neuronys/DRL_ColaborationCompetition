[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

## Introduction

The Report notebook and all the Python files in this repository propose a solution using the Unity ML-Agents environment [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis), as modified by Udacity for the third project of the Udacity.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net.
- **Positive Rewards:** +0.1 if an agent hits the ball over the net.
- **Negative Rewards:** -0.01 if an agent lets a ball hit the ground or hits the ball out of bounds.
- **Observation Space:** 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.
- **Action Space:** Two continuous actions are available: 1) movement toward (or away from) the net, and 2) jumping. 

So, the **Goal** of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, our agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


### Dependencies :

To set up your Python environment to run the code in this repository, follow the instructions below.

  1. Create (and activate) a new environment with Python 3.6.
  
 ```
 conda create --name drlnd python=3.6
 activate drlnd
 ```
  2. Install Pytorch by following the instructions in the link below.
  
     https://pytorch.org/get-started/locally/
    
  3. Then navigate to DRLND_P3_collab-compet/ml-agents-0.4.0b/python and install ml-agent.
     ```
     pip install .
     ```
  4. Install matplotlib for plotting graphs.
     ```
     conda install -c conda-forge matplotlib
     ```
  5. (Optional) Install latest prompt_toolkit may help in case Jupyter Kernel keeps dying
     ```
     conda install -c anaconda prompt_toolkit 
     ```
     
## Run the code 

  Open **Report.ipynb** in Jupyter and press Ctrl+Enter to run the first cell to import all the libraries. You would have to modify the file `multi_agent.py` to change the hyperparameters and the file `model.py` to change the neural network topology.
  