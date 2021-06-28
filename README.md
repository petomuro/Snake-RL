# Snake_RL
 Snake RL - Reinforcement Learning that solves the Snake game. SL was implemented by [Gradient-Free-Optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers) library available for Python, neural networks was created in [Keras](https://keras.io/) and game was created in [Pygame](https://www.pygame.org/news).

You have to installed [Python 3.7.9](https://www.python.org/downloads/release/python-379/) and other requirements which are necessary for run the program (requirements.txt).

![Example](https://github.com/petomuro/Snake_RL/blob/main/Game.PNG)

## Hyperparams optimization
If you want to find optimal hyperparams for neural network, you need to: 
  1. change `OPTIMIZATION = True`
  2. change `LOAD_WEIHTS = False`
  3. run `main.py`

![Example](https://github.com/petomuro/Snake_RL/blob/main/Training_final.png)

## Train neural network on optimal hyperparams
When the hyperparams optimization is complete, a files will be created in the logs and weights folder. 

If you want to train neural network on optimal hyperparams you need to:
  1. change `OPTIMIZATION = False`
  2. change `LOAD_WEIGHTS = False`
  3. search all `logs/scores_name.txt` files in logs folder and find the file with highest score --> e.g. `logs/scores_20210519081957.txt`
  4. change in `main.py` --> `best_para = {'no_of_layers': number, 'no_of_neurons': number, 'lr': number, 'gamma': number}` --> e.g. `best_para = {'no_of_layers': 2, 'no_of_neurons': 224, 'lr': 0.00001, 'gamma': 0.9400000000000001}` from file with the highest score
  5. run `main.py`

## Test trained neural network on optimal hyperparams
When the training is complete, a file will be created in the weights folder. 

If you want to test trained neural network on optimal hyperparams you need to:
  1. change `OPTIMIZATION = False`
  2. change `LOAD_WEIGHTS = True`
  3. change in `neural_network.py` --> `self.network.load_weights('weights/model_name.h5')` --> e.g. `self.network.load_weights('weights/model20210527043225.h5')`
  4. run `main.py`

## Create graph
When the training is complete, a file will be created in the results folder.

If you want to create graph you need to: 
  1. change in `helper.py` --> `with open('results/results_name.csv') as results_file:` --> e.g. `with open('results/results20210525162028.csv') as results_file:`
  2. run `helper.py`

## Task lists
- [ ] Refactor source code
