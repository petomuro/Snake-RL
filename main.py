import math as m
from typing_extensions import final
import numpy as np
import random
from collections import deque
from datetime import datetime
from gradient_free_optimizers import HillClimbingOptimizer, StochasticHillClimbingOptimizer

from snake_game import SnakeGame
from helper import Helper
from neural_network import NeuralNetwork

# Change OPTIMIZATION to True if you want to optimize hyperparams
OPTIMIZATION = False
LOAD_WEIGHTS = True


class Optimization:
    def __init__(self):
        pass

    def optimize(self, param):
        # Print parameters
        print('Parameters:', param)

        # Optimization process is based on this variable
        score = run(param)

        # Save logs
        self.save_logs(param, score)

        return score

    def save_logs(self, param, score):
        with open('logs/scores_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(
                str('no_of_layers{}_no_of_neurons{}_snake_lr{}_gamma{}_score{}'.format(
                    int(
                        param['no_of_layers']),
                    param['no_of_neurons'],
                    param['lr'],
                    param['gamma'],
                    score)) + '\n')
            f.write('Params: ' + str(param) + '\n')


class Agent:
    def __init__(self, epsilon, epsilon_min, epsilon_decay, batch_size, gamma, memory, vectors_and_keys):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = memory
        self.vectors_and_keys = vectors_and_keys

    def get_state(self, game):
        _, _, food, snake, length = game.generate_observations()

        return self.generate_observation(snake, food, length, game)

    def get_food_distance(self, snake, food, length):
        # Euklidovská vzdialenosť
        return np.linalg.norm(self.get_food_distance_vector(snake, food, length))

    def get_food_distance_vector(self, snake, food, length):
        return np.array(food) - np.array(snake[length - 1])

    def add_action_to_observation(self, observation, final_move):
        return np.append([final_move], observation)

    def generate_observation(self, snake, food, length, game):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        food_distance_vector = self.get_food_distance_vector(
            snake, food, length)
        obstacle_front = self.get_obstacles(
            snake, snake_direction_vector, length, game)
        obstacle_right = self.get_obstacles(
            snake, self.turn_vector_to_the_right(snake_direction_vector), length, game)
        obstacle_left = self.get_obstacles(
            snake, self.turn_vector_to_the_left(snake_direction_vector), length, game)
        angle, snake_direction_vector_normalized, food_distance_vector_normalized = self.get_angle(
            snake_direction_vector, food_distance_vector, game)

        return np.array([int(obstacle_front), int(obstacle_right), int(obstacle_left), snake_direction_vector_normalized[0], food_distance_vector_normalized[0], snake_direction_vector_normalized[1], food_distance_vector_normalized[1], angle])

    def get_snake_direction_vector(self, snake, length):
        return np.array(snake[length - 1]) - np.array(snake[length - 2])

    def get_obstacles(self, snake, snake_direction_vector, length, game):
        point = np.array(snake[length - 1]) + np.array(snake_direction_vector)

        return point.tolist() in snake[:-1] or point[0] < 0 or point[1] < 0 or point[0] >= game.DISPLAY_WIDHT or point[1] >= game.DISPLAY_HEIGHT

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, snake_direction_vector, food_distance_vector, game):
        norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
        norm_of_food_distance_vector = np.linalg.norm(food_distance_vector)

        if norm_of_snake_direction_vector == 0:
            norm_of_snake_direction_vector = game.SNAKE_BLOCK

        if norm_of_food_distance_vector == 0:
            norm_of_food_distance_vector = game.SNAKE_BLOCK

        snake_direction_vector_normalized = snake_direction_vector / \
            norm_of_snake_direction_vector
        food_distance_vector_normalized = food_distance_vector/norm_of_food_distance_vector
        angle = m.atan2(food_distance_vector_normalized[1] * snake_direction_vector_normalized[0] - food_distance_vector_normalized[0] * snake_direction_vector_normalized[1],
                        food_distance_vector_normalized[1] * snake_direction_vector_normalized[1] + food_distance_vector_normalized[0] * snake_direction_vector_normalized[0]) / m.pi

        return angle, snake_direction_vector_normalized, food_distance_vector_normalized

    def generate_action(self, snake, length, observation, model):
        if np.random.rand() <= self.epsilon and OPTIMIZATION and LOAD_WEIGHTS == False:
            action = random.randint(0, 2) - 1

            final_move = self.get_game_action(snake, action, length)

            return final_move
        elif np.random.rand() <= self.epsilon and OPTIMIZATION == False and LOAD_WEIGHTS == False:
            action = random.randint(0, 2) - 1

            final_move = self.get_game_action(snake, action, length)

            return final_move
        else:
            final_move = np.argmax(np.array(model.predict(observation)))

        return final_move

    def get_game_action(self, snake, action, length):
        snake_direction_vector = self.get_snake_direction_vector(snake, length)
        new_direction = snake_direction_vector

        if action == -1:
            new_direction = self.turn_vector_to_the_left(
                snake_direction_vector)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(
                snake_direction_vector)

        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]

                return game_action

    def remember(self, observation, final_move, reward, new_observation, done):
        self.memory.append((observation, final_move, reward,
                           new_observation, done))  # Kolekcia (Tuple)

    def replay(self, model):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        observations = np.array([i[0] for i in minibatch])
        final_moves = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        new_observations = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        observations = np.squeeze(observations)  # 0D pole
        new_observations = np.squeeze(new_observations)  # 0D pole

        # Bellmanova rovnica (Bellman Equation)
        targets = rewards + self.gamma * \
            (np.amax(model.predict_on_batch(new_observations), axis=1))*(1-dones)
        targets_full = model.predict_on_batch(observations)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [final_moves]] = targets

        model.fit(observations, targets_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_test_logs(self, start_time, record_score, total_score):
        with open('logs/test_' + str(datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt', 'a') as f:
            f.write(str('start_time{}_record_score{}_total_score{}'.format(
                start_time, record_score, total_score)) + '\n')
            f.write('Values: {start_time: ' + str(start_time) + ', record_score: ' + str(record_score) +
                    ', total_score: ' + str(total_score) + '}\n')


def run(param):
    # Initialize game
    game = SnakeGame()

    # Hyperparams retyping and other variables
    no_of_layers = int(param['no_of_layers'])
    no_of_neurons = param['no_of_neurons']
    lr = param['lr']
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 500
    gamma = param['gamma']
    memory = deque(maxlen=2500)

    # Snake move vectors
    vectors_and_keys = [[[-game.SNAKE_BLOCK, 0], 0],  # LEFT
                        [[game.SNAKE_BLOCK, 0], 1],  # RIGHT
                        [[0, -game.SNAKE_BLOCK], 2],  # UP
                        [[0, game.SNAKE_BLOCK], 3]]  # DOWN

    # Initialize nn
    neural_network = NeuralNetwork(no_of_layers, no_of_neurons, lr)
    model = neural_network.model()

    # Initialize agent
    agent = Agent(epsilon, epsilon_min, epsilon_decay,
                  batch_size, gamma, memory, vectors_and_keys)

    # Initialize helper
    helper = Helper()

    # Optimization episodes and test episodes
    episodes = 1000
    test_episodes = 10000

    n_games = 0
    total_score = 0
    start_time = str(datetime.now().strftime("%Y%m%d%H%M%S"))

    if OPTIMIZATION and LOAD_WEIGHTS == False:
        for _ in range(episodes):
            game.reset()
            observation = agent.get_state(game)
            observation = np.array(observation).reshape(-1, 8)

            while game.MAX_STEPS != 0:
                _, score, food, snake, length = game.generate_observations()
                food_distance = agent.get_food_distance(snake, food, length)
                final_move = agent.generate_action(
                    snake, length, observation, model)
                agent.add_action_to_observation(observation, final_move)

                prev_observation = observation

                done, new_score, new_food, new_snake, new_length = game.game_loop(
                    final_move)
                new_observation = agent.get_state(game)
                new_food_distance = agent.get_food_distance(
                    new_snake, new_food, new_length)

                if new_score > score:
                    reward = 10

                if food_distance > new_food_distance:
                    reward = 1
                else:
                    reward = -1

                if done:
                    reward = -100

                new_observation = np.array(new_observation).reshape(-1, 8)

                agent.remember(observation, final_move,
                               reward, new_observation, done)

                observation = new_observation

                if agent.batch_size > 1:
                    agent.replay(model)

                if done:
                    break

            if new_score > game.RECORD:
                game.RECORD = new_score

            n_games += 1
            total_score += new_score

            print('Game: ', n_games, 'from: ', episodes, 'Score: ',
                  new_score, 'Record: ', game.RECORD)
            # print('Previous observation: ', prev_observation)
            # print('Total score: ', total_score)

        # Save weights
        neural_network.save_weights()

        return total_score
    elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
        for _ in range(test_episodes):
            game.reset()
            observation = agent.get_state(game)
            observation = np.array(observation).reshape(-1, 8)

            while game.MAX_STEPS != 0:
                _, score, food, snake, length = game.generate_observations()
                food_distance = agent.get_food_distance(snake, food, length)
                final_move = agent.generate_action(
                    snake, length, observation, model)
                agent.add_action_to_observation(observation, final_move)

                prev_observation = observation

                done, new_score, new_food, new_snake, new_length = game.game_loop(
                    final_move)
                new_observation = agent.get_state(game)
                new_food_distance = agent.get_food_distance(
                    new_snake, new_food, new_length)

                if new_score > score:
                    reward = 10

                if food_distance > new_food_distance:
                    reward = 1
                else:
                    reward = -1

                if done:
                    reward = -100

                new_observation = np.array(new_observation).reshape(-1, 8)

                agent.remember(observation, final_move,
                               reward, new_observation, done)

                observation = new_observation

                if agent.batch_size > 1:
                    agent.replay(model)

                if done:
                    break

            if new_score > game.RECORD:
                game.RECORD = new_score

            n_games += 1
            total_score += new_score

            print('Game: ', n_games, 'from: ', test_episodes, 'Score: ',
                  new_score, 'Record: ', game.RECORD)
            # print('Previous observation: ', prev_observation)
            # print('Total score: ', total_score)

            helper.write_result_to_list(n_games, new_score)

        # Save weights
        neural_network.save_weights()

        helper.write_result_to_csv()
        agent.save_test_logs(start_time, game.RECORD, total_score)
    else:
        neural_network.load_weights_()

        for _ in range(test_episodes):
            game.reset()
            observation = agent.get_state(game)
            observation = np.array(observation).reshape(-1, 8)

            while game.MAX_STEPS != 0:
                _, _, _, snake, length = game.generate_observations()
                final_move = agent.generate_action(
                    snake, length, observation, model)

                prev_observation = observation

                done, new_score, _, _, _ = game.game_loop(final_move)
                new_observation = agent.get_state(game)

                new_observation = np.array(new_observation).reshape(-1, 8)

                observation = new_observation

                if done:
                    break

            if new_score > game.RECORD:
                game.RECORD = new_score

            n_games += 1
            total_score += new_score

            print('Game: ', n_games, 'Score: ',
                  new_score, 'Record: ', game.RECORD)
            # print('Previous observation: ', prev_observation)
            # print('Total score: ', total_score)

            helper.write_result_to_list(n_games, new_score)

        helper.write_result_to_csv()
        agent.save_test_logs(start_time, game.RECORD, total_score)


if __name__ == '__main__':
    # Initialize optimization
    optimization = Optimization()

    if OPTIMIZATION and LOAD_WEIGHTS == False:
        # Hyperparams
        search_space = {
            'no_of_layers': np.arange(2, 6, 1),
            'no_of_neurons': np.arange(32, 256, 32),
            'lr': np.array([0.01, 0.001, 0.0001, 0.00001]),
            'gamma': np.arange(0.90, 0.99, 0.01)
        }

        # Optimization algorithms
        # opt = HillClimbingOptimizer(search_space)
        opt = StochasticHillClimbingOptimizer(search_space)

        # Run optimization for the N of iterations  
        opt.search(optimization.optimize, n_iter=100)
    elif OPTIMIZATION == False and LOAD_WEIGHTS == False:
        best_para = {
            'no_of_layers': 2,
            'no_of_neurons': 224,
            'lr': 0.00001,
            'gamma': 0.9400000000000001
        }

        # Run training nn with optimized hyperparams
        run(best_para)
    else:
        best_para = {
            'no_of_layers': 2,
            'no_of_neurons': 224,
            'lr': 0.00001,
            'gamma': 0.9400000000000001
        }

        # Run game with optimized hyperparams
        run(best_para)
