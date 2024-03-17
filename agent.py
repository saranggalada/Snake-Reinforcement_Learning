import numpy as np
import random
import torch
from collections import deque
from game import SnakeGameRL, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot

# Hyperparameters

HIDDEN_LAYER = 256  # number of units in QN hidden layer
BATCH_SIZE = 1000  # number of SARS tuples to sample from memory
LR = 0.001  # learning rate
GAMMA = 0.9  # discount rate
EPSILON = 0.4  # exploration rate
EXPLORATION_PHASE = 100 # number of games over which to linearly decay epsilon

MAX_MEMORY = 100000  # maximum length of memory
PIXEL_SIZE = 20 # size of each pixel in the game

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, HIDDEN_LAYER, 3)  # 11 inputs (state vector), 256 hidden units, 3 outputs (actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # State is a list of 11 binary values
        # 3 dirs of danger wrt snake dir (ie. arena boundaries) (NOTE: danger is always towards front or side of snake)
        # 4 dirs of snake
        # 4 dirs of food

        # Snake's head and neighbouring pixels
        head = game.snake[0] # head of snake
        pixel_l = Point(head.x - PIXEL_SIZE, head.y) # pixel left of head
        pixel_r = Point(head.x + PIXEL_SIZE, head.y) # pixel right of head
        pixel_u = Point(head.x, head.y - PIXEL_SIZE) # pixel above head
        pixel_d = Point(head.x, head.y + PIXEL_SIZE) # pixel below head

        # Direction of snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Construct state vector
        state = [
            # Danger straight (dir of danger = dir of snake)
            (dir_r and game.is_danger(pixel_r)) or
            (dir_l and game.is_danger(pixel_l)) or
            (dir_u and game.is_danger(pixel_u)) or
            (dir_d and game.is_danger(pixel_d)),

            # Danger right (dir of danger = right of dir of snake)
            (dir_u and game.is_danger(pixel_r)) or
            (dir_d and game.is_danger(pixel_l)) or
            (dir_l and game.is_danger(pixel_u)) or
            (dir_r and game.is_danger(pixel_d)),

            # Danger left (dir of danger = left of dir of snake)
            (dir_d and game.is_danger(pixel_r)) or
            (dir_u and game.is_danger(pixel_l)) or
            (dir_r and game.is_danger(pixel_u)) or
            (dir_l and game.is_danger(pixel_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # is food left?
            game.food.x > game.head.x,  # is food right?
            game.food.y < game.head.y,  # is food up?
            game.food.y > game.head.y  # is food down?
        ]

        return np.array(state, dtype=int) 

    # Train the model using a batch of SARS tuples from memory
    def train_long_memory(self):
        # Randomly pick a batch of SARS tuples from memory (BATCH_SIZE many)
        if len(self.memory) > BATCH_SIZE:
            train_sample = random.sample(self.memory, BATCH_SIZE)  # list of SARS tuples
        else:   # If memory is not large enough, use all of it
            train_sample = self.memory

        # For each SARS tuple in the batch, update the model
        # for state, action, reward, next_state, game_over in train_sample:
        #     self.trainer.train_step(state, action, reward, next_state, game_over)
            
        # Alternatively, group together all states, actions, rewards, next states, and game overs
        # and update the model in one go
        states, actions, rewards, next_states, game_overs = zip(*train_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    # Train the model using a single SARS tuple
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    # Policy implementation: Choose an action to take given the current state
    def get_action(self, state):
        # Use an Epsilon-Greedy policy to choose an action
        # Trade-off between exploration and exploitation

        # Linearly decrease epsilon as training progresses: transition from exploration to exploitation
        self.epsilon = max(EPSILON * (1-self.n_games/EXPLORATION_PHASE), 0)
        final_move = [0, 0, 0] # boolean [straight, right, left]
        if random.randint(0, 100) < self.epsilon*100:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def play():
    scores = []
    mean_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGameRL()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get action as per policy
        final_move = agent.get_action(state_old)

        # perform action and reach a new state, getting a reward
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short term memory - trains on single SARS tuples
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # store SARS (state, action, reward, next state) tuple in memory
        agent.memory.append((state_old, final_move, reward, state_new, game_over))

        if game_over:
            # train long term memory, plot results
            game.game_reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Track and save best score and model
            if score > record_score:
                record_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record_score)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores, agent.epsilon)

if __name__ == '__main__':
    play()