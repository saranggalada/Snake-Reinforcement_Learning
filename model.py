import torch
import os

class Linear_QNet(torch.nn.Module):  # Shallow (3 layer) Neural Network for Q Learning
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # inherit from nn.Module

        # Define the layers of the neural network
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # Save the model
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:  # Train the QNet
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # if the state is 1D, reshape it to 2D using torch.unsqueeze()
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # 1: predicted Q values with current state
        prediction = self.model(state)

        # 2: Bellman update: Q_new = r + gamma * max(next_predicted Q value) => only if not done
        # expected Q values
        target = prediction.clone()
        # target = target.detach()

        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # if game_over:
        #     target[0][torch.argmax(action).item()] = reward
        # else:
        #     target[0][torch.argmax(action).item()] = reward + self.gamma * torch.max(self.model(next_state))

        # 3: loss = (Q_new - Q_old)^2
        loss = self.criterion(target, prediction)

        # 4: backpropagation
        self.optimiser.zero_grad() # standard step to reset the gradients to zero
        loss.backward() # backpropagation
        self.optimiser.step() # update the weights