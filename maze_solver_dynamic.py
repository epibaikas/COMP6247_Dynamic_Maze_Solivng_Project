import read_maze
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# Call load_maze() function to load maze from .npy file
read_maze.load_maze()
# Save the number of rows and columns of the maze
maze_num_of_rows = read_maze.maze_cells.shape[0]
maze_num_of_cols = read_maze.maze_cells.shape[1]

num_of_actions = 5
action_keys = ['left', 'right', 'up', 'down', 'stay']
actions = {'left': torch.tensor([0, -1]),
           'right': torch.tensor([0, 1]),
           'up': torch.tensor([-1, 0]),
           'down': torch.tensor([1, 0]),
           'stay': torch.tensor([0, 0])}

state = torch.zeros((maze_num_of_rows, maze_num_of_cols, 2, 2, 2, 2))
Q_table = torch.zeros((maze_num_of_rows, maze_num_of_cols, 2, 2, 2, 2, num_of_actions))
gamma = 0.9
epsilon = 0.1

def navigate_maze(train=True, render=False):
    if render:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Q-learning Dynamic Maze Solver', fontsize=14)

        # Load maze_cells in an image array to print maze through imshow
        # Multiply values by 4 and subtract -2 to match the color map range
        # Values of -2 (walls) are printed with black colour
        # Values of 2 (empty cells) are printed with white colour
        maze_img = 4 * read_maze.maze_cells[:, :, 0] - 2
        img = ax.imshow(maze_img, 'hot')

    # Create a list with the cells you have already visited and add the initial position
    cells_already_visited = [(1, 1)]

    # Set initial position to [1, 1]
    position = torch.tensor([1, 1])
    total_reward = 0
    terminal_position = False

    # Record the number of samples acquired for each state-action pair
    sample_count = torch.zeros((maze_num_of_rows, maze_num_of_cols, 2, 2, 2, 2, num_of_actions))

    while not terminal_position and total_reward > -200:
        x = position[0]
        y = position[1]

        # Call get_local_maze_information() to observe the environment
        around = read_maze.get_local_maze_information(x, y)

        fire_up = 0
        fire_below = 0
        fire_left = 0
        fire_right = 0
        if around[0, 1, 1] > 0:
            fire_up = 1
        if around[2, 1, 1] > 0:
            fire_below = 1
        if around[1, 0, 1] > 0:
            fire_left = 1
        if around[1, 2, 1] > 0:
            fire_right = 1

        state = torch.tensor((x, y, fire_up, fire_below, fire_left, fire_right))

        if render:
            # Reset maze_img
            maze_img = 4 * read_maze.maze_cells[:, :, 0] - 2

            # Add colour to visited cells
            for cell in cells_already_visited:
                maze_img[cell[0]][cell[1]] = 1.0

            # Add colour to cells on fire
            fire_cell_idx = np.argwhere(read_maze.maze_cells[:, :, 1] > 0)
            for idx in fire_cell_idx:
                maze_img[idx[0]][idx[1]] = 0.5

            # Paint red the current position of the agent
            maze_img[x, y] = -1.5  # np.random.rand();
            img.set_data(maze_img)
            plt.draw(), plt.pause(1e-1)

        # Choose action / get random action only when training
        random_num = random.uniform(0, 1)
        if random_num < epsilon and train:
            action_idx = random.randint(0, num_of_actions - 1)
        else:
            # Find the action that maximizes Q(s, a)
            action_idx = torch.argmax(Q_table[x, y, fire_up, fire_below, fire_left, fire_right, :])
        current_action_key = action_keys[action_idx]
        current_action = actions[current_action_key]

        # Apply action and get reward
        potential_next_position = position + current_action
        potential_next_x = potential_next_position[0]
        potential_next_y = potential_next_position[1]
        if around[1 + current_action[0]][1 + current_action[1]][0] == 0: # Action leads to wall
            next_position = position # Stay in the same position
            reward = -0.8#-0.3
        elif around[1 + current_action[0]][1 + current_action[1]][1] > 0: # Action leads to fire
            next_position = position # Stay in the same position
            reward = -0.7
        elif (potential_next_x, potential_next_y) in cells_already_visited: # Action leads to stay in the same cell or move to a cell you have already visited
            next_position = potential_next_position
            reward = -0.5
        else:
            next_position = potential_next_position
            cells_already_visited.append((potential_next_x, potential_next_y))
            reward = -0.05 # Move to a valid cell you haven't visited before

        next_x = next_position[0]
        next_y = next_position[1]

        # Check if the terminal state has been reached
        if next_x == maze_num_of_rows - 2 and next_y == maze_num_of_cols - 2:
            terminal_position = True
            reward = 1

        # Increase sample count
        sample_count[x, y, fire_up, fire_below, fire_left, fire_right, action_idx] += 1

        if train:
            # Find the action a' that maximizes Q(s', a')
            next_action_idx = torch.argmax(Q_table[next_x, next_y, fire_up, fire_below, fire_left, fire_right, :])

            alpha = 1/sample_count[state, action_idx]

            # Update Q_table
            Q_table[x, y, fire_up, fire_below, fire_left, fire_right, action_idx] = Q_table[x, y, fire_up, fire_below, fire_left, fire_right, action_idx] + alpha * (reward + gamma * Q_table[next_x, next_y, fire_up, fire_below, fire_left, fire_right, next_action_idx] - Q_table[x, y, fire_up, fire_below, fire_left, fire_right, action_idx])

        # Update the current position
        position = next_position

        total_reward += reward

    return sample_count, total_reward, position

def train_maze_solver(num_of_epochs=4000):
    total_rewards_list = []

    for epoch in range(num_of_epochs):
        epoch_sample_count, epoch_total_reward, state = navigate_maze(train=True, render=False)
        total_rewards_list.append(epoch_total_reward)
        print("Epoch: ", epoch, " Total_reward: ", epoch_total_reward, "Last_state: ", state)

    # Plot total rewards for each epoch
    fig_2 = plt.figure(2)
    ax_2 = fig_2.gca()
    ax_2.plot(np.arange(1, num_of_epochs+1), total_rewards_list)
    ax_2.set_title('Total rewards plot', fontsize=14)
    ax_2.set_xlabel('epoch')
    ax_2.set_ylabel('Total reward')
    ax_2.grid()
    ax_2.set_xticks(range(1, num_of_epochs+1, 99))
    fig_2.savefig('total_rewards_plot.png')

    # Save Q-table to file q_table.pt
    torch.save(Q_table, 'q_table.pt')

# Execute training and evaluation sequence
train_maze_solver(num_of_epochs=2000)
Q_table = torch.load('q_table.pt')
navigate_maze(train=False, render=True)
