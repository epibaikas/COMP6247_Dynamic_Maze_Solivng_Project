import read_maze
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from celluloid import Camera

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

# Initialize q-table with zeros
Q_table = torch.zeros((maze_num_of_rows, maze_num_of_cols, num_of_actions))

# Initialize discount factor gamma
gamma = 0.9

# List of cells leading to a dead end
cells_leading_to_dead_end = []

# Setup figure for maze rendering
fig, ax = plt.subplots(figsize=(9, 6))

# Create camera object to record rendered frames
camera = Camera(fig)

def navigate_maze(epsilon=0.1, train=True, static=True, render=False):
    # Create a list with the cells you have already visited and add the initial position
    cells_already_visited = [(1, 1)]

    # Set initial state to [1, 1]
    position = torch.tensor([1, 1])
    total_reward = 0
    terminal_position = False

    # Record the number of samples acquired for each state-action pair
    sample_count = torch.zeros((maze_num_of_rows, maze_num_of_cols, num_of_actions))

    # Push visited junctions to a stack for backtracking
    junction_stack = []

    # Record actions taken at each junction in order to block cells leading to dead ends
    actions_taken_at_junction_stack = []

    # Number of steps
    steps = 0

    while not terminal_position and total_reward > -2000:
        x = position[0]
        y = position[1]
        if render:
            # Set title
            ax.set_title('Q-learning Dynamic Maze Solver', fontsize=14, pad=15)
            ax.axis('off')

            # Add position, total reward and number of steps text
            position_string = "Position: (" + str(x.item()) + ", " + str(y.item()) + ")"
            total_reward_string = "Total reward " + str(round(total_reward, 2))
            steps_string = "Steps: " + str(steps)

            # Add text to the side
            ax.annotate(position_string, (220, 100))
            ax.annotate(total_reward_string, (220, 110))
            ax.annotate(steps_string, (220, 120))

            # Reset maze_img
            maze_img = np.zeros((maze_num_of_rows, maze_num_of_cols + 50)) + 2
            maze_img[0:maze_num_of_rows, 0:maze_num_of_cols] = 4 * read_maze.maze_cells[:, :, 0] - 2

            # Add colour to visited cells
            for cell in cells_already_visited:
                maze_img[cell[0]][cell[1]] = 1.0

            # Add colour to cells on fire
            if not static:
                fire_cell_idx = np.argwhere(read_maze.maze_cells[:, :, 1] > 0)
                for idx in fire_cell_idx:
                    maze_img[idx[0]][idx[1]] = 0.5

            # Paint red the current position of the agent
            maze_img[x, y] = -1.5
            ax.imshow(maze_img, 'hot')

            # Update figure contents
            plt.draw()
            camera.snap()

        # Call get_local_maze_information() to observe the environment
        around = read_maze.get_local_maze_information(x, y)

        # Check if the current cell is a dead end or junction by counting the number of walls and empty cells
        # surrounding it. If a neighboring cell leads to a dead end count it as a wall.
        wall_count = 0
        empty_cell_count = 0
        if around[0, 1, 0] == 0 or (x-1, y) in cells_leading_to_dead_end:  # Cell above
            wall_count += 1
        else:
            empty_cell_count += 1

        if around[1, 0, 0] == 0 or (x, y-1) in cells_leading_to_dead_end:  # Cell to the left
            wall_count += 1
        else:
            empty_cell_count += 1

        if around[1, 2, 0] == 0 or (x, y+1) in cells_leading_to_dead_end:  # Cell to the right
            wall_count += 1
        else:
            empty_cell_count += 1

        if around[2, 1, 0] == 0 or (x+1, y) in cells_leading_to_dead_end:  # Cell below
            wall_count += 1
        else:
            empty_cell_count += 1

        # If the cell is a dead end, reset position to last junction and block the neighboring cell that led to dead end
        # Exclude the initial position (1, 1)
        if wall_count >= 3 and not (x == 1 and y == 1):
            position = junction_stack.pop()
            action_taken_at_junction = actions_taken_at_junction_stack.pop()

            # Change position to tensor format
            position = torch.tensor([position[0], position[1]])

            # Find the neighboring cell that led to dead end by adding the action
            # taken at the junction the previous time
            temp = (position[0] + action_taken_at_junction[0], position[1] + action_taken_at_junction[1])
            cells_leading_to_dead_end.append(temp)
        else:
            # Choose action / get random action only when training
            random_num = random.uniform(0, 1)
            if random_num < epsilon and train:
                action_idx = random.randint(0, num_of_actions - 1)
            else:
                # Find the action that maximizes Q(s, a)
                action_idx = torch.argmax(Q_table[x, y, :])
            current_action_key = action_keys[action_idx]
            current_action = actions[current_action_key]

            # Check if intersection
            if empty_cell_count >= 3 or (x == 1 and y == 1):
                # if the current junction exists in the junction stack, discard all the junctions after its index
                if (x, y) in junction_stack:
                    junction_idx = junction_stack.index((x, y))
                    junction_stack = junction_stack[0: junction_idx]
                    actions_taken_at_junction_stack = actions_taken_at_junction_stack[0: junction_idx]

                # Push junction position and current_action to the corresponding stacks
                junction_stack.append((x, y))
                actions_taken_at_junction_stack.append(current_action)

            # Apply action and get reward
            potential_next_position = position + current_action
            potential_next_x = potential_next_position[0]
            potential_next_y = potential_next_position[1]

            if (potential_next_x, potential_next_y) in cells_leading_to_dead_end:  # Action leads to dead end
                next_position = position  # Stay in the same position
                reward = -2
            elif around[1 + current_action[0]][1 + current_action[1]][0] == 0:  # Action leads to wall
                next_position = position  # Stay in the same position
                reward = -0.8
            elif around[1 + current_action[0]][1 + current_action[1]][1] > 0 and not static:  # Action leads to fire
                next_position = position  # Stay in the same position
                reward = -0.7
            elif (potential_next_x, potential_next_y) in cells_already_visited:  # Action leads to staying in the same cell or moving to a cell you have already visited
                next_position = position  # Stay in the same position
                reward = -0.5
            else:
                next_position = potential_next_position
                cells_already_visited.append((potential_next_x, potential_next_y))
                reward = -0.05  # Move to a valid cell you haven't visited before

            next_x = next_position[0]
            next_y = next_position[1]

            # Check if the terminal state has been reached
            if next_x == maze_num_of_rows - 2 and next_y == maze_num_of_cols - 2:
                terminal_position = True
                reward = 100

            # Increase sample count
            sample_count[x, y, action_idx] += 1

            if train:
                # Find the action a' that maximizes Q(s', a')
                next_action_idx = torch.argmax(Q_table[next_x, next_y, :])

                alpha = 1/sample_count[x, y, action_idx]

                # Update Q_table
                Q_table[x, y, action_idx] = Q_table[x, y, action_idx] + alpha * (reward + gamma * Q_table[next_x, next_y, next_action_idx] - Q_table[x, y, action_idx])

            if not train:
                with open('output.txt', 'a') as f:
                    line_string = "Step: " + str(steps) + ", "
                    line_string += "Position: (" + str(x.item()) + ", " + str(y.item()) + "), "
                    line_string += "Action: " + current_action_key + ", "
                    line_string += "Total reward: " + str(total_reward) + "\n"
                    f.write(line_string)
                    line_string = "Around cells         Fire\n"
                    f.write(line_string)
                    for i in range(3):
                            line_string = "[" + str(around[i, 0, 0]) + "]" + "[" + str(around[i, 1, 0]) + "]" + "[" + str(around[i, 2, 0]) + "]          "
                            line_string += "[" + str(around[i, 0, 1]) + "]" + "[" + str(around[i, 1, 1]) + "]" + "[" + str(around[i, 2, 1]) + "]\n"
                            f.write(line_string)

                    f.write('\n')

            # Update the current position
            position = next_position

            total_reward += reward
            steps += 1

    return total_reward, position


def train_maze_solver(num_of_epochs=100):
    total_rewards_list = []
    epsilon=0.1
    epsilon_decay = 0.99

    for epoch in range(num_of_epochs):
        epoch_total_reward, last_position = navigate_maze(epsilon, train=True, render=False)
        total_rewards_list.append(epoch_total_reward)
        print("Epoch: ", epoch, " Total reward: ", epoch_total_reward, "Last position: ", last_position)
        epsilon = epsilon * epsilon_decay

    # Plot total rewards for each epoch
    fig_2 = plt.figure(2)
    ax_2 = fig_2.gca()
    ax_2.plot(np.arange(1, num_of_epochs+1), total_rewards_list)
    ax_2.set_title('Total rewards plot', fontsize=14)
    ax_2.set_xlabel('epoch')
    ax_2.set_ylabel('Total reward')
    ax_2.grid()
    ax_2.set_xticks(range(1, num_of_epochs+1, 19))
    fig_2.savefig('total_rewards_plot.png')

    # Save Q-table to file q_table.pt
    torch.save(Q_table, 'q_table.pt')

# Execute training sequence
print("Train maze solver...")
train_maze_solver(num_of_epochs=100)

# Load trained q_table
Q_table = torch.load('q_table.pt')

# Evaluate
print("Navigate maze in evaluation mode...")
# Create .txt output file
f = open('output.txt', 'w')
f.write('Dynamic maze solving algorithm - output file \n')
f.close()
navigate_maze(train=False, static=False, render=True)

print("Creating animation and saving it as an mp4 file... (may take a few minutes)")
animation = camera.animate(interval=20, repeat=True)
animation.save("maze_animation.mp4")

print("Done!")
plt.show()