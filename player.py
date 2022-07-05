from re import A
import pygame
import torch
import numpy

import collections as c
import random
import argparse

#model stuff

# input - [paddle x, ball x, ball y, ball vx, ball vy]
# output - [left, still, right]

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model file to load')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
args = parser.parse_args()

class GameModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # taken from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        # no idea if they're good here, should think this through
        self.first = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.second = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.third = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear = torch.nn.Linear(9216, 256)
        self.linear_two = torch.nn.Linear(256, 3)
        # here we're predicting scores for each of the 3 actions
        # rather than predicting score given an action

    def forward(self, x):
        x = torch.nn.functional.silu(self.first(x))
        x = torch.nn.functional.silu(self.second(x))
        x = torch.nn.functional.silu(self.third(x))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.silu(self.linear(x))
        x = self.linear_two(x)

        return x

#LR = 1e-3
# init model
model = GameModel()
if args.model:
    model.load_state_dict(torch.load(args.model))
    print('Loaded model from', args.model)

#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# initialize device
device = torch.device(args.device)
model.to(device)

MEMORY_LEN = 100

#state_memory = c.deque([], maxlen=MEMORY_LEN)

state_memory = torch.tensor([]).to(device)
action_memory = torch.tensor([], dtype=torch.int64).to(device)

score_memory = c.deque([], maxlen=MEMORY_LEN)
reward_memory = c.deque([], maxlen=MEMORY_LEN)

decay = 0.99

displaying = True

running_average_loss = 0.0

reward = 0.0

hits = 0.0

if __name__=="__main__":
    # game stuff

    pygame.init()

    HEIGHT = 800
    WIDTH = 500

    screen = pygame.display.set_mode([WIDTH, HEIGHT])

    ball_loc = [250, 0]
    ball_speed = [5, 5]

    paddle_width = 100
    paddle_height = 5
    paddle_loc = [250, HEIGHT-paddle_height-5]
    paddle_speed = 10

    moving_left = False
    moving_right = False
    ml_control = True # always true because rl

    running = True
    agent_score = 0

    rendering = True

    counter = 0

    old_pixels = None

    last_state_and_action = None

    random_action = 0
    random_movement_counter = 0

    while running:
        counter += 1
        #print(foo_counter)
        # Fill the background with white
        screen.fill((255, 255, 255))

        # Draw a solid blue circle in the center
        pygame.draw.circle(screen, (0, 0, 0), ball_loc, 5)
        pygame.draw.rect(screen, (0, 0, 0), (paddle_loc[0]-paddle_width/2, paddle_loc[1], paddle_width, paddle_height))
        
        if rendering:
            # Flip the display
            pygame.display.flip()
            # sleep for a bit
            pygame.time.delay(10)

        # can we just reference them?
        small = pygame.transform.scale(screen, (128, 128))

        pixels = torch.tensor(1 - (pygame.surfarray.array_red(small) / 255.0)).float().unsqueeze(0)
        if old_pixels is None:
            old_pixels = pixels

        pixels_diff = pixels - old_pixels

        old_pixels = pixels
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rendering = not rendering


        # predict value of each action
        action_values = model(pixels_diff.unsqueeze(0).to(device))[0]

        ai_moving_left = False
        ai_moving_right = False

        if ml_control:
            #print(random_movement_counter, random_action)
            # overwrite player action
            if random_movement_counter == 0 and random.random() < 0.01:
                random_movement_counter = 20
                random_action = random.randint(0, 2)
                #print("choosing randomly")
            else:
                #print(action_values)
                # RSI select probabilistically
                #action_probs = torch.nn.functional.softmax(torch.tensor(action_values), dim=0)
                #action_to_pick = torch.multinomial(action_probs, 1)[0].item()
                action_to_pick = torch.argmax(action_values).item()
                #print("picking", action_values, action_to_pick)

            if random_movement_counter > 0:
                random_movement_counter -= 1
                action_to_pick = random_action
                
            #print("Picked ", action_to_pick)
            #print

            if action_to_pick == 0:
                pass
            elif action_to_pick == 1:
                ai_moving_left = True
                ai_moving_right = False
            elif action_to_pick == 2:
                ai_moving_left = False
                ai_moving_right = True
        else:
            print("unsupported")
            pass

        new_reward = 0.0

        # update physics for next frame
        if moving_left or ai_moving_left:
            paddle_loc[0] -= 5
            if paddle_loc[0] < 0:
                paddle_loc[0] = 0
        if moving_right or ai_moving_right:
            paddle_loc[0] += 5
            if paddle_loc[0] > WIDTH:
                paddle_loc[0] = WIDTH

        ball_loc[0] += ball_speed[0]
        ball_loc[1] += ball_speed[1]

        #torch.autograd.set_detect_anomaly(True)

        if ball_loc[1] > HEIGHT-paddle_height-5:
            if ball_loc[0] > paddle_loc[0] - paddle_width/2 and ball_loc[0] < paddle_loc[0] + paddle_width/2:
                ball_speed[1] = -ball_speed[1]
                new_reward = 100.0
                hits += 1
            else:
                # Reset, episode over
                ball_loc[1] = 0
                new_reward = -100.0


        # Advance the ball
        if ball_loc[1] < 0:
            ball_speed[1] = -ball_speed[1]
        if ball_loc[0] > WIDTH:
            ball_speed[0] = -ball_speed[0]
        if ball_loc[0] < 0:
            ball_speed[0] = -ball_speed[0]
 

        # save state and reward
        #print(action_states)
        if last_state_and_action is not None:
            state_memory = torch.cat((state_memory, last_state_and_action[0].unsqueeze(0).to(device)), dim=0)
            action_memory = torch.cat((action_memory, torch.tensor([last_state_and_action[1]]).to(device)), dim=0)

            if state_memory.shape[0] > MEMORY_LEN:
                state_memory = state_memory[1:]
                action_memory = action_memory[1:]

            reward_memory.append(reward)
        
        pred_reward = action_values[action_to_pick]

        reward_for_training = reward_memory.copy()
        for i in range(len(reward_for_training)-1, 0-1, -1):
            #print(i)
            reward_for_training[i] += pred_reward
            pred_reward = reward_for_training[i] * decay


        if reward != 0.0:
            optimizer.zero_grad()

            predicted_scores = model(state_memory)
            predicted_and_chosen_scores = predicted_scores.gather(1, torch.tensor(action_memory).unsqueeze(-1).to(device))

            actual_scores = torch.tensor(reward_for_training).to(device)

            #import pdb; pdb.set_trace()
            loss = loss_fn(predicted_and_chosen_scores, actual_scores)

            #import pdb; pdb.set_trace()

            loss.backward()        
            optimizer.step()

        if counter % 1000 == 999:
            avg_loss = torch.mean(loss).item()
            running_average_loss = (running_average_loss * 0.90) + (avg_loss * 0.1)
            print("avg loss: ", avg_loss, running_average_loss, "hits: ", hits)
            hits = 0


    
        reward = new_reward
        last_state_and_action = (pixels_diff, action_to_pick)
        

    # Done! Time to quit.
    pygame.quit()

# Save model to file
torch.save(model.state_dict(), 'model.pt')
