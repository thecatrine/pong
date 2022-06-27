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
        self.first = torch.nn.Linear(8, 256)
        self.second = torch.nn.Linear(256, 256)
        self.second_two = torch.nn.Linear(256, 256)
        self.third = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.first(x))
        x = torch.nn.functional.silu(self.second(x))
        x = torch.nn.functional.silu(self.second_two(x))
        x = self.third(x)

        return x

#LR = 1e-3
# init model
model = GameModel()
if args.model:
    model.load_state_dict(torch.load(args.model))
    print('Loaded model from', args.model)

#loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# initialize device
device = torch.device(args.device)
model.to(device)

MEMORY_LEN = 1000

state_memory = c.deque([], maxlen=MEMORY_LEN)
score_memory = c.deque([], maxlen=MEMORY_LEN)
reward_memory = c.deque([], maxlen=MEMORY_LEN)

decay = 0.99

displaying = True

running_average_loss = 0.0

reward = 0.0
action_we_picked_last = torch.zeros(8).numpy()

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

    rendering = False

    counter = 0
    while running:
        counter += 1
        #print(foo_counter)
        if rendering:
            # Fill the background with white
            screen.fill((255, 255, 255))

            # Draw a solid blue circle in the center
            pygame.draw.circle(screen, (0, 0, 0), ball_loc, 5)
            pygame.draw.rect(screen, (0, 0, 0), (paddle_loc[0]-paddle_width/2, paddle_loc[1], paddle_width, paddle_height))
            # Flip the display
            pygame.display.flip()
            # sleep for a bit
            pygame.time.delay(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rendering = not rendering

        state = [
            ((paddle_loc[0]/WIDTH)-0.5),
            ((ball_loc[0]/WIDTH)-0.5),
            ((ball_loc[1]/HEIGHT)-0.5),
            ((ball_speed[0]/5)-0.5),
            ((ball_speed[1]/5)-0.5),
        ]


        action_values = []
        action_states = []
        for action in range(3):
            # convert action to one-hot
            action_one_hot = torch.zeros(3)
            action_one_hot[action] = 1

            state_and_action = torch.cat([torch.tensor(state), action_one_hot]).to(device)

            # Predict value
            #print("1", state_and_action)
            action_values.append(model(state_and_action)[0])
            action_states.append(state_and_action)

        # Add randomness?

        ai_moving_left = False
        ai_moving_right = False

        if ml_control:
            # overwrite player action
            if random.random() < 0.03:
                action_to_pick = random.randint(0, 2)
                #print("choosing randomly")
            else:
                #print(action_values)
                # RSI select probabilistically
                #action_probs = torch.nn.functional.softmax(torch.tensor(action_values), dim=0)
                #action_to_pick = torch.multinomial(action_probs, 1)[0].item()
                action_to_pick = torch.argmax(torch.tensor(action_values)).item()
                #print(action_probs, action_to_pick)

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

        real_new_score = 0

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

        scored = False
        #torch.autograd.set_detect_anomaly(True)

        new_reward = 0.0
        #if ai_moving_left:
        #    new_reward -= 1.0
        #if ai_moving_right:
        #    new_reward -= 1.0

        if ball_loc[1] > HEIGHT-paddle_height-5:
            if ball_loc[0] > paddle_loc[0] - paddle_width/2 and ball_loc[0] < paddle_loc[0] + paddle_width/2:
                ball_speed[1] = -ball_speed[1]
                new_reward = 100.0
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
 

        score = action_values[action_to_pick]

        # save state and reward
        #print(action_states)
        state_memory.append(action_we_picked_last)
        score_memory.append(score)
        
        pred_reward = action_values[action_to_pick].item()

        reward_memory.append(reward)

        reward_for_training = reward_memory.copy()
        for i in range(len(reward_for_training)-1, 0-1, -1):
            #print(i)
            reward_for_training[i] += pred_reward
            pred_reward = reward_for_training[i] * decay


        if reward != 0.0:
            optimizer.zero_grad()

            predicted_score = model(torch.tensor(numpy.array(state_memory)).to(device))

            loss = loss_fn(predicted_score, torch.tensor(reward_for_training).unsqueeze(-1).to(device))

            loss.backward(retain_graph=True)        
            optimizer.step()

        if counter % 1000 == 999:
            avg_loss = torch.mean(loss).item()
            running_average_loss = (running_average_loss * 0.90) + (avg_loss * 0.1)
            print("avg loss: ", avg_loss, running_average_loss)


    
        reward = new_reward
        action_we_picked_last = action_states[action_to_pick].cpu().detach().numpy()
        

    # Done! Time to quit.
    pygame.quit()

# Save model to file
torch.save(model.state_dict(), 'model.pt')
