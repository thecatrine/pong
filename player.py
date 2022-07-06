from re import A
import pygame
import torch
import numpy as np

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

class PongEnv():
    def __init__(self):
        self.paddle_width = 100
        self.paddle_height = 5
        self.paddle_speed = 5

        self.ball_loc = None
        self.ball_speed = None
        self.paddle_loc = None

        pygame.init()

        self.HEIGHT = 800
        self.WIDTH = 500

        self.screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT])

        self.GAME_LIMIT = 1_000

    def reset(self):
        self.ball_loc = [250, 0]
        self.ball_speed = [5, 5]
        self.paddle_loc = [250, self.HEIGHT-self.paddle_height-5]
        self.game_counter = 0
        self.old_pixels = None

        return self.step()

    def step(self, action=0, render=False):
        reward = 0
        done = False
        self.game_counter += 1
        if self.game_counter > self.GAME_LIMIT:
            done = True

        # 0: still, 1: left, 2: right
        if action == 0:
            pass
        elif action == 1:
            self.paddle_loc[0] = self.paddle_loc[0] - self.paddle_speed
            if self.paddle_loc[0] < 0:
                self.paddle_loc[0] = 0
        elif action == 2:
            self.paddle_loc[0] = self.paddle_loc[0] + self.paddle_speed
            if self.paddle_loc[0] > self.WIDTH:
                self.paddle_loc[0] = self.WIDTH
            

        # Advance the ball
        if self.ball_loc[1] < 0:
            self.ball_speed[1] = -self.ball_speed[1]
        
        self.ball_loc[0] += self.ball_speed[0]
        if self.ball_loc[0] > self.WIDTH:
            self.ball_speed[0] = -self.ball_speed[0]
        if self.ball_loc[0] < 0:
            self.ball_speed[0] = -self.ball_speed[0]
        
        self.ball_loc[1] += self.ball_speed[1]

        if self.ball_loc[1] > self.HEIGHT-self.paddle_height-5:
            if (self.ball_loc[0] > self.paddle_loc[0] - self.paddle_width/2) and \
               (self.ball_loc[0] < self.paddle_loc[0] + self.paddle_width/2):
                self.ball_speed[1] = -self.ball_speed[1]
                reward = 100.0
            else:
                # Reset the ball
                self.ball_loc[1] = 0
                reward = -100.0


        # Render image to get next state
        self.screen.fill((255, 255, 255))

        # Draw a solid blue circle in the center
        pygame.draw.circle(self.screen, (0, 0, 0), self.ball_loc, 5)
        pygame.draw.rect(self.screen, (0, 0, 0), (self.paddle_loc[0]-self.paddle_width/2, self.paddle_loc[1], self.paddle_width, self.paddle_height))
        
        if render:
            # Flip the display
            pygame.display.flip()
            # sleep for a bit
            pygame.time.delay(10)

        # can we just reference them?
        small = pygame.transform.scale(self.screen, (128, 128))

        pixels = torch.tensor(1 - (pygame.surfarray.array_red(small) / 255.0)).float().unsqueeze(0).detach()
        if self.old_pixels is None:
            self.old_pixels = pixels

        pixels_diff = pixels - self.old_pixels

        self.old_pixels = pixels
        
        # return values are (state, reward, done)
        return pixels, reward, done
        


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

        return torch.nn.functional.softmax(x, dim=1)


def compute_loss(net, obvs, actions, rewards):
    res = net(obvs)
    logprobs = res.gather(1, actions.unsqueeze(-1))
    return -torch.mean(logprobs * rewards)

rendering = True
running = True

def train_one_epoch(net, optimizer, env):
    global rendering
    global running

    batch_obvs = []
    batch_actions = []
    batch_weights = []
    batch_returns = []
    batch_lens = []

    pixels_diff, reward, done = env.reset()
    ep_rewards = []
    done = False

    BATCH_SIZE = 5_000

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rendering = not rendering
        
        batch_obvs.append(pixels_diff.unsqueeze(0))

        #print(len(batch_obvs))

        #import pdb; pdb.set_trace()

        act_probs = net(pixels_diff.unsqueeze(0).to(device))
        act = torch.multinomial(act_probs, 1).item()

        pixels_diff, reward, done = env.step(act, rendering)

        batch_actions.append(act)
        ep_rewards.append(reward)

        if done:
            ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
            batch_returns.append(ep_return)
            batch_lens.append(ep_len)

            batch_weights += [ep_return] * ep_len

            #import pdb; pdb.set_trace()

            # Reset the environment for another episode
            pixels_diff, reward, done = env.reset()
            done = False
            ep_rewards = []

            if len(batch_obvs) > BATCH_SIZE:
                break

    optimizer.zero_grad()

    #import pdb; pdb.set_trace()
    batch_loss = compute_loss(
        net, 
        torch.cat(batch_obvs, dim=0).to(device), 
        torch.tensor(batch_actions).to(device), 
        torch.tensor(batch_weights).to(device),
    )

    batch_loss.backward()
    optimizer.step()

    return batch_loss, batch_returns, batch_lens


#LR = 1e-3
# init model
model = GameModel()
if args.model:
    model.load_state_dict(torch.load(args.model))
    print('Loaded model from', args.model)

#loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# initialize device
device = torch.device(args.device)
model.to(device)


if __name__=="__main__":
    env = PongEnv()

    i = 0
    while running:
        i += 1
        batch_loss, batch_returns, batch_lens = train_one_epoch(model, optimizer, env)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_returns), np.mean(batch_lens)))
        

    # Done! Time to quit.
    pygame.quit()

# Save model to file
torch.save(model.state_dict(), 'model.pt')
