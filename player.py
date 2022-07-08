from re import A
import pygame
import torch
import numpy as np

import collections as c
import random
import argparse
import collections

import matplotlib.pyplot as plt
#model stuff

# input - [paddle x, ball x, ball y, ball vx, ball vy]
# output - [left, still, right]

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model file to load')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
args = parser.parse_args()

def show(pic):
    plt.imshow(pic)
    plt.show()

class PongEnv():
    def __init__(self):
        self.paddle_width = 100
        self.paddle_height = 10
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
            pass
        #   RSI done = True

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
            #pygame.time.delay(10)

        # can we just reference them?
        small = pygame.transform.scale(self.screen, (64, 64))

        pixels = torch.tensor(1 - (pygame.surfarray.array_red(small) / 255.0)).float().unsqueeze(0).detach()
        if self.old_pixels is None:
            self.old_pixels = pixels

        pixels_to_return = torch.cat((pixels, self.old_pixels))

        #import pdb; pdb.set_trace()

        self.old_pixels = pixels
        
        # return values are (state, reward, done)
        return pixels_to_return, reward, done
        

class RewardNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # taken from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        # no idea if they're good here, should think this through
        self.first = torch.nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.second = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.third = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear = torch.nn.Linear(1024, 256)
        self.linear_two = torch.nn.Linear(256, 3)
        # here we're predicting scores for each of the 3 a

    def forward(self, x):
        x = torch.nn.functional.silu(self.first(x))
        x = torch.nn.functional.silu(self.second(x))
        x = torch.nn.functional.silu(self.third(x))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.silu(self.linear(x))
        x = self.linear_two(x)

        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # taken from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        # no idea if they're good here, should think this through
        self.first = torch.nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.second = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.third = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.linear = torch.nn.Linear(1024, 256)
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

memory = collections.deque(maxlen=2000)

def train_one_epoch(p_net, p_optimizer, r_net, r_optimizer, env):
    global rendering
    global running

    batch_obvs = []
    batch_policy_actions = []
    batch_weights = []
    batch_returns = []
    batch_lens = []

    pixels_diff, reward, done = env.step()
    ep_rewards = []
    done = False

    BATCH_SIZE = 1000
    BUFFER_SIZE = 200

    ALPHA = 0.95

    score_loss = 0

    saved_r_model = RewardNetwork().to(device)
    saved_r_model.load_state_dict(r_net.state_dict())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rendering = not rendering

        first_obvs = pixels_diff.unsqueeze(0)
        
        batch_obvs.append(first_obvs)

        # get the action

        #act_probs = p_net(first_obvs.to(device))
        #act = torch.multinomial(act_probs, 1).item()

        # Get rewards estimate from critic
        pred_rewards = r_net(first_obvs.to(device))[0]
        act = torch.argmax(pred_rewards)

        pixels_diff, reward, done = env.step(act, rendering)
        second_obvs = pixels_diff.unsqueeze(0).detach().clone()


        memory.append((first_obvs[0], second_obvs[0], torch.tensor(act), torch.tensor(reward)))

        batch_weights.append(pred_rewards[act].item())

        if rendering:
            print(pred_rewards)

        if done or len(batch_obvs) >= BATCH_SIZE:
            ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
            batch_returns.append(ep_return)
            batch_lens.append(ep_len)

            #import pdb; pdb.set_trace()

            # Reset the environment for another episode
            #pixels_diff, reward, done = env.reset()
            done = False
            ep_rewards = []

            if len(batch_obvs) >= BATCH_SIZE:
                break

        if len(memory) >= BUFFER_SIZE:
            batch = random.sample(memory, BUFFER_SIZE)
            batch_first_obvs, batch_after_obvs, batch_actions, rewards = map(torch.stack, zip(*batch))
            # Do optimizer step for reward network
            r_optimizer.zero_grad()

            predicted_scores = r_net(batch_first_obvs.to(device))

            predicted_and_chosen_scores = predicted_scores.gather(
                1, 
                batch_actions.unsqueeze(-1).to(device),
            )

            shifted_predicted_scores = r_net(batch_after_obvs.to(device))
            best_actions = torch.argmax(shifted_predicted_scores, axis=1)

            old_preds = saved_r_model(batch_after_obvs.to(device)).gather(
                1,
                best_actions.unsqueeze(-1).to(device),
            )

            actual_scores = torch.tensor(rewards).to(device) + ALPHA * old_preds[0]

            loss_fn = torch.nn.MSELoss()
            score_loss = loss_fn(predicted_and_chosen_scores, actual_scores)

            #import pdb; pdb.set_trace()

            score_loss.backward()
            r_optimizer.step()

    # save changes to value net
    r_net.load_state_dict(saved_r_model.state_dict())

    # Do optimizer step for policy network
    p_optimizer.zero_grad()

    #import pdb; pdb.set_trace()
    batch_loss = compute_loss(
        p_net, 
        torch.cat(batch_obvs, dim=0).to(device), 
        torch.tensor(batch_policy_actions).to(device), 
        torch.tensor(batch_weights).to(device),
    )

    batch_loss.backward()
    p_optimizer.step()

    if rendering:
        pass
        #import pdb; pdb.set_trace()

    return batch_loss, score_loss, batch_returns, batch_lens


#LR = 1e-3
# init model
policy_model = PolicyNetwork()

reward_model = RewardNetwork()

#loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.SmoothL1Loss()
policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)
reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)

# initialize device
device = torch.device(args.device)
reward_model.to(device)
policy_model.to(device)


if __name__=="__main__":
    envs = PongEnv()
    env.reset()

    i = 0
    while running:
        i += 1
        batch_loss, score_loss, batch_returns, batch_lens = train_one_epoch(
            policy_model, 
            policy_optimizer, 
            reward_model, 
            reward_optimizer, 
            env,
        )
        print('epoch: %3d \t loss: %.3f \t score_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, score_loss, np.mean(batch_returns), np.mean(batch_lens)))
        

    # Done! Time to quit.
    pygame.quit()

# Save model to file
torch.save(model.state_dict(), 'model.pt')
