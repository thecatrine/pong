from re import A
import pygame
import torch
import numpy as np

import collections as c
import random
import argparse
import collections

from datetime import datetime
import matplotlib.pyplot as plt

import concurrent.futures
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
writer = SummaryWriter()
#model stuff

# input - [paddle x, ball x, ball y, ball vx, ball vy]
# output - [left, still, right]

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='model file to load')
parser.add_argument('--device', type=str, default='cuda', help='device to use')
args = parser.parse_args()

logdir = "runs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

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
        self.old_game_state = None

        return self.step()

    def step(self, action=0, render=False):
        reward = 0.0
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
                reward = 1.0
            else:
                # Reset the ball
                self.ball_loc[1] = 0
                reward = -1.0


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

        game_state = torch.tensor([self.ball_loc[0], self.ball_loc[1], self.ball_speed[0], self.ball_speed[1], self.paddle_loc[0]])

        if self.old_game_state is None:
            self.old_game_state = game_state

        pixels_to_return = torch.tensor(
            [self.ball_loc[0], self.ball_loc[1], self.ball_speed[0], self.ball_speed[1], self.paddle_loc[0],
            ],
            dtype=torch.float)

        #import pdb; pdb.set_trace()

        pixels_to_return = torch.cat((pixels_to_return, self.old_game_state))

        self.old_pixels = pixels
        self.old_game_state = game_state
        
        # return values are (state, reward, done)
        #if reward != 0:
        #    import pdb; pdb.set_trace()
        return pixels_to_return, reward, done
        

class RewardNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # taken from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
        # no idea if they're good here, should think this through
        #self.first = torch.nn.Conv2d(2, 32, kernel_size=8, stride=4)
        #self.second = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.third = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.first = torch.nn.Linear(10, 32)
        self.second = torch.nn.Linear(32, 64)

        #self.linear = torch.nn.Linear(1024, 256)
        self.linear = torch.nn.Linear(64, 256)
        self.reward_head_1 = torch.nn.Linear(256, 3)
        #self.reward_head_2 = torch.nn.Linear(16, 3)

        self.policy_head_1 = torch.nn.Linear(256, 3)
       # self.policy_head_2 = torch.nn.Linear(16, 3)

    def forward(self, x):
        x = torch.nn.functional.silu(self.first(x))
        x = torch.nn.functional.silu(self.second(x))
        #x = torch.nn.functional.silu(self.third(x))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.silu(self.linear(x))
       
        return x

    def reward(self, x ):
        x = self.reward_head_1(x)
        #x = self.reward_head_2(x)
        return x

    def policy(self, x ):
        x = self.policy_head_1(x)
        #x = self.policy_head_2(x)
        
        return torch.softmax(x, dim=-1)

    


def compute_loss(net, obvs, actions, rewards):
    res = net.policy(net(obvs))
    #import pdb; pdb.set_trace()
    logprobs = torch.log(res.gather(1, actions))

    if logprobs.shape != rewards.shape:
        import pdb; pdb.set_trace()

    return -torch.mean(logprobs * rewards)

EPISODE_LENGTH = 33


def gradient_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
    return total_norm

# globals
rendering = True
running = True


NUM_THREADS = 16

def train_one_epoch(r_net, r_optimizer, envs):
    global rendering
    global running
    
    ep_rewards = []


    BATCH_SIZE = 1000

    ALPHA = 0.95

    score_loss = 0
    actor_loss = 0

    saved_r_model = RewardNetwork().to(device)
    saved_r_model.load_state_dict(r_net.state_dict())

    batch_lens = []

    reporting_actor_loss = []
    reporting_score_loss = []
    reporting_entropy_loss = []
    reporting_rewards = []

    all_first_obvs =  [[] for _ in range(NUM_THREADS)]
    all_after_obvs =  [[] for _ in range(NUM_THREADS)]
    all_actions =  [[] for _ in range(NUM_THREADS)]
    all_after_actions = [[] for _ in range(NUM_THREADS)]
    all_rewards =  [[] for _ in range(NUM_THREADS)]

    state = [None for _ in range(NUM_THREADS)]
    
    for env_i in range(NUM_THREADS):
        state[env_i] = envs[env_i].step()[0]

    for step in range(EPISODE_LENGTH):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rendering = not rendering
       
        first_obvs = torch.stack(state).to(device)
        x = r_net(first_obvs)
        act_probs = r_net.policy(x)
        
        # take a random action 3% of the time
        act = torch.multinomial(act_probs, 1)

       
        if rendering:
            print(
                "probs: ", act_probs[0].cpu().detach().numpy(),
                "action: ", act[0].cpu().detach().numpy(),
                "pred scores: ", r_net.reward(x)[0].cpu().detach().numpy(),
            )

        for env_i in range(NUM_THREADS):
            new_state, r, d = envs[env_i].step(act[env_i], render=(rendering and env_i == 0))
            
            all_first_obvs[env_i].append(state[env_i].to(device))
            all_after_obvs[env_i].append(new_state.to(device))
            all_actions[env_i].append(act[env_i].item())
            
            if step != 0:
                all_after_actions[env_i].append(act[env_i].item())

            #if r != 0:
            #    import pdb; pdb.set_trace()
            all_rewards[env_i].append(r)
            state[env_i] = new_state

    # do one last step for the last state
    first_obvs = torch.stack(state)
    x = r_net(first_obvs.to(device))
    act_probs = r_net.policy(x)
    act = torch.multinomial(act_probs, 1)
    for env_i in range(NUM_THREADS):
        all_after_actions[env_i].append(act[env_i].item())
        
    # Now do training

    batch_lens.append(EPISODE_LENGTH * NUM_THREADS)

    first_obvs = torch.cat([torch.stack(x) for x in all_first_obvs], dim=0)
    after_obvs = torch.cat([torch.stack(x) for x in all_after_obvs], dim=0)
    actions = torch.cat([torch.tensor(x).to(device) for x in all_actions], dim=0).unsqueeze(-1)
    after_actions = torch.cat([torch.tensor(x).to(device) for x in all_after_actions], dim=0).unsqueeze(-1)
    after_obvs_batches = torch.stack([torch.stack(x) for x in all_after_obvs])
    last_step = after_obvs_batches[:, -1, :]
    last_x = saved_r_model(last_step)
    last_guess = saved_r_model.reward(last_x)
    last_action_probs = r_net.policy(last_x)
    last_action = torch.multinomial(last_action_probs, 1)
    last_value = torch.gather(last_guess, dim=1, index = last_action)
    
    orig_rewards = torch.stack([torch.tensor(x).to(device) for x in all_rewards])
    rewards = orig_rewards.clone()

    #normalize rewards
    rewards = torch.nn.functional.normalize(rewards, dim=-1)

    rewards[:, -1] += last_value[:, 0]
    for step in range(EPISODE_LENGTH-2, -1, -1):
        rewards[:, step] = rewards[:, step] + ALPHA * rewards[:, step+1]

    
    rewards = rewards.view(-1, 1)

    # have to do rollout reward logic specially
    r_optimizer.zero_grad()
    with autograd.detect_anomaly():
        predicted_scores = r_net.reward(r_net(first_obvs))

        #import pdb; pdb.set_trace()
        predicted_and_chosen_scores = predicted_scores.gather(1, actions)

        old_preds = saved_r_model.reward(
            saved_r_model(after_obvs.to(device))
        ).gather(
            1,
            after_actions.to(device),
        )


        actual_scores = rewards

        loss_fn = torch.nn.SmoothL1Loss()
        score_loss = loss_fn(predicted_and_chosen_scores, actual_scores)

        # Actor

        advantage = actual_scores - predicted_and_chosen_scores.detach()

        #import pdb; pdb.set_trace()
        actor_loss = compute_loss(
            r_net,
            first_obvs, 
            actions, 
            advantage.to(device),
        )

        # Entropy Loss

        act_probs2 = r_net.policy(r_net(first_obvs))
        act_probs2 = act_probs2.gather(1, actions)

        entropy_loss = torch.mean(act_probs2 * torch.log(act_probs2))
            
        total_loss = 1*score_loss + 0.5*actor_loss + 2*entropy_loss

        total_loss.backward()

    torch.nn.utils.clip_grad_norm_(r_net.parameters(), 200)

    r_optimizer.step()

    reporting_actor_loss.append(actor_loss.item())
    reporting_score_loss.append(score_loss.item())
    reporting_entropy_loss.append(entropy_loss.item())
    reporting_rewards.append(orig_rewards.sum())

    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()

    return torch.mean(torch.tensor(reporting_actor_loss), dtype=torch.float), \
        torch.mean(torch.tensor(reporting_score_loss), dtype=torch.float),    \
        torch.mean(torch.tensor(reporting_entropy_loss), dtype=torch.float),  \
        torch.sum(torch.tensor(reporting_rewards), dtype=torch.float), \
        torch.mean(torch.tensor(batch_lens, dtype=torch.float))


#LR = 1e-3
# init model

reward_model = RewardNetwork()

#loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.SmoothL1Loss()
reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=3e-4)

# initialize device
device = torch.device(args.device)
reward_model.to(device)


if __name__=="__main__":
    envs = [PongEnv() for i in range(NUM_THREADS)]
    for i in range(NUM_THREADS):
        envs[i].reset()
        [envs[i].step() for j in range(i*EPISODE_LENGTH)]

    i = 0
    while running:
        i += 1
        actor_loss, score_loss, entropy_loss, rewards, batch_lens = train_one_epoch(
            reward_model, 
            reward_optimizer, 
            envs,
        )
        print('epoch: %3d \t actor_loss: %.3f \t score_loss: %.3f \t entropy_loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, actor_loss, score_loss, entropy_loss, rewards, batch_lens))
        writer.add_scalar('actor_loss', actor_loss, i)
        writer.add_scalar('score_loss', score_loss, i)
        writer.add_scalar('entropy_loss', entropy_loss, i)
        writer.add_scalar('rewards', rewards, i)
        
        

    # Done! Time to quit.
    pygame.quit()

# Save model to file
torch.save(reward_model.state_dict(), 'model.pt')
