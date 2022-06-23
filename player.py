from turtle import forward
import pygame
import torch

#model stuff

# input - [paddle x, ball x, ball y, ball vx, ball vy]
# output - [left, still, right]

class GameModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(5, 10)
        self.second = torch.nn.Linear(10, 10)
        self.third = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = torch.nn.functional.silu(self.first(x))
        x = torch.nn.functional.silu(self.second(x))
        x = self.third(x)

        return torch.nn.functional.softmax(x, -1)

#LR = 1e-3
# init model
model = GameModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

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
    ml_control = False

    running = True

    while running:
        # Fill the background with white
        screen.fill((255, 255, 255))

        # Draw a solid blue circle in the center
        pygame.draw.circle(screen, (0, 0, 0), ball_loc, 5)
        pygame.draw.rect(screen, (0, 0, 0), (paddle_loc[0]-paddle_width/2, paddle_loc[1], paddle_width, paddle_height))

        # Flip the display
        pygame.display.flip()

        # predict player action
        in_vec = torch.tensor([[
            ((paddle_loc[0]/WIDTH)-0.5),
            ((ball_loc[0]/WIDTH)-0.5),
            ((ball_loc[1]/HEIGHT)-0.5),
            ((ball_speed[0]/5)-0.5),
            ((ball_speed[1]/5)-0.5),
        ]])

        out_vec = model(in_vec)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    moving_left = True
                if event.key == pygame.K_RIGHT:
                    moving_right = True
                if event.key == pygame.K_SPACE:
                    ml_control = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    moving_left = False
                if event.key == pygame.K_RIGHT:
                    moving_right = False
                if event.key == pygame.K_SPACE:
                    ml_control = False

        ai_moving_left = False
        ai_moving_right = False

        if ml_control:
            # overwrite player action
            action_to_pick = torch.argmax(out_vec)

            print(out_vec)
            print("Picked ", action_to_pick)
            print

            if action_to_pick == 0:
                pass
            elif action_to_pick == 1:
                ai_moving_left = True
                ai_moving_right = False
            elif action_to_pick == 2:
                ai_moving_left = False
                ai_moving_right = True
        else:
            # we're training
            if not (moving_left ^ moving_right):
                player_action = 0
            elif moving_left:
                player_action = 1
            elif moving_right:
                player_action = 2

            player_action = torch.Tensor([player_action]).to(torch.int64)

            #print(out_vec, player_action)
            optimizer.zero_grad()
            loss = loss_fn(out_vec, player_action)
            loss.backward()
            optimizer.step()

            print("Loss was ", loss.item())

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

        if ball_loc[1] > HEIGHT-paddle_height-5:
            if ball_loc[0] > paddle_loc[0] - paddle_width/2 and ball_loc[0] < paddle_loc[0] + paddle_width/2:
                ball_speed[1] = -ball_speed[1]
            else:
                ball_loc[1] = 0
        if ball_loc[1] < 0:
            ball_speed[1] = -ball_speed[1]
        if ball_loc[0] > WIDTH:
            ball_speed[0] = -ball_speed[0]
        if ball_loc[0] < 0:
            ball_speed[0] = -ball_speed[0]
        

    # Done! Time to quit.
    pygame.quit()
