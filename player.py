import pygame
import torch

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

running = True
while running:
    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 0), ball_loc, 5)
    pygame.draw.rect(screen, (0, 0, 0), (paddle_loc[0]-paddle_width/2, paddle_loc[1], paddle_width, paddle_height))

    # Flip the display
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                moving_left = True
            if event.key == pygame.K_RIGHT:
                moving_right = True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                moving_left = False
            if event.key == pygame.K_RIGHT:
                moving_right = False

    if moving_left:
        paddle_loc[0] -= 5
    if moving_right:
        paddle_loc[0] += 5

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
