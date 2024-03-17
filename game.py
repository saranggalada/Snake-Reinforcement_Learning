import numpy as np
import random
import pygame
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED1 = (200,0,0)
RED2 = (255, 0, 0)
GREEN1 = (0, 230, 0)
GREEN2 = (0, 255, 100)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

PIXEL_SIZE = 20
GAME_SPEED = 20

class SnakeGameRL:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.game_reset()

    # To reset the game
    def game_reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-PIXEL_SIZE, self.head.y),
                      Point(self.head.x-(2*PIXEL_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    # To place the food
    def _place_food(self):
        x = random.randint(0, (self.w-PIXEL_SIZE )//PIXEL_SIZE )*PIXEL_SIZE 
        y = random.randint(0, (self.h-PIXEL_SIZE )//PIXEL_SIZE )*PIXEL_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    # To play a step of the game
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_danger() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(GAME_SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    # To check if there is a collision
    def is_danger(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - PIXEL_SIZE or point.x < 0 or point.y > self.h - PIXEL_SIZE or point.y < 0:
            return True
        # bites itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        i=0
        for pixel in self.snake:
            if i == 0:
                pygame.draw.rect(self.display, RED1, pygame.Rect(pixel.x, pixel.y, PIXEL_SIZE, PIXEL_SIZE))
                pygame.draw.rect(self.display, RED2, pygame.Rect(pixel.x+4, pixel.y+4, 12, 12))
                i+=1
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pixel.x, pixel.y, PIXEL_SIZE, PIXEL_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pixel.x+4, pixel.y+4, 12, 12))
            
        pygame.draw.rect(self.display, GREEN1, pygame.Rect(self.food.x, self.food.y, PIXEL_SIZE, PIXEL_SIZE))
        pygame.draw.rect(self.display, GREEN2, pygame.Rect(self.food.x+4, self.food.y+4, 12, 12))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # straight: no change in direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # turn right
        else: # turn left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += PIXEL_SIZE
        elif self.direction == Direction.LEFT:
            x -= PIXEL_SIZE
        elif self.direction == Direction.DOWN:
            y += PIXEL_SIZE
        elif self.direction == Direction.UP:
            y -= PIXEL_SIZE
            
        self.head = Point(x, y)