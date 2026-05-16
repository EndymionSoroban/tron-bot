import pygame
import sys
import time
from collections import deque

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_SPEED = 3
PLAYER_SIZE = 8

# Colors
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
GRID_COLOR = (0, 50, 50)

# Direction vectors (up, right, down, left)
DIRECTIONS = [
    (0, -1),  # 0: up
    (1, 0),   # 1: right
    (0, 1),   # 2: down
    (-1, 0)   # 3: left
]


class Player:
    def __init__(self, x, y, direction, color, name):
        self.x = x
        self.y = y
        self.direction = direction  # 0-3 index in DIRECTIONS
        self.color = color
        self.name = name
        self.alive = True
        self.trail = deque()  # Store (x, y) tuples

    def turn_left(self):
        self.direction = (self.direction + 3) % 4

    def turn_right(self):
        self.direction = (self.direction + 1) % 4

    def move(self):
        if not self.alive:
            return

        dx, dy = DIRECTIONS[self.direction]
        self.x += dx * PLAYER_SPEED
        self.y += dy * PLAYER_SPEED

        # Add trail point
        self.trail.append((self.x, self.y))

    def draw(self, surface):
        if not self.alive:
            return

        # Draw trail
        if len(self.trail) > 1:
            points = list(self.trail)
            pygame.draw.lines(surface, self.color, False, points, PLAYER_SIZE)

        # Draw player head
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), PLAYER_SIZE // 2 + 2)

    def check_collision(self, other_player):
        if not self.alive:
            return False

        # Check wall collision
        if self.x < 0 or self.x > SCREEN_WIDTH or self.y < 0 or self.y > SCREEN_HEIGHT:
            return True

        # Check collision with own trail (skip recent points to avoid self-collision on turn)
        trail_list = list(self.trail)
        for i in range(len(trail_list) - 10):
            if self.check_point_collision(trail_list[i][0], trail_list[i][1]):
                return True

        # Check collision with other player's trail
        for x, y in other_player.trail:
            if self.check_point_collision(x, y):
                return True

        return False

    def check_point_collision(self, px, py):
        dx = self.x - px
        dy = self.y - py
        distance = (dx * dx + dy * dy) ** 0.5
        return distance < PLAYER_SIZE


def draw_grid(surface):
    for x in range(0, SCREEN_WIDTH, 40):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
    for y in range(0, SCREEN_HEIGHT, 40):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)


def draw_text(surface, text, size, color, x, y, center=False):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    if center:
        rect = text_surface.get_rect(center=(x, y))
        surface.blit(text_surface, rect)
    else:
        surface.blit(text_surface, (x, y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tron Game - 2 Players")
    clock = pygame.time.Clock()

    # Scores
    scores = {'player1': 0, 'player2': 0}

    def init_game():
        # Initialize players on opposite sides
        player1 = Player(
            SCREEN_WIDTH * 0.25,
            SCREEN_HEIGHT / 2,
            1,  # facing right
            CYAN,
            'Player 1'
        )

        player2 = Player(
            SCREEN_WIDTH * 0.75,
            SCREEN_HEIGHT / 2,
            3,  # facing left
            MAGENTA,
            'Player 2'
        )

        return player1, player2

    player1, player2 = init_game()
    game_running = True
    game_over = False
    winner_text = ""

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_SPACE:
                        player1, player2 = init_game()
                        game_running = True
                        game_over = False
                else:
                    # Player 1 controls: A (left), D (right)
                    if event.key == pygame.K_a:
                        player1.turn_left()
                    elif event.key == pygame.K_d:
                        player1.turn_right()

                    # Player 2 controls: Arrow Left (left), Arrow Right (right)
                    if event.key == pygame.K_LEFT:
                        player2.turn_left()
                    elif event.key == pygame.K_RIGHT:
                        player2.turn_right()

        if game_running and not game_over:
            player1.move()
            player2.move()

            # Check collisions
            player1_died = player1.check_collision(player2)
            player2_died = player2.check_collision(player1)

            if player1_died and player2_died:
                game_over = True
                winner_text = "Draw!"
            elif player1_died:
                player1.alive = False
                scores['player2'] += 1
                game_over = True
                winner_text = "Player 2 Wins!"
            elif player2_died:
                player2.alive = False
                scores['player1'] += 1
                game_over = True
                winner_text = "Player 1 Wins!"

        # Draw everything
        screen.fill(BLACK)
        draw_grid(screen)
        player1.draw(screen)
        player2.draw(screen)

        # Draw scores
        draw_text(screen, f"Player 1: {scores['player1']}", 36, CYAN, 20, 20)
        draw_text(screen, f"Player 2: {scores['player2']}", 36, MAGENTA, SCREEN_WIDTH - 180, 20)

        # Draw controls
        draw_text(screen, "Player 1: A/D to turn", 24, CYAN, 20, SCREEN_HEIGHT - 40)
        draw_text(screen, "Player 2: <-/-> to turn", 24, MAGENTA, SCREEN_WIDTH - 220, SCREEN_HEIGHT - 40)

        # Draw game over screen
        if game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))

            # Winner text
            draw_text(screen, winner_text, 72, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, center=True)
            draw_text(screen, "Press SPACE to play again", 36, WHITE, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30, center=True)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
