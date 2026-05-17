import numpy as np
from collections import deque

# Game settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_SPEED = 3
PLAYER_SIZE = 8

# Direction vectors (up, right, down, left)
DIRECTIONS = [
    (0, -1),  # 0: up
    (1, 0),   # 1: right
    (0, 1),   # 2: down
    (-1, 0)   # 3: left
]


class Player:
    def __init__(self, x, y, direction, name):
        self.x = x
        self.y = y
        self.direction = direction  # 0-3 index in DIRECTIONS
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


class TronEnv:
    def __init__(self, render=False, grid_size=(80, 60)):
        """
        ML-friendly Tron environment
        
        Args:
            render: If True, enables pygame rendering (for testing/visualization)
            grid_size: Size of grid for state representation (width, height)
        """
        self.render = render
        self.grid_size = grid_size
        self.grid_width, self.grid_height = grid_size
        
        # Initialize pygame if rendering
        if self.render:
            import pygame
            self.pygame = pygame
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Tron ML Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            self.pygame = None
            self.screen = None
            self.clock = None
            self.font = None

        # Colors
        self.BLACK = (0, 0, 0)
        self.CYAN = (0, 255, 255)
        self.MAGENTA = (255, 0, 255)
        self.WHITE = (255, 255, 255)
        self.GRID_COLOR = (0, 50, 50)

        self.reset()

    def reset(self):
        """Reset the environment for a new episode"""
        # Initialize players on opposite sides
        self.player1 = Player(
            SCREEN_WIDTH * 0.25,
            SCREEN_HEIGHT / 2,
            1,  # facing right
            'Player 1'
        )

        self.player2 = Player(
            SCREEN_WIDTH * 0.75,
            SCREEN_HEIGHT / 2,
            3,  # facing left
            'Player 2'
        )

        self.done = False
        self.winner = None
        self.step_count = 0
        self.max_steps = 2000  # Prevent infinite games

        return self.get_state()

    def step(self, action1, action2=None):
        """
        Execute one step in the environment
        
        Args:
            action1: Action for player 1 [0=straight, 1=turn_left, 2=turn_right]
            action2: Action for player 2 (optional, if None, player 2 uses simple heuristic)
        
        Returns:
            observation: Current state
            reward: Reward for player 1
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            return self.get_state(), 0, True, {}

        self.step_count += 1

        # Execute actions
        if action1 == 1:
            self.player1.turn_left()
        elif action1 == 2:
            self.player1.turn_right()
        # action1 == 0 means go straight (no turn)

        # Player 2 action (either provided or simple heuristic)
        if action2 is not None:
            if action2 == 1:
                self.player2.turn_left()
            elif action2 == 2:
                self.player2.turn_right()
            # action2 == 0 means go straight (no turn)
        else:
            # Simple heuristic: try to avoid walls
            self._simple_heuristic(self.player2)

        # Move players
        self.player1.move()
        self.player2.move()

        # Check collisions
        player1_died = self.player1.check_collision(self.player2)
        player2_died = self.player2.check_collision(self.player1)

        # Determine outcome
        reward = 0
        if player1_died and player2_died:
            self.done = True
            self.winner = 'draw'
            reward = -100  # Draw is same as loss
        elif player1_died:
            self.player1.alive = False
            self.done = True
            self.winner = 'player2'
            reward = -100  # Loss
        elif player2_died:
            self.player2.alive = False
            self.done = True
            self.winner = 'player1'
            reward = 100  # Win
        elif self.step_count >= self.max_steps:
            self.done = True
            self.winner = 'draw'
            reward = -10  # Timeout is slightly bad

        # Small survival reward
        if not self.done:
            reward += 1.0

        # Render if enabled
        if self.render:
            self._render()

        info = {
            'winner': self.winner,
            'step_count': self.step_count,
            'player1_alive': self.player1.alive,
            'player2_alive': self.player2.alive
        }

        return self.get_state(), reward, self.done, info

    def _simple_heuristic(self, player):
        """Simple heuristic for opponent AI"""
        # Check if moving straight is dangerous
        dx, dy = DIRECTIONS[player.direction]
        next_x = player.x + dx * PLAYER_SPEED * 5
        next_y = player.y + dy * PLAYER_SPEED * 5
        
        # If going to hit wall or trail, turn randomly
        if (next_x < 0 or next_x > SCREEN_WIDTH or 
            next_y < 0 or next_y > SCREEN_HEIGHT):
            import random
            if random.random() < 0.5:
                player.turn_left()
            else:
                player.turn_right()

    def get_state(self):
        """
        Get current state representation
        
        Returns:
            state: Dictionary containing different state representations
        """
        state = {
            'grid': self._get_grid_state(),
            'features': self._get_feature_state(),
            'vector': self._get_vector_state()
        }
        return state

    def _get_grid_state(self):
        """
        Get grid-based state representation
        
        Returns:
            grid: numpy array of shape (4, grid_height, grid_width)
                  channels: [player1_trail, player2_trail, player1_head, player2_head]
        """
        grid = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        
        # Scale factors
        scale_x = self.grid_width / SCREEN_WIDTH
        scale_y = self.grid_height / SCREEN_HEIGHT
        
        # Add player 1 trail
        for x, y in self.player1.trail:
            gx = int(x * scale_x)
            gy = int(y * scale_y)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                grid[0, gy, gx] = 1.0
        
        # Add player 2 trail
        for x, y in self.player2.trail:
            gx = int(x * scale_x)
            gy = int(y * scale_y)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                grid[1, gy, gx] = 1.0
        
        # Add player 1 head
        gx1 = int(self.player1.x * scale_x)
        gy1 = int(self.player1.y * scale_y)
        if 0 <= gx1 < self.grid_width and 0 <= gy1 < self.grid_height:
            grid[2, gy1, gx1] = 1.0
        
        # Add player 2 head
        gx2 = int(self.player2.x * scale_x)
        gy2 = int(self.player2.y * scale_y)
        if 0 <= gx2 < self.grid_width and 0 <= gy2 < self.grid_height:
            grid[3, gy2, gx2] = 1.0
        
        return grid

    def _get_feature_state(self):
        """
        Get feature-based state representation (like snake-ai)
        
        Returns:
            features: numpy array of feature values
        
        Feature explanation:
        - Positions (p1_x_norm, p1_y_norm, p2_x_norm, p2_y_norm): 
          Normalized coordinates [0,1] of both players. Useful for knowing where you are on the map.
        - Directions (p1_dir, p2_dir): 
          One-hot encoding of current facing direction [up, right, down, left]. 
          Essential for knowing which way you're moving.
        - Distance to danger (dist_straight, dist_right, dist_left):
          Normalized distance [0,1] to nearest wall/trail in each direction.
          0 = immediate danger, 1 = far away. Critical for survival.
        - Relative opponent position (rel_x, rel_y): 
          COMMENTED OUT - Distance/direction to opponent. 
          May not be useful for basic survival since trails are the main danger, not the opponent directly.
          Can be uncommented later for more advanced strategies like trapping opponent.
        """
        # Normalize positions to [0, 1]
        p1_x_norm = self.player1.x / SCREEN_WIDTH
        p1_y_norm = self.player1.y / SCREEN_HEIGHT
        p2_x_norm = self.player2.x / SCREEN_WIDTH
        p2_y_norm = self.player2.y / SCREEN_HEIGHT
        
        # Relative position to opponent
        # COMMENTED OUT: Training basic survival first without opponent position info
        # rel_x = (self.player2.x - self.player1.x) / SCREEN_WIDTH
        # rel_y = (self.player2.y - self.player1.y) / SCREEN_HEIGHT
        
        # Directions (one-hot)
        p1_dir = [0, 0, 0, 0]
        p1_dir[self.player1.direction] = 1
        p2_dir = [0, 0, 0, 0]
        p2_dir[self.player2.direction] = 1
        
        # Distance to danger in each direction (normalized 0-1, 0=immediate danger, 1=far)
        dist_straight = self._get_distance_to_danger(self.player1, self.player1.direction)
        dist_right = self._get_distance_to_danger(self.player1, (self.player1.direction + 1) % 4)
        dist_left = self._get_distance_to_danger(self.player1, (self.player1.direction + 3) % 4)
        
        features = np.array([
            p1_x_norm, p1_y_norm,
            p2_x_norm, p2_y_norm,
            *p1_dir,
            *p2_dir,
            dist_straight, dist_right, dist_left,
            self.player1.alive, self.player2.alive
        ], dtype=np.float32)
        
        return features

    def _get_vector_state(self):
        """
        Get simple vector state (positions and directions only)
        
        Returns:
            vector: numpy array of basic state info
        """
        vector = np.array([
            self.player1.x / SCREEN_WIDTH,
            self.player1.y / SCREEN_HEIGHT,
            self.player1.direction / 4,
            self.player2.x / SCREEN_WIDTH,
            self.player2.y / SCREEN_HEIGHT,
            self.player2.direction / 4,
            (self.player2.x - self.player1.x) / SCREEN_WIDTH,
            (self.player2.y - self.player1.y) / SCREEN_HEIGHT
        ], dtype=np.float32)
        
        return vector

    def _get_distance_to_danger(self, player, direction):
        """
        Calculate distance to nearest danger (wall or trail) in given direction
        Returns normalized distance 0-1 where 0=immediate danger, 1=far
        """
        dx, dy = DIRECTIONS[direction]
        
        # Check distance to wall
        if dx > 0:  # Moving right
            wall_dist = (SCREEN_WIDTH - player.x) / SCREEN_WIDTH
        elif dx < 0:  # Moving left
            wall_dist = player.x / SCREEN_WIDTH
        elif dy > 0:  # Moving down
            wall_dist = (SCREEN_HEIGHT - player.y) / SCREEN_HEIGHT
        else:  # Moving up
            wall_dist = player.y / SCREEN_HEIGHT
        
        # Check distance to own trail
        trail_dist = 1.0  # Default to far
        for tx, ty in player.trail:
            # Check if trail point is in the direction we're moving
            to_trail_x = tx - player.x
            to_trail_y = ty - player.y
            
            # Check if trail is in front of us (same direction)
            if (dx > 0 and to_trail_x > 0) or (dx < 0 and to_trail_x < 0) or \
               (dy > 0 and to_trail_y > 0) or (dy < 0 and to_trail_y < 0):
                
                # Calculate distance to this trail point
                dist = ((to_trail_x**2 + to_trail_y**2)**0.5) / max(SCREEN_WIDTH, SCREEN_HEIGHT)
                trail_dist = min(trail_dist, dist)
        
        # Check distance to opponent's trail
        for tx, ty in self.player2.trail:
            to_trail_x = tx - player.x
            to_trail_y = ty - player.y
            
            if (dx > 0 and to_trail_x > 0) or (dx < 0 and to_trail_x < 0) or \
               (dy > 0 and to_trail_y > 0) or (dy < 0 and to_trail_y < 0):
                
                dist = ((to_trail_x**2 + to_trail_y**2)**0.5) / max(SCREEN_WIDTH, SCREEN_HEIGHT)
                trail_dist = min(trail_dist, dist)
        
        # Return minimum distance (normalized)
        return min(wall_dist, trail_dist)

    def _render(self):
        """Render the game using pygame"""
        if not self.render or self.screen is None:
            return

        # Fill background
        self.screen.fill(self.BLACK)

        # Draw grid
        for x in range(0, SCREEN_WIDTH, 40):
            self.pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, 40):
            self.pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (SCREEN_WIDTH, y), 1)

        # Draw player 1 trail
        if len(self.player1.trail) > 1:
            points = list(self.player1.trail)
            self.pygame.draw.lines(self.screen, self.CYAN, False, points, PLAYER_SIZE)

        # Draw player 2 trail
        if len(self.player2.trail) > 1:
            points = list(self.player2.trail)
            self.pygame.draw.lines(self.screen, self.MAGENTA, False, points, PLAYER_SIZE)

        # Draw player 1 head
        if self.player1.alive:
            self.pygame.draw.circle(self.screen, self.WHITE, (int(self.player1.x), int(self.player1.y)), PLAYER_SIZE // 2 + 2)

        # Draw player 2 head
        if self.player2.alive:
            self.pygame.draw.circle(self.screen, self.WHITE, (int(self.player2.x), int(self.player2.y)), PLAYER_SIZE // 2 + 2)

        # Draw info
        text = self.font.render(f"Step: {self.step_count}", True, self.WHITE)
        self.screen.blit(text, (20, 20))

        if self.done:
            winner_text = f"Winner: {self.winner.upper()}"
            text = self.font.render(winner_text, True, self.WHITE)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))

        self.pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """Clean up resources"""
        if self.render and self.pygame:
            self.pygame.quit()


if __name__ == "__main__":
    # Test the environment
    env = TronEnv(render=True)
    
    state = env.reset()
    print(f"Initial state shapes:")
    print(f"  Grid: {state['grid'].shape}")
    print(f"  Features: {state['features'].shape}")
    print(f"  Vector: {state['vector'].shape}")
    
    # Run a few random steps
    import random
    for _ in range(100):
        action1 = random.randint(0, 1)
        action2 = random.randint(0, 1)
        state, reward, done, info = env.step(action1, action2)
        
        if done:
            print(f"Game over! Winner: {info['winner']}")
            env.reset()
    
    env.close()
