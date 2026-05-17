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
        
        # Spatial hashing for collision detection (cell_size = 16)
        self.spatial_grid = {}
        
        # Fast direct-line lookup for _get_distance_to_danger
        self.x_to_y = {} # x -> set of y
        self.y_to_x = {} # y -> set of x

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
        point = (self.x, self.y)
        step_idx = len(self.trail)
        self.trail.append(point)
        
        # Update spatial hash grid: store (x, y, step_idx) to allow precise time-based self-collision skips
        gx = int(self.x / 16)
        gy = int(self.y / 16)
        cell = (gx, gy)
        if cell not in self.spatial_grid:
            self.spatial_grid[cell] = []
        self.spatial_grid[cell].append((self.x, self.y, step_idx))
        
        # Update fast direct-line lookup
        if self.x not in self.x_to_y:
            self.x_to_y[self.x] = set()
        self.x_to_y[self.x].add(self.y)
        
        if self.y not in self.y_to_x:
            self.y_to_x[self.y] = set()
        self.y_to_x[self.y].add(self.x)

    def check_collision(self, other_player):
        if not self.alive:
            return False

        # Check wall collision
        if self.x < 0 or self.x > SCREEN_WIDTH or self.y < 0 or self.y > SCREEN_HEIGHT:
            return True

        # Current step count of this player
        curr_step = len(self.trail)
        
        # Check own trail and other player's trail using spatial hash
        # Player head bounding box coordinates
        gx_min = int((self.x - PLAYER_SIZE) / 16)
        gx_max = int((self.x + PLAYER_SIZE) / 16)
        gy_min = int((self.y - PLAYER_SIZE) / 16)
        gy_max = int((self.y + PLAYER_SIZE) / 16)
        
        # Check own trail and other player's trail in overlapping cells
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                cell = (gx, gy)
                # Check own trail: skip the last 5 steps of history (curr_step - step_idx < 6) to allow normal 90-degree turns
                if cell in self.spatial_grid:
                    for px, py, step_idx in self.spatial_grid[cell]:
                        if curr_step - step_idx >= 6:
                            if self.check_point_collision_squared(px, py):
                                return True
                # Check other player's trail
                if cell in other_player.spatial_grid:
                    for px, py, *rest in other_player.spatial_grid[cell]:
                        if self.check_point_collision_squared(px, py):
                            return True

        return False

    def check_point_collision(self, px, py):
        dx = self.x - px
        dy = self.y - py
        distance = (dx * dx + dy * dy) ** 0.5
        return distance < PLAYER_SIZE

    def check_point_collision_squared(self, px, py):
        dx = self.x - px
        dy = self.y - py
        return (dx * dx + dy * dy) < (PLAYER_SIZE * PLAYER_SIZE)


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
            action1: Action for player 1 [0=straight, 1=turn_left, 2=turn_right, 'heuristic']
            action2: Action for player 2 [0=straight, 1=turn_left, 2=turn_right, 'heuristic', None]
        
        Returns:
            observation: Current state
            reward: Reward for player 1
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            return self.get_state(), 0, True, {}

        self.step_count += 1

        # Execute action 1
        if action1 == 'heuristic':
            self._simple_heuristic(self.player1, self.player2)
        elif action1 == 1:
            self.player1.turn_left()
        elif action1 == 2:
            self.player1.turn_right()

        # Execute action 2
        if action2 == 'heuristic':
            self._simple_heuristic(self.player2, self.player1)
        elif action2 is not None:
            if action2 == 1:
                self.player2.turn_left()
            elif action2 == 2:
                self.player2.turn_right()
        else:
            self._simple_heuristic(self.player2, self.player1)

        # Move players
        self.player1.move()
        self.player2.move()

        # Check collisions
        player1_died = self.player1.check_collision(self.player2)
        player2_died = self.player2.check_collision(self.player1)

        # Determine outcome
        reward = 0
        reward2 = 0
        if player1_died and player2_died:
            self.done = True
            self.winner = 'none'
            reward = -100  # Both lost
            reward2 = -100
        elif player1_died:
            self.player1.alive = False
            self.done = True
            self.winner = 'player2'
            reward = -100  # Loss
            reward2 = 100
        elif player2_died:
            self.player2.alive = False
            self.done = True
            self.winner = 'player1'
            reward = 100  # Win
            reward2 = -100
        elif self.step_count >= self.max_steps:
            self.done = True
            self.winner = 'none'
            reward = -100  # Timeout is a loss for both
            reward2 = -100

        # Small survival reward
        if not self.done:
            reward += 5
            reward2 += 5
            
            # Tiny turn bonus to encourage exploration of turning instead of going straight
            if action1 in (1, 2):
                reward += 0.5
            if action2 in (1, 2):
                reward2 += 0.5

        # Render if enabled
        if self.render:
            self._render()

        info = {
            'winner': self.winner,
            'step_count': self.step_count,
            'player1_alive': self.player1.alive,
            'player2_alive': self.player2.alive,
            'player2_reward': reward2,
            'player2_state': self.get_state(player_id=2)
        }

        return self.get_state(player_id=1), reward, self.done, info

    def _simple_heuristic(self, player, opponent):
        """
        Simple Heuristic Opponent using direct direction-based danger avoidance:
        - If danger close forward: Turn to the safer side (away from nearest danger).
        - If danger close right: Move forward (if safe) or turn left.
        - If danger close left: Move forward (if safe) or turn right.
        """
        THRESHOLD = 0.02
        
        dist_straight = self._get_distance_to_danger(player, opponent, player.direction)
        dist_left = self._get_distance_to_danger(player, opponent, (player.direction + 3) % 4)
        dist_right = self._get_distance_to_danger(player, opponent, (player.direction + 1) % 4)
        
        # 1. Danger close forward: Must turn!
        if dist_straight < THRESHOLD:
            if dist_left > dist_right:
                player.turn_left()
            else:
                player.turn_right()
                
        # 2. Danger close right: Move forward (if safe) or turn left
        elif dist_right < THRESHOLD:
            if dist_straight < THRESHOLD:
                player.turn_left()
            # Else, keep going straight (default)
            
        # 3. Danger close left: Move forward (if safe) or turn right
        elif dist_left < THRESHOLD:
            if dist_straight < THRESHOLD:
                player.turn_right()
            # Else, keep going straight (default)

    def get_state(self, player_id=1):
        """
        Get current state representation
        
        Args:
            player_id: 1 for player1's perspective, 2 for player2's perspective
            
        Returns:
            state: Dictionary containing different state representations
        """
        if player_id == 1:
            p1, p2 = self.player1, self.player2
        else:
            p1, p2 = self.player2, self.player1
            
        state = {
            'grid': self._get_grid_state(p1, p2),
            'features': self._get_feature_state(p1, p2),
            'vector': self._get_vector_state(p1, p2)
        }
        return state

    def _get_grid_state(self, p1, p2):
        """
        Get grid-based state representation
        
        Returns:
            grid: numpy array of shape (4, grid_height, grid_width)
                  channels: [p1_trail, p2_trail, p1_head, p2_head]
        """
        grid = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        
        # Scale factors
        scale_x = self.grid_width / SCREEN_WIDTH
        scale_y = self.grid_height / SCREEN_HEIGHT
        
        # Add player 1 trail
        for x, y in p1.trail:
            gx = int(x * scale_x)
            gy = int(y * scale_y)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                grid[0, gy, gx] = 1.0
        
        # Add player 2 trail
        for x, y in p2.trail:
            gx = int(x * scale_x)
            gy = int(y * scale_y)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                grid[1, gy, gx] = 1.0
        
        # Add player 1 head
        gx1 = int(p1.x * scale_x)
        gy1 = int(p1.y * scale_y)
        if 0 <= gx1 < self.grid_width and 0 <= gy1 < self.grid_height:
            grid[2, gy1, gx1] = 1.0
            
        # Add player 2 head
        gx2 = int(p2.x * scale_x)
        gy2 = int(p2.y * scale_y)
        if 0 <= gx2 < self.grid_width and 0 <= gy2 < self.grid_height:
            grid[3, gy2, gx2] = 1.0
        
        return grid

    def _get_feature_state(self, p1, p2):
        """
        Get feature-based state representation with spatial awareness
        
        Features (22 total):
        - Positions: p1_x, p1_y, p2_x, p2_y (4)
        - Relative opponent: rel_x, rel_y (2)
        - Directions: p1_dir one-hot (4), p2_dir one-hot (4)
        - Danger distances: straight, right, left, behind (4)
        - Spatial: flood_fill_self, flood_fill_opponent, flood_fill_ratio (3)
        - Game state: step_progress (1)
        
        Returns:
            features: numpy array of 22 feature values
        """
        # Normalize positions to [0, 1]
        p1_x_norm = p1.x / SCREEN_WIDTH
        p1_y_norm = p1.y / SCREEN_HEIGHT
        p2_x_norm = p2.x / SCREEN_WIDTH
        p2_y_norm = p2.y / SCREEN_HEIGHT
        
        # Relative position to opponent (helps with trapping strategies)
        rel_x = (p2.x - p1.x) / SCREEN_WIDTH
        rel_y = (p2.y - p1.y) / SCREEN_HEIGHT
        
        # Directions (one-hot)
        p1_dir = [0, 0, 0, 0]
        p1_dir[p1.direction] = 1
        p2_dir = [0, 0, 0, 0]
        p2_dir[p2.direction] = 1
        
        # Distance to danger in all 4 directions (normalized 0-1, 0=immediate danger, 1=far)
        dist_straight = self._get_distance_to_danger(p1, p2, p1.direction)
        dist_right = self._get_distance_to_danger(p1, p2, (p1.direction + 1) % 4)
        dist_left = self._get_distance_to_danger(p1, p2, (p1.direction + 3) % 4)
        dist_behind = self._get_distance_to_danger(p1, p2, (p1.direction + 2) % 4)
        
        # Flood fill: how many cells can each player reach? (spatial awareness)
        flood_self, flood_opp = self._get_flood_fill_counts(p1, p2)
        max_cells = self._ff_grid_w * self._ff_grid_h
        flood_self_norm = flood_self / max_cells
        flood_opp_norm = flood_opp / max_cells
        # Ratio: >0.5 means we have more space, <0.5 means opponent has more
        flood_ratio = flood_self / (flood_self + flood_opp) if (flood_self + flood_opp) > 0 else 0.5
        
        # Game progress (helps agent adjust strategy over time)
        step_progress = self.step_count / self.max_steps
        
        features = np.array([
            p1_x_norm, p1_y_norm,
            p2_x_norm, p2_y_norm,
            rel_x, rel_y,
            *p1_dir,
            *p2_dir,
            dist_straight, dist_right, dist_left, dist_behind,
            flood_self_norm, flood_opp_norm, flood_ratio,
            step_progress
        ], dtype=np.float32)
        
        return features

    def _get_vector_state(self, p1, p2):
        """
        Get simple vector state representation
        """
        vector = np.array([
            p1.x / SCREEN_WIDTH,
            p1.y / SCREEN_HEIGHT,
            p1.direction / 4,
            p2.x / SCREEN_WIDTH,
            p2.y / SCREEN_HEIGHT,
            p2.direction / 4,
            (p2.x - p1.x) / SCREEN_WIDTH,
            (p2.y - p1.y) / SCREEN_HEIGHT
        ], dtype=np.float32)
        
        return vector

    # Flood fill grid resolution (coarse for speed)
    _ff_grid_w = 40
    _ff_grid_h = 30
    
    def _get_flood_fill_counts(self, p1, p2):
        """
        Build a single occupancy grid and compute flood fill counts for both players.
        Saves redundant set creation and traversal work.
        """
        gw, gh = self._ff_grid_w, self._ff_grid_h
        scale_x = gw / SCREEN_WIDTH
        scale_y = gh / SCREEN_HEIGHT
        
        # Precompute walls set once and store on env if not present
        if not hasattr(self, '_ff_walls'):
            self._ff_walls = set()
            for x in range(gw):
                self._ff_walls.add((x, -1))
                self._ff_walls.add((x, gh))
            for y in range(gh):
                self._ff_walls.add((-1, y))
                self._ff_walls.add((gw, y))
                
        # Start with precomputed walls
        occupied = self._ff_walls.copy()
        
        # Add both players' trails
        for tx, ty in p1.trail:
            gx = int(tx * scale_x)
            gy = int(ty * scale_y)
            if 0 <= gx < gw and 0 <= gy < gh:
                occupied.add((gx, gy))
        
        for tx, ty in p2.trail:
            gx = int(tx * scale_x)
            gy = int(ty * scale_y)
            if 0 <= gx < gw and 0 <= gy < gh:
                occupied.add((gx, gy))
                
        # BFS function for a start position using the built occupancy grid
        def bfs_count(start_player):
            start_x = min(max(int(start_player.x * scale_x), 0), gw - 1)
            start_y = min(max(int(start_player.y * scale_y), 0), gh - 1)
            
            if (start_x, start_y) in occupied:
                return 0
                
            visited = {(start_x, start_y)}
            queue = [(start_x, start_y)]
            count = 0
            
            while queue:
                x, y = queue.pop(0)
                count += 1
                
                # Check neighbors
                for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                    if 0 <= nx < gw and 0 <= ny < gh and (nx, ny) not in visited and (nx, ny) not in occupied:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            return count

        return bfs_count(p1), bfs_count(p2)

    def _flood_fill_count(self, player, opponent):
        """Wrapper for backward compatibility"""
        self_count, opp_count = self._get_flood_fill_counts(player, opponent)
        return self_count

    def _get_distance_to_danger(self, player, opponent, direction):
        """
        Calculate distance to nearest danger (wall or trail) in given direction.
        Returns normalized distance 0-1 where 0=immediate danger, 1=far.
        Optimized by scanning outward along the same row/column and stopping at the first hit.
        """
        dx, dy = DIRECTIONS[direction]
        
        # 1. Wall distance (O(1) base check)
        if dx > 0:  # Moving right
            wall_dist = (SCREEN_WIDTH - player.x) / SCREEN_WIDTH
        elif dx < 0:  # Moving left
            wall_dist = player.x / SCREEN_WIDTH
        elif dy > 0:  # Moving down
            wall_dist = (SCREEN_HEIGHT - player.y) / SCREEN_HEIGHT
        else:  # Moving up
            wall_dist = player.y / SCREEN_HEIGHT
            
        # 2. Trail distance: Scan outward from player's cell
        max_cells_x = int(SCREEN_WIDTH / 16)
        max_cells_y = int(SCREEN_HEIGHT / 16)
        
        curr_gx = int(player.x / 16)
        curr_gy = int(player.y / 16)
        
        min_sq = float('inf')
        
        # Helper to check cells and update min_sq
        def check_cells(cells_to_check):
            nonlocal min_sq
            found = False
            for cell in cells_to_check:
                # Check player's trail
                if cell in player.spatial_grid:
                    for tx, ty, *rest in player.spatial_grid[cell]:
                        to_trail_x = tx - player.x
                        to_trail_y = ty - player.y
                        if (dx > 0 and to_trail_x > 0) or (dx < 0 and to_trail_x < 0) or \
                           (dy > 0 and to_trail_y > 0) or (dy < 0 and to_trail_y < 0):
                            dist_sq = to_trail_x * to_trail_x + to_trail_y * to_trail_y
                            if dist_sq < min_sq:
                                min_sq = dist_sq
                                found = True
                # Check opponent's trail
                if cell in opponent.spatial_grid:
                    for tx, ty, *rest in opponent.spatial_grid[cell]:
                        to_trail_x = tx - player.x
                        to_trail_y = ty - player.y
                        if (dx > 0 and to_trail_x > 0) or (dx < 0 and to_trail_x < 0) or \
                           (dy > 0 and to_trail_y > 0) or (dy < 0 and to_trail_y < 0):
                            dist_sq = to_trail_x * to_trail_x + to_trail_y * to_trail_y
                            if dist_sq < min_sq:
                                min_sq = dist_sq
                                found = True
            return found

        # Horizontal ray
        if dx != 0:
            gy_min = max(0, int((player.y - PLAYER_SIZE) / 16))
            gy_max = min(max_cells_y, int((player.y + PLAYER_SIZE) / 16))
            gy_range = list(range(gy_min, gy_max + 1))
            
            # Scan outward column by column
            if dx > 0:  # Scan Right
                for gx in range(curr_gx, max_cells_x + 1):
                    cells = [(gx, gy) for gy in gy_range]
                    if check_cells(cells) and min_sq < float('inf'):
                        break  # Stop immediately! The nearest danger has been found.
            else:  # Scan Left
                for gx in range(curr_gx, -1, -1):
                    cells = [(gx, gy) for gy in gy_range]
                    if check_cells(cells) and min_sq < float('inf'):
                        break  # Stop immediately! The nearest danger has been found.
                        
        # Vertical ray
        else:
            gx_min = max(0, int((player.x - PLAYER_SIZE) / 16))
            gx_max = min(max_cells_x, int((player.x + PLAYER_SIZE) / 16))
            gx_range = list(range(gx_min, gx_max + 1))
            
            # Scan outward row by row
            if dy > 0:  # Scan Down
                for gy in range(curr_gy, max_cells_y + 1):
                    cells = [(gx, gy) for gx in gx_range]
                    if check_cells(cells) and min_sq < float('inf'):
                        break  # Stop immediately! The nearest danger has been found.
            else:  # Scan Up
                for gy in range(curr_gy, -1, -1):
                    cells = [(gx, gy) for gx in gx_range]
                    if check_cells(cells) and min_sq < float('inf'):
                        break  # Stop immediately! The nearest danger has been found.

        max_dist_val = max(SCREEN_WIDTH, SCREEN_HEIGHT)
        if min_sq < float('inf'):
            trail_dist = (min_sq ** 0.5) / max_dist_val
        else:
            trail_dist = 1.0
            
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
            self.pygame.display.quit()


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
