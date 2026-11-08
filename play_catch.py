"""Catch Lab: a timed game and tabular Q-learning demonstration."""
import argparse
import random
from collections import deque

import pygame

COLS = ROWS = 9
ACTIONS = (-1, 0, 1)
ROUND_SECONDS = 60
STEP_SECONDS = 0.20
BG = (12, 19, 34)
CARD = (23, 34, 53)
INK = (235, 242, 252)
MUTED = (151, 169, 195)
TEAL = (66, 220, 182)
RED = (255, 111, 124)


class CatchEnv:
    """One episode is one apple. Each action moves the basket then drops it."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    @property
    def state(self):
        return self.basket, self.apple, self.row

    def reset(self):
        self.basket = self.rng.randrange(COLS)
        self.apple = self.rng.randrange(COLS)
        self.row = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the environment before the next episode")
        if action not in range(3):
            raise ValueError("Action must be 0 (left), 1 (stay), or 2 (right)")
        self.basket = max(0, min(COLS - 1, self.basket + ACTIONS[action]))
        self.row += 1
        self.done = self.row == ROWS - 1
        reward = (10 if self.basket == self.apple else -10) if self.done else 0
        return self.state, reward, self.done


class QAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.q = {}
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 1.0
        self.episodes = 0
        self.recent = deque(maxlen=500)

    def values(self, state):
        return self.q.setdefault(state, [0.0, 0.0, 0.0])

    def action(self, state, explore=False):
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(3)
        values = self.values(state)
        best = max(values)
        return self.rng.choice([i for i, value in enumerate(values) if value == best])

    def learn(self, state, action, reward, next_state, done):
        # Terminal transitions have no future value.
        target = reward if done else reward + self.gamma * max(self.values(next_state))
        values = self.values(state)
        values[action] += self.alpha * (target - values[action])

    def train_episode(self, env):
        state = env.reset()
        while True:
            action = self.action(state, explore=True)
            next_state, reward, done = env.step(action)
            self.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        self.episodes += 1
        self.epsilon = max(0.03, self.epsilon * 0.9998)
        self.recent.append(reward > 0)


def evaluate(agent, episodes=2000, seed=42, random_policy=False):
    env = CatchEnv(seed)
    rng = random.Random(seed)
    caught = 0
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = rng.randrange(3) if random_policy else agent.action(state)
            state, reward, done = env.step(action)
            if done:
                caught += reward > 0
                break
    return caught / episodes


class CatchApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((960, 720))
        pygame.display.set_caption("Catch Lab | Q-Learning Project")
        self.clock = pygame.time.Clock()
        self.fonts = {size: pygame.font.SysFont("segoeui", size, bold=size >= 28)
                      for size in (16, 18, 22, 28, 40)}
        self.agent = QAgent(7)
        self.training_env = CatchEnv(17)
        self.env = CatchEnv()
        self.page = "menu"
        self.duration = ROUND_SECONDS
        self.remaining = float(self.duration)
        self.paused = False
        self.buttons = []
        self.target_episodes = 30000
        self.score = self.caught = self.missed = 0
        self.accumulator = 0.0
        self.feedback = "Catch apples. Build a smarter agent."
        self.mode = "human"
        self.running = True

    def text(self, text, x, y, size=18, color=INK):
        self.screen.blit(self.fonts[size].render(str(text), True, color), (x, y))

    def button(self, label, rect, action, primary=False):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        color = TEAL if primary else ((53, 73, 99) if hover else (37, 53, 77))
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        surface = self.fonts[18].render(label, True, BG if primary else INK)
        self.screen.blit(surface, surface.get_rect(center=rect.center))
        self.buttons.append((rect, action))

    def start(self, mode):
        self.mode = mode
        self.env.reset()
        self.score = self.caught = self.missed = 0
        self.remaining = float(self.duration)
        self.accumulator = 0.0
        self.paused = False
        self.page = "game"
        self.feedback = "Left / right arrows or A / D to move" if mode == "human" else "AI chooses the action with the highest Q-value"

    def activate(self, action):
        if action == "human":
            self.start("human")
        elif action == "ai":
            if self.agent.episodes:
                self.start("ai")
        elif action == "train":
            self.target_episodes = self.agent.episodes + 30000
            self.page = "training"
        elif action == "menu":
            self.page = "menu"
        elif action == "pause":
            self.paused = not self.paused
        elif action == "restart":
            self.start(self.mode)
        elif action.startswith("time:"):
            self.duration = int(action.split(":")[1])

    def update(self, dt):
        if self.page == "training":
            for _ in range(min(200, self.target_episodes - self.agent.episodes)):
                self.agent.train_episode(self.training_env)
            if self.agent.episodes >= self.target_episodes:
                self.page = "menu"
            return
        if self.page != "game" or self.paused:
            return
        active_dt = min(dt, self.remaining)
        self.remaining = max(0.0, self.remaining - active_dt)
        self.accumulator += active_dt
        while self.accumulator + 1e-9 >= STEP_SECONDS:
            self.accumulator -= STEP_SECONDS
            if self.mode == "ai":
                action = self.agent.action(self.env.state)
            else:
                keys = pygame.key.get_pressed()
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                action = 1 + int(right) - int(left)
            _, reward, done = self.env.step(action)
            if done:
                self.score += reward
                self.caught += reward > 0
                self.missed += reward < 0
                self.feedback = "Nice catch!  +10 points" if reward > 0 else "Missed apple  -10 points"
                self.env.reset()
        if self.remaining <= 0:
            self.page = "results"

    def draw(self):
        self.buttons = []
        self.screen.fill(BG)
        self.text("CATCH LAB", 36, 24, 40)
        self.text("REINFORCEMENT LEARNING / TABULAR Q-LEARNING", 38, 76, 16, TEAL)
        if self.page in ("menu", "training"):
            pygame.draw.rect(self.screen, CARD, (36, 120, 888, 540), border_radius=20)
            self.text("Play. Train. Watch it learn.", 66, 151, 28)
            self.text("Catch the falling apple with your basket before it reaches the floor.", 66, 199)
            self.text("A catch earns +10. A miss costs 10. Every round has a timer.", 66, 230, color=MUTED)
            self.text("ROUND LENGTH", 66, 286, 16, TEAL)
            for i, seconds in enumerate((30, 60, 90)):
                self.button(f"{seconds} seconds", (66 + i * 152, 318, 136, 44), f"time:{seconds}", self.duration == seconds)
            self.text(f"Training episodes: {self.agent.episodes:,}", 66, 399, 22)
            rate = sum(self.agent.recent) / len(self.agent.recent) if self.agent.recent else 0
            self.text(f"Recent training catches: {rate:.0%}   |   Exploration: {self.agent.epsilon:.1%}", 66, 436, color=MUTED)
            if self.page == "training":
                self.text("Training agent... the window stays responsive.", 66, 491, 22, TEAL)
                progress = 1 - (self.target_episodes - self.agent.episodes) / 30000
                pygame.draw.rect(self.screen, BG, (66, 536, 810, 12), border_radius=6)
                pygame.draw.rect(self.screen, TEAL, (66, 536, int(810 * progress), 12), border_radius=6)
                self.button("Stop training", (66, 576, 220, 48), "menu")
            else:
                self.button("Play yourself", (66, 512, 246, 54), "human", True)
                self.button("Train AI (+30,000)", (330, 512, 260, 54), "train")
                self.button("Watch AI" if self.agent.episodes else "Train before watching", (608, 512, 268, 54), "ai")
                self.text("Controls: hold arrows / A D   |   P: pause   |   R: restart   |   Esc: menu", 66, 599, 16, MUTED)
        else:
            self.draw_game()
        self.text("9 x 9 environment   /   3 actions: left, stay, right   /   Reward: +10 or -10", 38, 685, 16, MUTED)
        pygame.display.flip()

    def draw_game(self):
        origin_x, origin_y, cell = 36, 160, 50
        self.text("HUMAN PLAY" if self.mode == "human" else "AI DEMONSTRATION", 36, 115, 22, TEAL)
        for row in range(ROWS):
            for col in range(COLS):
                rect = (origin_x + col * cell, origin_y + row * cell, cell - 2, cell - 2)
                pygame.draw.rect(self.screen, CARD if row < ROWS - 1 else (35, 67, 73), rect, border_radius=5)
        ax = origin_x + self.env.apple * cell + 24
        ay = origin_y + self.env.row * cell + 25
        pygame.draw.circle(self.screen, RED, (ax, ay), 17)
        pygame.draw.circle(self.screen, (255, 182, 185), (ax - 5, ay - 6), 5)
        pygame.draw.line(self.screen, TEAL, (ax, ay - 15), (ax + 5, ay - 23), 4)
        bx = origin_x + self.env.basket * cell
        pygame.draw.rect(self.screen, TEAL, (bx + 5, origin_y + 8 * cell + 16, 39, 27), border_radius=8)
        pygame.draw.line(self.screen, INK, (bx + 5, origin_y + 8 * cell + 16), (bx + 44, origin_y + 8 * cell + 16), 3)
        pygame.draw.rect(self.screen, CARD, (514, 120, 410, 490), border_radius=20)
        self.text("TIME REMAINING", 540, 145, 16, MUTED)
        seconds = max(0, int(self.remaining + 0.999))
        self.text(f"{seconds // 60:02}:{seconds % 60:02}", 540, 169, 40, RED if seconds <= 10 else TEAL)
        pygame.draw.rect(self.screen, BG, (540, 230, 356, 8), border_radius=4)
        pygame.draw.rect(self.screen, TEAL, (540, 230, int(356 * self.remaining / self.duration), 8), border_radius=4)
        self.text(f"Score   {self.score}", 540, 260, 28)
        self.text(f"Caught: {self.caught}       Missed: {self.missed}", 540, 308, 22)
        total = self.caught + self.missed
        self.text(f"Catch rate: {self.caught / total:.0%}" if total else "Catch rate: --", 540, 345, color=MUTED)
        self.text("AGENT INSPECTOR", 540, 401, 16, TEAL)
        self.text(f"State (basket, apple, row): {self.env.state}", 540, 431, 16)
        values = self.agent.values(self.env.state)
        self.text("Q-values:  " + "   ".join(f"{name} {v:.1f}" for name, v in zip(("L", "S", "R"), values)), 540, 461, 16)
        self.text(f"Episodes: {self.agent.episodes:,}", 540, 493, 16, MUTED)
        self.text("Alpha 0.20  |  Gamma 0.95", 540, 522, 16, MUTED)
        self.text("AI demo uses greedy actions; learning is off.", 540, 554, 16, MUTED)
        self.text(self.feedback, 36, 625, 16, TEAL)
        self.button("Menu", (786, 628, 138, 40), "menu")
        self.button("Resume" if self.paused else "Pause", (630, 628, 138, 40), "pause")
        if self.paused or self.page == "results":
            shade = pygame.Surface((960, 720), pygame.SRCALPHA)
            shade.fill((5, 10, 22, 210))
            self.screen.blit(shade, (0, 0))
            self.buttons = []
            pygame.draw.rect(self.screen, CARD, (220, 210, 520, 300), border_radius=22)
            ended = self.page == "results"
            self.text("Round complete!" if ended else "Game paused", 258, 240, 40)
            self.text(f"Score: {self.score}   |   Caught: {self.caught}   |   Missed: {self.missed}", 258, 310, 22)
            self.text("The timer stops while paused." if not ended else "Train the agent and compare its catch rate!", 258, 355, 18, MUTED)
            self.button("Play again" if ended else "Resume", (258, 420, 210, 50), "restart" if ended else "pause", True)
            self.button("Main menu", (488, 420, 210, 50), "menu")

    def run(self):
        self.draw()
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, action in self.buttons:
                        if rect.collidepoint(event.pos):
                            self.activate(action)
                            break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.page = "menu"
                    elif event.key == pygame.K_p and self.page == "game":
                        self.paused = not self.paused
                    elif event.key == pygame.K_r and self.page in ("game", "results"):
                        self.start(self.mode)
            if self.running:
                self.update(dt)
                self.draw()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Train and compare AI with random actions without opening a window")
    args = parser.parse_args()
    if args.evaluate:
        agent = QAgent(7)
        env = CatchEnv(17)
        for _ in range(30000):
            agent.train_episode(env)
        print(f"Random policy catch rate: {evaluate(agent, random_policy=True):.1%}")
        print(f"Trained Q-learning catch rate: {evaluate(agent):.1%}")
    else:
        CatchApp().run()
