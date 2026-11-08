import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import unittest
import pygame
from play_catch import CatchEnv, QAgent, CatchApp, evaluate


class CatchTests(unittest.TestCase):
    def test_boundaries_and_rewards(self):
        env = CatchEnv(1)
        env.basket = env.apple = 0
        for _ in range(8):
            _, reward, done = env.step(0)
        self.assertEqual(env.basket, 0)
        self.assertTrue(done)
        self.assertEqual(reward, 10)
        with self.assertRaises(RuntimeError):
            env.step(1)
        env.reset()
        env.basket, env.apple = 8, 0
        for _ in range(8):
            _, reward, done = env.step(2)
        self.assertEqual(env.basket, 8)
        self.assertEqual(reward, -10)

    def test_terminal_update(self):
        agent = QAgent(1)
        agent.q[(0, 0, 8)] = [100, 100, 100]
        agent.learn((0, 0, 7), 1, 10, (0, 0, 8), True)
        self.assertEqual(agent.values((0, 0, 7))[1], 2)

    def test_learning_beats_random(self):
        agent, env = QAgent(7), CatchEnv(17)
        for _ in range(30000):
            agent.train_episode(env)
        baseline = evaluate(agent, random_policy=True)
        trained = evaluate(agent)
        print(f'Random: {baseline:.1%}; trained: {trained:.1%}')
        self.assertGreater(trained, 0.90)
        self.assertGreater(trained, baseline + 0.60)

    def test_timer_pause_restart_and_screens(self):
        app = CatchApp()
        try:
            app.draw()
            app.start('human')
            app.update(0.1)
            self.assertAlmostEqual(app.remaining, 59.9)
            app.paused = True
            app.update(5)
            self.assertAlmostEqual(app.remaining, 59.9)
            app.draw()
            app.paused = False
            app.remaining = 0.05
            app.update(0.1)
            self.assertEqual(app.remaining, 0)
            self.assertEqual(app.page, 'results')
            app.draw()
            app.start('ai')
            self.assertEqual(app.remaining, 60)
            self.assertEqual(app.score, 0)
            app.draw()
            pygame.image.save(app.screen, 'catch_lab_preview.png')
            app.activate('train')
            app.update(0.1)
            app.draw()
        finally:
            pygame.quit()


if __name__ == '__main__':
    unittest.main()
