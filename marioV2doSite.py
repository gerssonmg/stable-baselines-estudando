# Import GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Frame Stacker Wrapper Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# Wrapping the environment

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v3')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Greyscale the environment
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Create the stacked frames
env = VecFrameStack(env, 4,channels_order='last')

# env.seed(0)

state = env.reset()
state.shape()
