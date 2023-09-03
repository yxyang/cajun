"""Example of the point-mass environment."""
from absl import app
from absl import flags
from absl import logging

from isaacgym.torch_utils import to_torch
from ml_collections import config_flags
import torch

from src.envs import jump_env

config_flags.DEFINE_config_file('config',
                                'src/envs/configs/single_jump.py',
                                'environment config file.')
flags.DEFINE_integer('num_envs', 10, "number of environments in parallel.")
flags.DEFINE_bool('use_gpu', True, 'whether to use gpu.')
flags.DEFINE_bool('show_gui', True, 'whether to show GUI.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  device = 'cuda' if FLAGS.use_gpu else 'cpu'

  env = jump_env.JumpEnv(num_envs=FLAGS.num_envs,
                         config=FLAGS.config,
                         device=device,
                         show_gui=FLAGS.show_gui)
  # action = torch.stack(
  #     [torch.tensor([0., 0., 0.25, 0., 0., 0.], device=device)] *
  #     FLAGS.num_envs)
  state, _ = env.reset()
  env.robot.render()
  # input("Any Key...")

  steps_count = 0
  sum_reward = torch.zeros(FLAGS.num_envs, device=device)
  action = (env.action_space[0] + env.action_space[1]) / 2
  action = torch.stack([action] * FLAGS.num_envs, dim=0)
  while True:
    steps_count += 1
    state, _, reward, done, info = env.step(action)
    print("Step: {}, Reward: {}".format(steps_count, reward))
    input("Any Key...")
    sum_reward += reward
    if done.any():
      break

  print("Total reward: {}".format(sum_reward.cpu().numpy()))


if __name__ == "__main__":
  app.run(main)
