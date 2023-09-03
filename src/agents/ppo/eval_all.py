"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

import os

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch
from tqdm import tqdm
import yaml

from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
FLAGS = flags.FLAGS


def get_all_policy_paths(logdir):
  files = [
      entry for entry in os.listdir(logdir)
      if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))

  paths = []
  for entry in files:
    if entry.startswith("model"):
      paths.append(os.path.join(logdir, entry))
  return paths


def main(argv):
  del argv  # unused

  device = "cpu"

  # Load config and policy
  config_path = os.path.join(FLAGS.logdir, "config.yaml")
  policy_paths = get_all_policy_paths(FLAGS.logdir)

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.environment.terrain.type = "plane"
    config.environment.jumping_distance_schedule = [1.]
    config.environment.max_jumps = 6
    for idx in range(len(config.environment.rewards)):
      if config.environment.rewards[idx][0] == "stepping_freq":
        config.environment.rewards[idx] = ("stepping_freq", 0.)

  env = config.env_class(num_envs=1,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=False)
  env = env_wrappers.RangeNormalize(env)

  # Retrieve policy

  results = dict(policy_path=[], end_position=[], total_reward=[])
  for policy_path in tqdm(policy_paths[:110]):
    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    # runner.alg.actor_critic.train()

    # Reset environment
    state, _ = env.reset()
    total_reward = torch.zeros(1, device=device)
    steps_count = 0

    logs = []
    with torch.no_grad():
      while True:
        steps_count += 1
        action = policy(state).detach()
        # action = runner.alg.act(state, state)
        state, _, reward, done, info = env.step(action)
        # print(f"Ang Vel: {env.robot.base_angular_velocity_world_frame}")
        # print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")
        # print(f"Desired contact: {env.gait_generator.desired_contact_state}")
        # print(f"Foot contact: {env.robot.foot_contacts}")
        # print(f"Contact force: {env.robot.foot_contact_forces}")
        # print(env.robot.base_position)
        # input("Any Key...")

        total_reward += reward
        logs.extend(info["logs"])
        # print(f"Per-step time: {time.time() - start_time}")
        if done.any():
          # print(info["episode"])
          break

    results["policy_path"].append(policy_path)
    results["end_position"].append(logs[-2]["base_position"].cpu().numpy()[0])
    results["total_reward"].append(total_reward.detach().cpu().numpy()[0])
  with open(os.path.join(FLAGS.logdir, "all_evals.npz"), "wb") as f:
    np.savez(f, **results)


if __name__ == "__main__":
  app.run(main)
