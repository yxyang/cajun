"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

from src.envs import env_wrappers
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
  files = [
      entry for entry in os.listdir(logdir)
      if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  for entry in files:
    if entry.startswith("model"):
      return os.path.join(logdir, entry)
  raise ValueError("No Valid Policy Found.")


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  if FLAGS.logdir.endswith("pt"):
    config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
    policy_path = FLAGS.logdir
    root_path = os.path.dirname(FLAGS.logdir)
  else:
    # Find the latest policy ckpt
    config_path = os.path.join(FLAGS.logdir, "config.yaml")
    policy_path = get_latest_policy_path(FLAGS.logdir)
    root_path = FLAGS.logdir

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.environment.jumping_distance_schedule = [1., 0.3]
    # config.environment.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 6
    config.environment.max_jumps = 6

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
        not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env, config.training, policy_path, device=device)
  runner.load(policy_path)
  policy = runner.get_inference_policy()
  runner.alg.actor_critic.train()

  # Reset environment
  state, _ = env.reset()
  total_reward = torch.zeros(FLAGS.num_envs, device=device)
  steps_count = 0

  start_time = time.time()
  logs = []
  with torch.inference_mode():
    while True:
      steps_count += 1
      action = policy(state)
      state, _, reward, done, info = env.step(action)
      print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")


      total_reward += reward
      logs.extend(info["logs"])
      if done.any():
        print(info["episode"])
        break

  print(f"Total reward: {total_reward}")
  print(f"Time elapsed: {time.time() - start_time}")
  if FLAGS.use_real_robot or FLAGS.save_traj:
    mode = "real" if FLAGS.use_real_robot else "sim"
    output_dir = (
        f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
    output_path = os.path.join(root_path, output_dir)

    with open(output_path, "wb") as fh:
      pickle.dump(logs, fh)
    print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
