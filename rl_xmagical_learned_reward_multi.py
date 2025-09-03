# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""X-MAGICAL: Train a policy with a learned reward."""

import os
import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from torchkit.experiment import string_from_kwargs
from torchkit.experiment import unique_id
import torch.multiprocessing as mp
import torch.distributed as dist
import yaml
import torch

import os

import sys
import train_policy_multi as train_policy_multi
from absl import app as absl_app
# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_path", None, "Path to pretraining experiment.")
flags.DEFINE_list("seeds", [0, 5], "List specifying the range of seeds to run.")
flags.DEFINE_string("name_test", "xmagical", "Name of the test to run.")
flags.DEFINE_string("port_number", "20500", "Port number for distributed training.")

def run_seed(seed, kwargs, world_size, env_name, experiment_name):
    """Run a single seed distributed across GPUs."""
    # Launch torchrun for this seed
    port = int(FLAGS.port_number) + (seed * 10)  # Unique port for each seed
    cmd = [
        "torchrun",
        f"--nprocs_per_node={world_size}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{port}",
        "--max-restarts", "0",
        "train_policy_multi.py",
        "--experiment_name", f"{experiment_name}_seed{seed}",
        "--env_name", env_name,
        "--config", f"configs/xmagical/rl/env_reward.py:{kwargs['embodiment']}",
        "--config.reward_wrapper.pretrained_path", FLAGS.pretrained_path,
        "--config.reward_wrapper.type", "holdr",
        "--seed", str(seed),
        "--wandb"
    ]
    return subprocess.Popen(cmd)


def main(_):

  with open(os.path.join(FLAGS.pretrained_path, "metadata.yaml"), "r") as fp:
    kwargs = yaml.load(fp, Loader=yaml.FullLoader)

  reward_type = "holdr"  
  
  # if kwargs["algo"] == "goal_classifier":
  #   reward_type = "goal_classifier"
  # elif kwargs["algo"] == "holdr":
  #     reward_type = "holdr"    
  # elif kwargs["algo"] == "reds":
  #   reward_type = "reds"
  # else:
  #   reward_type = "distance_to_goal"

  # Map the embodiment to the x-MAGICAL env name.
  env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[kwargs["embodiment"]]
  
  # Generate experiment name deterministically based on input parameters
  # This ensures all DDP processes use the same experiment name
  import hashlib
  # Create a deterministic hash based on pretrained_path and current parameters
  hash_input = f"{FLAGS.pretrained_path}_{env_name}_{reward_type}_{kwargs['mode']}_{kwargs['algo']}_{FLAGS.name_test}"
  deterministic_uid = hashlib.md5(hash_input.encode()).hexdigest()[:8]
  
  experiment_name = string_from_kwargs(
        env_name=env_name,
        reward="learned",
        reward_type=reward_type,
        mode=kwargs["mode"],
        algo=kwargs["algo"],
        uid=deterministic_uid,  # Use deterministic UID instead of random
    )
  logging.info("Experiment name: %s", experiment_name)
  
  world_size = torch.cuda.device_count()
  seeds = range(0, int(FLAGS.seeds[0]))
  processes = []

  for seed in seeds:
        p = run_seed(seed, kwargs, world_size, env_name, experiment_name)
        processes.append(p)

    # Wait for all seeds to complete
  for p in processes:
      p.wait()
  # for seed in range(*list(map(int, FLAGS.seeds))):
  #   procs.append(
  #       subprocess.Popen([  # pylint: disable=consider-using-with
  #           "python",
  #           "train_policy.py",
  #           "--experiment_name",
  #           experiment_name,
  #           "--env_name",
  #           f"{env_name}",
  #           "--config",
  #           f"configs/xmagical/rl/env_reward.py:{kwargs['embodiment']}",
  #           "--config.reward_wrapper.pretrained_path",
  #           f"{FLAGS.pretrained_path}",
  #           "--config.reward_wrapper.type",
  #           f"{reward_type}",
  #           "--seed",
  #           f"{seed}",
  #           "--device",
  #           f"{FLAGS.device}",
  #           "--wandb",
  #           f"{FLAGS.wandb}",
  #       ]))

  # Wait for each seed to terminate.
  # for p in procs:
  #   p.wait()


if __name__ == "__main__":
  flags.mark_flag_as_required("pretrained_path")
  app.run(main)
