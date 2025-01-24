# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for building blocks for actors and critics.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/replay_memory.py
"""

from typing import List
import numpy as np
import torch as th
from collections import deque, namedtuple

# `Transition` is a named tuple representing a single transition in our
# RL environment. All the other information is stored in the `info`, e.g.,
# `g_x`, `l_x`, `binary_cost`, `append`, and `latent`, etc. Note that we also
# require all the values to be np.ndarray or float. Here `a` is a dictionary.
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class Batch(object):

  def __init__(self, transitions: List[Transition], device: th.device):
    self.device = device
    batch = Transition(*zip(*transitions))

    # Reward and Done.
    self.reward = th.FloatTensor(batch.r).to(device)
    self.non_final_mask = th.BoolTensor(np.logical_not(np.asarray(batch.done))).to(device)

    # obsrv.
    self.non_final_obsrv_nxt = th.cat(batch.s_)[self.non_final_mask].to(device)
    self.obsrv = th.cat(batch.s).to(device)

    # Action.
    self.action = {}
    for key in batch.a[0].keys():
      self.action[key] = th.cat([a[key] for a in batch.a]).to(device)

    # Info.
    self.info = {}
    for key, value in batch.info[0].items():
      if isinstance(value, np.ndarray) or isinstance(value, float):
        self.info[key] = th.FloatTensor(np.asarray([info[key] for info in batch.info])).to(device)
    if 'append' in self.info:
      self.info['non_final_append_nxt'] = (batch.info['append_nxt'][self.non_final_mask])

  @staticmethod
  def concat(batches: List['Batch'], first_done_idxs: List[int]):
    # concatenates all batches (reward, done, obsrv, action, info) into one (but only up to the first done index per batch)
    # `first_done_idxs` is a list of the first done index for each batch.
    first_done_idxs = [x+1 if x != -1 else x for x in first_done_idxs]
    b0 = batches[0]
    b0.reward = th.cat([b.reward[:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)])
    b0.non_final_mask = th.cat([b.non_final_mask[:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)])
    b0.non_final_obsrv_nxt = th.cat([b.non_final_obsrv_nxt[:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)])
    b0.obsrv = th.cat([b.obsrv[:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)])
    b0.action = {key: th.cat([b.action[key][:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)]) for key in b0.action.keys()}
    b0.info = {key: th.cat([b.info[key][:first_done_idx] for b, first_done_idx in zip(batches, first_done_idxs)]) for key in b0.info.keys()}
    
    return b0

class ReplayMemory(object):

  def __init__(self, capacity, seed):
    self.reset(capacity)
    self.capacity = capacity
    self.seed = seed
    self.rng = np.random.default_rng(seed=self.seed)

  def reset(self, capacity):
    if capacity is None:
      capacity = self.capacity
    self.memory = deque(maxlen=capacity)

  def update(self, transition):
    self.memory.appendleft(transition)  # pop from right if full

  def sample(self, batch_size):
    length = len(self.memory)
    indices = self.rng.integers(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def sample_recent(self, batch_size, recent_size):
    recent_size = min(len(self.memory), recent_size)
    indices = self.rng.integers(low=0, high=recent_size, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    return len(self.memory)


# class RolloutMemory2(object):
#   def __init__(self, capacity, seed, n_envs=1):
#     capacity_per_env = capacity // n_envs
#     self.capacity = capacity
#     self.capacity_per_env = capacity_per_env
#     self.seed = seed
#     self.rng = np.random.default_rng(seed=self.seed)
#     self.n_envs = n_envs
#     self.first_done_idx = [-1]*n_envs
#     self.reset(capacity_per_env)

#   # TODO: add function to flush data, keeping partial rollouts that were unused in last update

#   def reset(self, capacity):
#     if capacity is None:
#       capacity = self.capacity_per_env
#     # TODO: consider using tensors directly for fast data flushing
#     self.memory = [deque(maxlen=capacity) for _ in range(self.n_envs)]

#   def update(self, env_idx, transition):
#     if transition.done:
#       if self.first_done_idx[env_idx] == -1:
#         self.first_done_idx[env_idx] = len(self.memory[env_idx])
#     self.memory[env_idx].append(transition)  # pop from left if full

#   def sample(self, batch_size):
#     length = len(self.memory[0])
#     indices = np.arange(length)
#     return [[self.memory[env_idx][i] for i in indices] for env_idx in range(self.n_envs)]

#   def __len__(self):
#     return len(self.memory[0])  # all envs have the same length

class RolloutMemory(object):
  def __init__(self, capacity, seed, n_envs=1, obsrv_dim=1, action_keys=[], action_dims=[]):
    capacity_per_env = capacity // n_envs
    self.capacity = capacity
    self.capacity_per_env = capacity_per_env
    self.seed = seed
    self.n_envs = n_envs
    self.obsrv_dim = obsrv_dim
    self.action_keys = action_keys
    self.action_dims = action_dims
    self.reset(capacity_per_env)

  def reset(self, capacity):
    if capacity is None:
      capacity = self.capacity_per_env
    self.reward = th.zeros((self.n_envs, capacity), dtype=th.float32)
    self.done = th.zeros((self.n_envs, capacity), dtype=th.bool)
    self.obsrv = th.zeros((self.n_envs, capacity, self.obsrv_dim), dtype=th.float32)
    self.obsrv_nxt = th.zeros((self.n_envs, capacity, self.obsrv_dim), dtype=th.float32)
    self.action = {k: th.zeros((self.n_envs, capacity, d), dtype=th.float32) for k, d in zip(self.action_keys, self.action_dims)}
    self.l_x = th.zeros((self.n_envs, capacity), dtype=th.float32)
    self.g_x = th.zeros((self.n_envs, capacity), dtype=th.float32)

    self.idx = th.zeros(self.n_envs, dtype=int)
    self.first_done_idx = th.full((self.n_envs,), -1)

  def update(self, env_idx, transition):
    idx = self.idx[env_idx]
    # don't update if idx is already at the end
    if idx >= self.capacity_per_env:
      return
    self.reward[env_idx, idx] = transition.r
    if type(transition.done) == np.bool_:
      self.done[env_idx, idx] = transition.done.item()
    else:
      self.done[env_idx, idx] = transition.done
    self.obsrv[env_idx, idx] = transition.s
    self.obsrv_nxt[env_idx, idx] = transition.s_
    for k, v in transition.a.items():
      self.action[k][env_idx, idx] = v
    self.l_x[env_idx, idx] = transition.info['l_x']
    self.g_x[env_idx, idx] = transition.info['g_x']
    
    self.idx[env_idx] += 1
    if transition.done:
      if self.first_done_idx[env_idx] == -1:
        self.first_done_idx[env_idx] = idx

  def process_data(self, device):
    obsrv = self.obsrv.to(device)
    obsrv_nxt = self.obsrv_nxt.to(device)
    reward = self.reward.to(device)
    done = self.done.to(device)
    action = {k: v.to(device) for k, v in self.action.items()}
    l_x = self.l_x.to(device)
    g_x = self.g_x.to(device)
    first_done_idx = self.first_done_idx.to(device)
    info = {'l_x': l_x, 'g_x': g_x, 'first_done_idx': first_done_idx}
    return obsrv, action, reward, obsrv_nxt, done, action, info

  def __len__(self):
    return len(self.obsrv[0])  # all envs have the same length