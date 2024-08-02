from copy import copy, deepcopy
from queue import Queue
import threading

import d4rl

import numpy as np
import torch

import os
import pickle
from typing import Dict, Any

class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )

class ReplayBufferCustom(object):
    """
    Simple replay buffer to store trajectories of the form (obs, act, next_obs, reward, done).
    Can be used to sample full trajectories or random transitions.
    The buffer is implemented as a circular buffer, so that old transitions are overwritten when the buffer is full.
    """
    def __init__(self, capacity: int, device: torch.device, file_path: os.PathLike, buffer_name: str = "buffer") -> None:
        """
        Parameters:
        -----------
        capacity: maximum number of transitions to store
        device: device to store the tensors on
        file_path: path to the folder where the replay buffer will be stored and loaded from
        buffer_name: name of the file where the replay buffer will be stored
        """
        self.capacity = capacity
        self.device = device
        self.file_path = file_path
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file_name = os.path.join(file_path, f"{buffer_name}.pkl")

        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = np.empty((capacity), dtype=np.float32)
        self.dones = np.empty((capacity), dtype=np.bool_)
        self.traj_starts = np.empty((capacity), dtype=np.int32)
        self.extra_info = []

        self.idx = 0
        self.full = False
    
    def __len__(self) -> int:
        """
        Returns the number of transitions currently stored in the buffer.
        """
        return self.capacity if self.full else self.idx
    
    def __str__(self) -> str:
        return f"ReplayBuffer with {len(self)} transitions stored."
    
    def _frame_to_tensor(self, frame: Any) -> torch.Tensor:
        """
        Converts a frame to a tensor and moves it to the device.
        """
        if isinstance(frame, np.ndarray):
            return torch.tensor(frame, device=self.device)
        elif isinstance(frame, torch.Tensor):
            return frame.to(self.device)
        elif isinstance(frame, list):
            try:
                return torch.tensor(np.array([self._frame_to_tensor(f) for f in frame]))
            except:
                return [self._frame_to_tensor(f) for f in frame]
        elif isinstance(frame, tuple):
            return tuple(self._frame_to_tensor(f) for f in frame)
        elif isinstance(frame, dict):
            return {k: self._frame_to_tensor(v) for k, v in frame.items()}
        elif isinstance(frame, np.dtype):
            return torch.tensor(frame, device=self.device)
        else:
            return frame

    def _add(self, obs: Any, act: np.ndarray, next_obs: Any, reward: float, done: bool, traj_start: int, extra_info: Dict = {}) -> None:
        """
        Add a transition to the replay buffer. Should not be called directly (unless trajectory_start is known).

        Parameters:
        -----------
        obs: observation (shape: obs_shape)
        act: action (shape: act_shape)
        next_obs: next observation (shape: obs_shape)
        reward: reward
        done: whether the episode is done
        traj_start: index of the first transition in the trajectory
        extra_info: extra information to store with the transition
        """
        if self.full:
            raise ValueError("Replay buffer is full, cannot add new transitions")
        
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(act)
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.traj_starts[self.idx] = traj_start
        self.extra_info.append(extra_info)

        self.idx += 1
        self.full = self.idx == self.capacity

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a full trajectory to the replay buffer.

        Parameters:
        -----------
        trajectory: tuple of (obs, act, next_obs, reward, done)
        """
        obs, act, next_obs, reward, done, extra_info = trajectory
        traj_start = self.idx
        for i in range(len(done)):
            if extra_info:
                self._add(obs[i], act[i], next_obs[i], reward[i], done[i], traj_start, extra_info[i])
            else:
                self._add(obs[i], act[i], next_obs[i], reward[i], done[i], traj_start)

    def sample(self, batch_size: int = None, sample_tensor: bool = False) -> tuple:
        """
        Samples a batch of transitions from the replay buffer.

        Parameters:
        -----------
        batch_size: number of transitions to sample. If None, load all transitions.
        sample_tensor: whether to convert the numpy arrays to torch tensors

        Returns:
        --------
        obs: observations (shape: (batch_size, *obs_shape))
        act: actions (shape: (batch_size, *act_shape))
        next_obs: next observations (shape: (batch_size, *obs_shape))
        reward: rewards (shape: (batch_size, 1))
        done: dones (shape: (batch_size, 1))
        extra: extra information (list of dictionaries)
        """
        if batch_size is None:
            idx = np.arange(self.idx)
        else:
            idx = np.random.randint(0, self.idx, size=batch_size)
        if sample_tensor:
            obs = self._frame_to_tensor([self.obs[i] for i in idx])
            next_obs = self._frame_to_tensor([self.next_obs[i] for i in idx])
            act = self._frame_to_tensor([self.actions[i] for i in idx])
            reward = torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32, device=self.device).unsqueeze(1)
            done = torch.tensor([self.dones[i] for i in idx], dtype=torch.bool, device=self.device).unsqueeze(1)
        else:
            obs = [self.obs[i] for i in idx]
            next_obs = [self.next_obs[i] for i in idx]
            act = [self.actions[i] for i in idx]
            reward = self.rewards[idx]
            done = self.dones[idx]

        extra = [self.extra_info[i] for i in idx]

        return obs, act, next_obs, reward, done, extra
    
    def sample_trajectory(self, sample_tensor: bool=False) -> tuple:
        """
        Samples a batch of full trajectories from the replay buffer.
        Full trajectories are defined by the traj_starts attribute and are contiguous in the buffer.

        Parameters:
        -----------
        sample_tensor: whether to convert the numpy arrays to torch tensors

        Returns:
        --------
        obs: observations (shape: (traj_len, *obs_shape))
        act: actions (shape: (traj_len, *act_shape))
        next_obs: next observations (shape: (traj_len, *obs_shape))
        reward: rewards (shape: (traj_len, 1))
        done: dones (shape: (traj_len, 1))
        extra: extra information (list of dictionaries)
        """
        idx = np.random.randint(0, self.idx)
        start = self.traj_starts[idx].squeeze()
        end = np.where(self.dones[start:] == 1)[0][0] + start + 1
        obs = self.obs[start:end]
        next_obs = self.next_obs[start:end]
        act = self.actions[start:end]
        reward = self.rewards[start:end]
        done = self.dones[start:end]
        
        if sample_tensor:
            obs = self._frame_to_tensor(obs)
            next_obs = self._frame_to_tensor(next_obs)
            act = self._frame_to_tensor(act)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(1)

        extra = [self.extra_info[i] for i in range(start, end)]

        return obs, act, next_obs, reward, done, extra
    
    def clear(self) -> None:
        """
        Clears the replay buffer.
        """
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = np.empty((self.capacity), dtype=np.float32)
        self.dones = np.empty((self.capacity), dtype=np.bool_)
        self.traj_starts = np.empty((self.capacity), dtype=np.int32)
        self.extra_info = []

        self.idx = 0
        self.full = False

    def save(self) -> None:
        """
        Saves the replay buffer to a file.
        """
        print(f"Saving replay buffer to {self.file_name}")
        pickle.dump(self.__dict__, open(self.file_name, "wb"))
        print(f"Replay buffer saved with {len(self)} transitions")

    def load(self) -> None:
        """
        Loads the replay buffer from a file.
        """
        print(f"Loading replay buffer from {self.file_name}")
        self.__dict__.update(pickle.load(open(self.file_name, "rb")))
        print(f"Replay buffer loaded with {len(self)} transitions")


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )

def get_custom_dataset(env):
    path = os.path.join("replay_buffers", env)
    buffer = ReplayBufferCustom(1000, torch.device("cpu"), file_path=path, buffer_name="train_buffer")
    buffer.load()
    
    return dict(
        observations=np.array(buffer.obs).astype(np.float32),
        actions=np.array(buffer.actions).astype(np.float32),
        next_observations=np.array(buffer.next_obs).astype(np.float32),
        rewards=buffer.rewards.astype(np.float32),
        dones=buffer.dones.astype(np.float32),
    )

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
