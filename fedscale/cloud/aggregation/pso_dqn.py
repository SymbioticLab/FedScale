import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn


class DqnNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_features = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            in_features = hidden
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DqnClientSelector:
    def __init__(
        self,
        client_ids: List[int],
        diversities: List[float],
        args,
        device: torch.device,
    ) -> None:
        self.client_ids = list(client_ids)
        self.client_index = {client_id: idx for idx, client_id in enumerate(self.client_ids)}
        self.num_clients = len(self.client_ids)
        self.num_features = 3
        self.state_size = self.num_clients * self.num_features
        self.device = device

        hidden_sizes = [args.pso_dqn_hidden1, args.pso_dqn_hidden2]
        self.model = DqnNetwork(self.state_size, hidden_sizes, self.num_clients).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pso_dqn_learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = args.pso_dqn_gamma
        self.epsilon = args.pso_dqn_epsilon
        self.epsilon_decay = args.pso_dqn_epsilon_decay
        self.min_epsilon = args.pso_dqn_min_epsilon
        self.replay_batch_size = args.pso_dqn_replay_batch
        self.memory = deque(maxlen=args.pso_dqn_memory_size)
        self.rng = random.Random(args.sample_seed)

        self.client_diversities = np.array(diversities, dtype=np.float32)
        self.last_local_losses = np.ones(self.num_clients, dtype=np.float32)
        self.contribution_scores = np.zeros(self.num_clients, dtype=np.float32)
        self.selection_counts = np.zeros(self.num_clients, dtype=np.float32)

        self.last_state: Optional[np.ndarray] = None
        self.last_selected_indices: Optional[List[int]] = None

    def update_diversities(self, diversities: List[float]) -> None:
        self.client_diversities = np.array(diversities, dtype=np.float32)

    def update_losses(self, client_ids: Iterable[int], local_losses: Iterable[float]) -> None:
        for client_id, loss in zip(client_ids, local_losses):
            idx = self.client_index.get(client_id)
            if idx is not None:
                self.last_local_losses[idx] = float(loss)

    def _get_state(self) -> np.ndarray:
        epsilon = 1e-9
        norm_diversities = self.client_diversities / (np.max(self.client_diversities) + epsilon)
        norm_losses = self.last_local_losses / (np.max(self.last_local_losses) + epsilon)
        norm_contributions = self.contribution_scores / (np.max(np.abs(self.contribution_scores)) + epsilon)
        state = (
            np.stack([norm_diversities, norm_losses, norm_contributions], axis=1)
            .flatten()
            .reshape(1, self.state_size)
        )
        return state.astype(np.float32)

    def select_clients(self, num_to_select: int, feasible_clients: List[int]) -> List[int]:
        if num_to_select <= 0:
            return []
        feasible_set = set(feasible_clients)
        state = self._get_state()
        self.last_state = state
        if len(feasible_set) <= num_to_select:
            selected_ids = list(feasible_clients)
            self.last_selected_indices = [self.client_index[cid] for cid in selected_ids]
            return selected_ids

        if self.rng.random() <= self.epsilon:
            selected_ids = self.rng.sample(list(feasible_set), num_to_select)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, device=self.device)).cpu().numpy().flatten()
            masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
            for client_id in feasible_set:
                idx = self.client_index.get(client_id)
                if idx is not None:
                    masked_q[idx] = q_values[idx]
            selected_indices = np.argsort(masked_q)[-num_to_select:]
            selected_ids = [self.client_ids[idx] for idx in selected_indices]

        self.last_selected_indices = [self.client_index[cid] for cid in selected_ids]
        return selected_ids

    def update_with_reward(self, reward: float) -> None:
        if self.last_state is None or self.last_selected_indices is None:
            return

        selected_losses = [self.last_local_losses[idx] for idx in self.last_selected_indices]
        for idx, _loss in zip(self.last_selected_indices, selected_losses):
            self.selection_counts[idx] += 1
            count = self.selection_counts[idx]
            old_score = self.contribution_scores[idx]
            self.contribution_scores[idx] = old_score + (1.0 / count) * (reward - old_score)

        next_state = self._get_state()
        self.memory.append((self.last_state, self.last_selected_indices, reward, next_state))
        if len(self.memory) >= self.replay_batch_size:
            self._replay()

    def _replay(self) -> None:
        batch = self.rng.sample(self.memory, self.replay_batch_size)
        states = torch.tensor(np.vstack([b[0] for b in batch]), device=self.device)
        next_states = torch.tensor(np.vstack([b[3] for b in batch]), device=self.device)

        current_q = self.model(states)
        with torch.no_grad():
            next_q = self.model(next_states)

        target_q = current_q.detach().clone()
        for i, (_, action_indices, reward, _) in enumerate(batch):
            target_value = reward + self.gamma * torch.max(next_q[i]).item()
            for action_index in action_indices:
                target_q[i, action_index] = target_value

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


@dataclass
class PsoParticleState:
    current_weights: List[np.ndarray]
    pbest_weights: List[np.ndarray]
    pbest_score: float
    velocities: List[np.ndarray]


class PsoAggregator:
    def __init__(self, args) -> None:
        self.inertia = args.pso_inertia
        self.local_acc = args.pso_local_acc
        self.global_acc = args.pso_global_acc
        self.particles: Dict[int, PsoParticleState] = {}
        self.rng = random.Random(args.sample_seed)

    def _init_velocities(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        velocities = []
        for layer in weights:
            velocities.append(np.random.rand(*layer.shape) / 5.0 - 0.10)
        return velocities

    def run_pso(
        self,
        client_ids: List[int],
        initial_weights: List[List[np.ndarray]],
        eval_fn,
        iterations: int,
    ) -> Optional[List[np.ndarray]]:
        if not client_ids:
            return None

        losses = [eval_fn(weights) for weights in initial_weights]
        gbest_score = float("inf")
        gbest_weights = None

        for client_id, weights, loss in zip(client_ids, initial_weights, losses):
            if client_id not in self.particles:
                self.particles[client_id] = PsoParticleState(
                    current_weights=weights,
                    pbest_weights=weights,
                    pbest_score=loss,
                    velocities=self._init_velocities(weights),
                )
            else:
                particle = self.particles[client_id]
                particle.current_weights = weights
                if loss < particle.pbest_score:
                    particle.pbest_score = loss
                    particle.pbest_weights = weights

            if loss < gbest_score:
                gbest_score = loss
                gbest_weights = weights

        for _ in range(iterations):
            for client_id in client_ids:
                particle = self.particles[client_id]
                new_weights = []
                local_rand = self.rng.random()
                global_rand = self.rng.random()
                for layer_idx, layer_weights in enumerate(particle.current_weights):
                    velocity = (
                        self.inertia * particle.velocities[layer_idx]
                        + self.local_acc
                        * local_rand
                        * (particle.pbest_weights[layer_idx] - layer_weights)
                        + self.global_acc
                        * global_rand
                        * (gbest_weights[layer_idx] - layer_weights)
                    )
                    particle.velocities[layer_idx] = velocity
                    new_weights.append(layer_weights + velocity)
                particle.current_weights = new_weights

            current_losses = []
            for client_id in client_ids:
                particle = self.particles[client_id]
                loss = eval_fn(particle.current_weights)
                current_losses.append(loss)
                if loss < particle.pbest_score:
                    particle.pbest_score = loss
                    particle.pbest_weights = particle.current_weights
                if loss < gbest_score:
                    gbest_score = loss
                    gbest_weights = particle.current_weights

        return gbest_weights
