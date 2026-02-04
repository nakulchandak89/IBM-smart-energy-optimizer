"""
Enhanced Q-Learning Agent with improved algorithms.
Features:
- Double Q-Learning to reduce overestimation bias
- Experience Replay for better learning stability
- Adaptive learning rate decay
- Reward shaping for optimal slot selection
"""

import numpy as np
import random
import pickle
from collections import deque
from typing import Tuple, Optional, List


class QLearningAgent:
    """
    Enhanced Tabular Q-Learning Agent.
    Learns to map discretized states to optimal time slots.
    """
    
    def __init__(
        self, 
        action_space_size: int = 6, 
        learning_rate: float = 0.15,
        discount_factor: float = 0.95, 
        epsilon: float = 1.0, 
        epsilon_decay: float = 0.998, 
        min_epsilon: float = 0.01,
        lr_decay: float = 0.9999,
        min_lr: float = 0.01,
        use_double_q: bool = True,
        replay_buffer_size: int = 10000,
        batch_size: int = 32
    ):
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.use_double_q = use_double_q
        self.batch_size = batch_size
        
        # Primary Q-Table
        self.q_table = {}
        
        # Secondary Q-Table for Double Q-Learning
        self.q_table_2 = {} if use_double_q else None
        
        # Experience Replay Buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training statistics
        self.training_steps = 0
        self.episodes_trained = 0

    def get_q_values(self, state: Tuple, table: int = 1) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        q_table = self.q_table if table == 1 else self.q_table_2
        
        if state not in q_table:
            # Smart initialization: slight preference for off-peak slots
            initial_values = np.array([0.5, 0.3, 0.1, 0.0, -0.1, 0.2])
            q_table[state] = initial_values
        
        return q_table[state]

    def choose_action(self, state: Tuple, force_greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection with enhanced exploration.
        """
        if not force_greedy and random.random() < self.epsilon:
            # Weighted exploration: prefer less-tried slots
            q_vals = self.get_q_values(state)
            try_counts = np.maximum(1, np.abs(q_vals) + 1)
            probs = 1 / try_counts
            probs = probs / probs.sum()
            return np.random.choice(self.action_space_size, p=probs)
        
        # Exploit: Use both Q-tables for Double Q-Learning
        if self.use_double_q and self.q_table_2:
            q1 = self.get_q_values(state, table=1)
            q2 = self.get_q_values(state, table=2)
            combined_q = (q1 + q2) / 2
        else:
            combined_q = self.get_q_values(state)
        
        # Random tie-breaking
        max_q = np.max(combined_q)
        best_actions = np.where(combined_q == max_q)[0]
        return np.random.choice(best_actions)

    def store_experience(self, state: Tuple, action: int, reward: float, next_state: Optional[Tuple] = None, done: bool = True):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, state: Tuple, action: int, reward: float, next_state: Optional[Tuple] = None):
        """
        Update Q-value using enhanced Bellman equation.
        Supports Double Q-Learning to reduce overestimation.
        """
        self.training_steps += 1
        
        # Store for experience replay
        self.store_experience(state, action, reward, next_state, done=True)
        
        if self.use_double_q and self.q_table_2 is not None:
            # Double Q-Learning: randomly update one table using other for evaluation
            if random.random() < 0.5:
                self._update_q_table(self.q_table, self.q_table_2, state, action, reward, next_state)
            else:
                self._update_q_table(self.q_table_2, self.q_table, state, action, reward, next_state)
        else:
            # Standard Q-Learning
            current_q = self.get_q_values(state)
            target = reward
            if next_state is not None:
                target += self.gamma * np.max(self.get_q_values(next_state))
            
            current_q[action] += self.lr * (target - current_q[action])
        
        # Decay learning rate
        self.lr = max(self.min_lr, self.lr * self.lr_decay)

    def _update_q_table(self, table_to_update: dict, other_table: dict, 
                        state: Tuple, action: int, reward: float, next_state: Optional[Tuple]):
        """Helper for Double Q-Learning update."""
        if state not in table_to_update:
            table_to_update[state] = np.zeros(self.action_space_size)
        
        current_q = table_to_update[state][action]
        
        if next_state is not None:
            # Use current table to select action, other table to evaluate
            if next_state not in table_to_update:
                table_to_update[next_state] = np.zeros(self.action_space_size)
            best_action = np.argmax(table_to_update[next_state])
            
            if next_state not in other_table:
                other_table[next_state] = np.zeros(self.action_space_size)
            target = reward + self.gamma * other_table[next_state][best_action]
        else:
            target = reward
        
        table_to_update[state][action] += self.lr * (target - current_q)

    def replay_learn(self, batch_size: Optional[int] = None):
        """Learn from random batch of stored experiences."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.learn(state, action, reward, next_state if not done else None)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def get_best_slot_for_price(self, rtp_profile: List[float], is_flexible: bool, start_slot: int = 0, end_slot: int = 5) -> int:
        """
        Direct price-based recommendation (fallback or verification).
        Returns the cheapest slot within constraints.
        """
        if not is_flexible:
            return start_slot
        
        valid_slots = range(start_slot, end_slot + 1)
        prices = [(slot, rtp_profile[slot]) for slot in valid_slots]
        return min(prices, key=lambda x: x[1])[0]

    def save_model(self, path: str):
        """Save both Q-tables and training state."""
        save_data = {
            'q_table': self.q_table,
            'q_table_2': self.q_table_2,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✅ Model saved to {path} ({len(self.q_table)} states learned)")

    def load_model(self, path: str):
        """Load Q-tables and training state."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new format
        if isinstance(data, dict) and 'q_table' in data:
            self.q_table = data['q_table']
            self.q_table_2 = data.get('q_table_2', {})
            self.epsilon = data.get('epsilon', self.min_epsilon)
            self.lr = data.get('lr', self.min_lr)
            self.training_steps = data.get('training_steps', 0)
            self.episodes_trained = data.get('episodes_trained', 0)
        else:
            # Old format: just Q-table dict
            self.q_table = data
            self.q_table_2 = {}
        
        print(f"✅ Model loaded from {path} ({len(self.q_table)} states)")

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            'states_learned': len(self.q_table),
            'training_steps': self.training_steps,
            'episodes': self.episodes_trained,
            'epsilon': self.epsilon,
            'learning_rate': self.lr,
            'replay_buffer_size': len(self.replay_buffer)
        }


# Backward compatibility alias
Agent = QLearningAgent
