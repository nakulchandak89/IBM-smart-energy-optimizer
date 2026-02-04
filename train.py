
import os
import numpy as np
import matplotlib.pyplot as plt
from environment import SmartHomeEnv
from agent import QLearningAgent
from rtp_model import RTPGenerator
from utils import StateUtils

DATA_PATH = "data_with_rtp.csv"
MODEL_PATH = "q_table.pkl"
EPISODES = 10000

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Initialize
    rtp_gen = RTPGenerator()
    env = SmartHomeEnv(DATA_PATH, rtp_gen)
    agent = QLearningAgent()
    
    rewards = []
    baseline_costs = []
    optimized_costs = []
    
    print("Starting training...")
    
    for episode in range(EPISODES):
        state_dict, record = env.reset()
        
        # Discretize state for Agent
        state_key = StateUtils.discretize_state(
            state_dict['appliance'], 
            state_dict['energy'], 
            state_dict['temp'], 
            state_dict['size'], 
            state_dict['rtp'],
            is_flexible=state_dict['is_flexible'],
            start_slot=state_dict['start_slot'],
            end_slot=state_dict['end_slot']
        )
        
        # Choose Action
        action = agent.choose_action(state_key)
        
        # Take Step
        reward, done = env.step(action, state_dict)
        
        # Learn
        agent.learn(state_key, action, reward)
        
        # Metrics
        opt_cost = -reward
        
        # Calculate Baseline Cost (what user originally paid)
        # We need to find which slot the original usage fell into.
        # Original Time is in record['Time'] (HH:MM).
        hour = int(record['Time'].split(':')[0])
        original_slot = hour // 4
        baseline_price = state_dict['rtp'][original_slot]
        base_cost = state_dict['energy'] * baseline_price
        
        rewards.append(reward)
        optimized_costs.append(opt_cost)
        baseline_costs.append(base_cost)
        
        # Decay Epsilon
        if episode % 100 == 0:
            agent.decay_epsilon()
            
        if episode % 1000 == 0:
            avg_rew = np.mean(rewards[-1000:])
            print(f"Episode {episode}/{EPISODES} | Avg Reward: {avg_rew:.2f} | Epsilon: {agent.epsilon:.2f}")

    # Save Model
    agent.save_model(MODEL_PATH)
    
    # Save Training History for Streamlit
    np.savez("training_history.npz", 
             rewards=rewards, 
             baseline_costs=baseline_costs, 
             optimized_costs=optimized_costs)
    print("Training history saved to training_history.npz")
    
    # Analysis
    avg_base = np.mean(baseline_costs)
    avg_opt = np.mean(optimized_costs)
    savings = ((avg_base - avg_opt) / avg_base) * 100
    
    print("\nTraining Complete.")
    print(f"Average Baseline Cost per usage: {avg_base:.2f}")
    print(f"Average Optimized Cost per usage: {avg_opt:.2f}")
    print(f"Total Savings: {savings:.2f}%")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(np.convolve(optimized_costs, np.ones(100)/100, mode='valid'), label='Optimized Cost (Moving Avg)')
    plt.plot(np.convolve(baseline_costs, np.ones(100)/100, mode='valid'), label='Baseline Cost (Moving Avg)', alpha=0.5)
    plt.legend()
    plt.title('Training Progress: Cost Minimization')
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.savefig('training_results.png')
    print("Training plot saved to training_results.png")

if __name__ == "__main__":
    train()
