
import os
import sys
from agent import QLearningAgent
from rtp_model import RTPGenerator
from utils import StateUtils, APPLIANCE_MAPPING, ApplianceCategory

MODEL_PATH = "q_table.pkl"

def get_slot_time_range(slot_idx):
    start_hour = slot_idx * 4
    end_hour = start_hour + 3
    return f"{start_hour:02d}:00 - {end_hour:02d}:59"

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run train.py first.")
        return

    # Load Agent
    agent = QLearningAgent()
    agent.load_model(MODEL_PATH)
    
    rtp_gen = RTPGenerator()
    
    print("\n--- Smart Appliance Scheduling Recommendation ---")
    
    # User Inputs
    # For demo purposes, we can hardcode or ask for input. 
    # Let's use prompts.
    try:
        print("\nAvailable Appliances: " + ", ".join(APPLIANCE_MAPPING.keys()))
        app_name = input("Enter Appliance Name: ").strip()
        if app_name not in APPLIANCE_MAPPING:
            print("Unknown appliance. Defaulting to Washing Machine.")
            app_name = "Washing Machine"
            
        category = StateUtils.get_appliance_category(app_name)
        if category == ApplianceCategory.NON_ELASTIC:
            print(f"Note: {app_name} is Non-Elastic. Recommendation is to run it when needed.")
            # We can still check cost but optimization is limited.
            
        date_str = input("Enter Date (YYYY-MM-DD): ").strip()
        if len(date_str) != 10: 
            print("Invalid date. Using 2023-10-01")
            date_str = "2023-10-01"
            
        energy = float(input("Enter Energy Consumption (kWh) [e.g., 1.5]: ") or 1.5)
        temp = float(input("Enter Outdoor Temp (°C) [e.g., 20]: ") or 20.0)
        
        # Get Context
        rtp_profile = rtp_gen.get_prices(date_str)
        print(f"\nRTP Prices for {date_str}: {rtp_profile}")
        
        # Get Recommendation
        state_key = StateUtils.discretize_state(app_name, energy, temp, 4, rtp_profile)
        action = agent.choose_action(state_key, force_greedy=True)
        
        # Analysis
        rec_slot_str = get_slot_time_range(action)
        rec_price = rtp_profile[action]
        est_cost = energy * rec_price
        
        # Compare with worst case
        worst_slot = rtp_profile.index(max(rtp_profile))
        worst_cost = energy * rtp_profile[worst_slot]
        savings = worst_cost - est_cost
        
        print(f"\n>>> Recommendation: Run {app_name} between {rec_slot_str}")
        print(f"Estimated Cost: ₹{est_cost:.2f}")
        print(f"Potential Savings vs Peak Time: ₹{savings:.2f}")
        
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
