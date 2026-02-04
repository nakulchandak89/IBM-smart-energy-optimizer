
import pandas as pd
import numpy as np
import random
from rtp_model import RTPGenerator
from utils import APPLIANCE_MAPPING, ApplianceCategory

COMFORT_PENALTY = 50.0 # High penalty for annoyed users

class SmartHomeEnv:
    """
    Simulated Environment for Smart Appliance Scheduling with User Constraints.
    """
    
    def __init__(self, data_path, rtp_generator):
        self.df = pd.read_csv(data_path)
        self.rtp_generator = rtp_generator
        self.current_idx = 0
        
        # Filter for known appliances
        known = list(APPLIANCE_MAPPING.keys())
        self.df = self.df[self.df['Appliance Type'].isin(known)].reset_index(drop=True)
        self.data_len = len(self.df)
        print(f"Environment initialized with {self.data_len} records.")

    def reset(self):
        """
        Start new episode.
        Simulates user constraints (Flexibility, Preferred Window).
        """
        self.current_idx = random.randint(0, self.data_len - 1)
        record = self.df.iloc[self.current_idx]
        app_name = record['Appliance Type']
        
        # Generate RTP
        date_str = record['Date']
        rtp_profile = self.rtp_generator.get_prices(date_str)
        
        # User Constraint Simulation
        category = APPLIANCE_MAPPING.get(app_name)
        
        if category == ApplianceCategory.NON_ELASTIC:
            # Non-elastic: Must run NOW. 
            # In simulation, let's say "Expected Time" is the constraint.
            hour = int(record['Time'].split(':')[0])
            slot = hour // 4
            is_flexible = False
            start_slot = slot
            end_slot = slot # Strict one-slot window
            
        else:
            # Elastic: Flexible but might have preferences
            is_flexible = True
            
            # Simulate a preferred window (e.g., "I want this done between 8am and 8pm")
            # Randomly pick a window of length 2-6 slots
            start_slot = random.randint(0, 4)
            duration = random.randint(1, 6 - start_slot)
            end_slot = start_slot + duration - 1
            
            # 20% chance user says "ANYTIME" (0-5)
            if random.random() < 0.2:
                start_slot = 0
                end_slot = 5
        
        # Construct State Dict
        state = {
            'appliance': app_name,
            'energy': record['Energy Consumption (kWh)'],
            'temp': record['Outdoor Temperature (°C)'],
            'size': record['Household Size'],
            'rtp': rtp_profile,
            'is_flexible': is_flexible,
            'start_slot': start_slot,
            'end_slot': end_slot
        }
        
        return state, record

    def step(self, action, state_details):
        """
        Execute action and calculate reward + penalty.
        """
        price = state_details['rtp'][action]
        consumption = state_details['energy']
        
        # 1. Base Cost
        cost = consumption * price
        
        # 2. Comfort Penalty
        penalty = 0.0
        start = state_details['start_slot']
        end = state_details['end_slot']
        
        # If action is outside the [start, end] window
        if action < start or action > end:
            penalty = COMFORT_PENALTY
            
        reward = -(cost + penalty)
        
        return reward, True
