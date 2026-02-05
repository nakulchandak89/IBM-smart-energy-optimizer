
import numpy as np
import hashlib

class RTPGenerator:
    """
    Generates deterministic Real-Time Pricing (RTP) curves based on date.
    Divides day into 6 fixed 4-hour slots.
    """
    
    NUM_SLOTS = 6
    MIN_PRICE = 0.5
    MAX_PRICE = 25.0
    
    @staticmethod
    def get_prices(date_str):
        """
        Generate 6 price points for the given date string (YYYY-MM-DD).
        Deterministic: Same date -> Same prices.
        """
        # Create a deterministic seed from the date string
        # We use MD5 hash of date to get a large integer seed
        hash_obj = hashlib.md5(date_str.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        np.random.seed(seed)
        
        # Base trend: Start low, peak in evening (Slots 4-5), drop at night
        # This is a common pattern, but we add noise.
        base_trend = np.array([2.0, 4.0, 10.0, 15.0, 20.0, 8.0])
        
        # Add random variation (-5 to +5)
        noise = np.random.uniform(-5, 5, size=RTPGenerator.NUM_SLOTS)
        
        # Occasionally inject a random spike (Super Peak)
        if np.random.random() < 0.2: # 20% chance of a spike day
            spike_slot = np.random.randint(0, RTPGenerator.NUM_SLOTS)
            noise[spike_slot] += np.random.uniform(5, 10)
            
        prices = base_trend + noise
        
        # Clip to valid range
        prices = np.clip(prices, RTPGenerator.MIN_PRICE, RTPGenerator.MAX_PRICE)
        
        return np.round(prices, 2).tolist()

if __name__ == "__main__":
    # Test
    print("Test Prices for 2023-12-01:")
    print(RTPGenerator.get_prices("2023-12-01"))
    print("Test Prices for 2023-12-01 (Repeat should match):")
    print(RTPGenerator.get_prices("2023-12-01"))
    print("Test Prices for 2023-12-02:")
    print(RTPGenerator.get_prices("2023-12-02"))
