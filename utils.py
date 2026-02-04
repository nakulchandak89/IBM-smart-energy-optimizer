
import enum

class ApplianceCategory(enum.Enum):
    ELASTIC = "Elastic"
    NON_ELASTIC = "Non-Elastic"

# Classification based on problem description
APPLIANCE_MAPPING = {
    "Washing Machine": ApplianceCategory.ELASTIC,
    "Dishwasher": ApplianceCategory.ELASTIC,
    "Microwave": ApplianceCategory.ELASTIC,
    "Oven": ApplianceCategory.ELASTIC,
    "Lights": ApplianceCategory.NON_ELASTIC,
    "Fridge": ApplianceCategory.NON_ELASTIC,
    "Air Conditioning": ApplianceCategory.NON_ELASTIC,
    "Heater": ApplianceCategory.NON_ELASTIC,
    "TV": ApplianceCategory.NON_ELASTIC,
    "Computer": ApplianceCategory.NON_ELASTIC
}

class StateUtils:
    """Helper to handle state representation and discretization."""
    
    @staticmethod
    def get_appliance_category(appliance_name):
        return APPLIANCE_MAPPING.get(appliance_name, ApplianceCategory.NON_ELASTIC)

    @staticmethod
    def calculate_kwh(watts, hours):
        """Convert Power (W) and Duration (h) to Energy (kWh)."""
        return (watts * hours) / 1000.0

    @staticmethod
    def time_to_slot(hour):
        """Convert 24h hour (0-23) to Slot (0-5)."""
        return int(hour // 4)

    @staticmethod
    def discretize_state(appliance_name, energy, temp, size, rtp_profile, 
                         is_flexible=True, start_slot=0, end_slot=5):
        """
        Convert continuous state into a discrete tuple for Q-Table.
        Structure: [AppID, EnergyBin, TempBin, PriceBin, FlexFlag, StartSlot, EndSlot]
        Using bins to keep state space manageable for tabular RL.
        """
        # 1. Appliance ID
        app_id = appliance_name_to_int(appliance_name)
        
        # 2. Energy Bin (0: <0.5, 1: <1.5, 2: <3.0, 3: >=3.0)
        if energy < 0.5: e_bin = 0
        elif energy < 1.5: e_bin = 1
        elif energy < 3.0: e_bin = 2
        else: e_bin = 3
        
        # 3. Temp Bin (0: <0, 1: <15, 2: <25, 3: <35, 4: >=35)
        if temp < 0: t_bin = 0
        elif temp < 15: t_bin = 1
        elif temp < 25: t_bin = 2
        elif temp < 35: t_bin = 3
        else: t_bin = 4
        
        # 4. Price Pattern Bin (Average price level)
        # Using 3 bins: Low, Medium, High relative to typical
        avg_price = sum(rtp_profile) / len(rtp_profile)
        if avg_price < 8: p_bin = 0
        elif avg_price < 15: p_bin = 1
        else: p_bin = 2
        
        # 5. Flexibility
        flex_flag = 1 if is_flexible else 0
        
        # 6. Constraints
        # Just use the raw slot indices (0-5)
        s_slot = start_slot
        e_slot = end_slot
        
        # Return tuple key
        return (app_id, e_bin, t_bin, p_bin, flex_flag, s_slot, e_slot)

def appliance_name_to_int(name):
    """Deterministic map string to int."""
    top_appliances = list(APPLIANCE_MAPPING.keys())
    try:
        return top_appliances.index(name)
    except ValueError:
        return 99

