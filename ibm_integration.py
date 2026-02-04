"""
IBM Watson Machine Learning Integration Module
Connects to IBM Cloud for AI-powered appliance scheduling recommendations.
"""

import os
from typing import Tuple, Optional

# --- IBM CLOUD CONFIGURATION ---
# These can be overridden via environment variables for security
API_KEY = os.environ.get("IBM_API_KEY", "WVyuJ5C1vjhx-CwlGEQeZh6USwDJzajGph9_jlQJvwth")
SPACE_ID = os.environ.get("IBM_SPACE_ID", "bea9575a-e95e-426b-9689-13f3051bcfa5")
DEPLOYMENT_ID = os.environ.get("IBM_DEPLOYMENT_ID", "4549cb28-e6fa-430b-85df-1f48dbae01af")
WML_URL = os.environ.get("IBM_WML_URL", "https://eu-gb.ml.cloud.ibm.com")

WML_CREDENTIALS = {
    "url": WML_URL,
    "apikey": API_KEY
}

# Lazy-loaded client to avoid import errors if ibm_watson not installed
_client = None
_client_initialized = False


def _get_client():
    """Lazy initialization of IBM Watson client."""
    global _client, _client_initialized
    
    if _client_initialized:
        return _client
    
    try:
        from ibm_watson_machine_learning import APIClient
        _client = APIClient(WML_CREDENTIALS)
        _client.set.default_space(SPACE_ID)
        _client_initialized = True
        print("✅ IBM Watson ML client initialized successfully.")
        return _client
    except ImportError:
        print("⚠️ ibm_watson_machine_learning not installed. Run: pip install ibm-watson-machine-learning")
        _client_initialized = True
        return None
    except Exception as e:
        print(f"⚠️ IBM Watson initialization failed: {e}")
        _client_initialized = True
        return None


def is_ibm_available() -> bool:
    """Check if IBM Watson ML is available and configured."""
    client = _get_client()
    return client is not None


def get_ibm_recommendation(state_tuple: Tuple) -> Optional[int]:
    """
    Sends state to IBM Watson ML and retrieves optimal time slot.
    
    Args:
        state_tuple: (app_id, energy_bin, temp_bin, price_bin, flex_flag, start_slot, end_slot)
    
    Returns:
        Recommended slot (0-5), or None if IBM unavailable.
    """
    client = _get_client()
    
    if client is None:
        return None
    
    try:
        # Format payload for IBM Watson
        payload = {
            "input_data": [{
                "fields": ["app_id", "energy_bin", "temp_bin", "price_bin", "flex_flag", "start_slot", "end_slot"],
                "values": [list(state_tuple)]
            }]
        }
        
        # Send request to IBM Cloud
        response = client.deployments.score(DEPLOYMENT_ID, payload)
        
        # Extract prediction
        predictions = response.get('predictions', [{}])[0].get('values', [[0]])
        recommended_action = predictions[0]
        
        # Handle both single value and list responses
        if isinstance(recommended_action, list):
            recommended_action = recommended_action[0]
        
        return int(recommended_action)
        
    except Exception as e:
        print(f"⚠️ IBM recommendation failed: {e}")
        return None


def get_recommendation_with_confidence(state_tuple: Tuple) -> Tuple[Optional[int], Optional[float]]:
    """
    Enhanced version that also returns confidence score if available.
    
    Returns:
        (recommended_slot, confidence) or (None, None) if unavailable.
    """
    client = _get_client()
    
    if client is None:
        return None, None
    
    try:
        payload = {
            "input_data": [{
                "fields": ["app_id", "energy_bin", "temp_bin", "price_bin", "flex_flag", "start_slot", "end_slot"],
                "values": [list(state_tuple)]
            }]
        }
        
        response = client.deployments.score(DEPLOYMENT_ID, payload)
        
        predictions = response.get('predictions', [{}])[0]
        values = predictions.get('values', [[0, 0.0]])
        
        # Try to extract action and confidence
        if len(values[0]) >= 2:
            action = int(values[0][0])
            confidence = float(values[0][1])
        else:
            action = int(values[0][0]) if values[0] else 0
            confidence = None
        
        return action, confidence
        
    except Exception as e:
        print(f"⚠️ IBM recommendation failed: {e}")
        return None, None


# --- Test block ---
if __name__ == "__main__":
    print("=" * 50)
    print("IBM Watson ML Connection Test")
    print("=" * 50)
    
    print(f"\n📡 Endpoint: {WML_URL}")
    print(f"🔑 API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"📦 Space ID: {SPACE_ID}")
    print(f"🚀 Deployment ID: {DEPLOYMENT_ID}")
    
    print("\n🔄 Testing connection...")
    
    if is_ibm_available():
        test_state = (1, 2, 1, 2, 1, 0, 5)
        print(f"\n📤 Sending test state: {test_state}")
        
        result = get_ibm_recommendation(test_state)
        
        if result is not None:
            print(f"✅ IBM Recommendation: Run appliance in Slot {result}")
            
            # Test with confidence
            result_conf, conf = get_recommendation_with_confidence(test_state)
            if conf is not None:
                print(f"   Confidence: {conf:.2%}")
        else:
            print("❌ Failed to get recommendation from IBM.")
    else:
        print("❌ IBM Watson ML is not available.")
    
    print("\n" + "=" * 50)
