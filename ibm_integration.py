"""
IBM Watson Machine Learning Integration Module (Lightweight REST API Version)
Connects to IBM Cloud for AI-powered appliance scheduling recommendations.
Uses direct HTTP requests to avoid heavy SDK dependencies.
"""

import os
import requests
import json
from typing import Tuple, Optional

# --- IBM CLOUD CONFIGURATION ---
API_KEY = os.environ.get("IBM_API_KEY", "WVyuJ5C1vjhx-CwlGEQeZh6USwDJzajGph9_jlQJvwth")
SPACE_ID = os.environ.get("IBM_SPACE_ID", "bea9575a-e95e-426b-9689-13f3051bcfa5")
DEPLOYMENT_ID = os.environ.get("IBM_DEPLOYMENT_ID", "4549cb28-e6fa-430b-85df-1f48dbae01af")
WML_URL = os.environ.get("IBM_WML_URL", "https://eu-gb.ml.cloud.ibm.com")

# Token management
_access_token = None


def _get_access_token():
    """Retrieves IAM access token using API Key."""
    global _access_token
    
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}"
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        _access_token = token_data.get("access_token")
        return _access_token
    except Exception as e:
        print(f"⚠️ Failed to get IBM access token: {e}")
        return None


def is_ibm_available() -> bool:
    """Check if IBM Watson ML is reachable."""
    token = _get_access_token()
    return token is not None


def get_ibm_recommendation(state_tuple: Tuple) -> Optional[int]:
    """
    Sends state to IBM Watson ML via REST API.
    """
    token = _get_access_token()
    if not token:
        return None
    
    scoring_url = f"{WML_URL}/ml/v4/deployments/{DEPLOYMENT_ID}/predictions?version=2021-10-01"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input_data": [{
            "fields": ["app_id", "energy_bin", "temp_bin", "price_bin", "flex_flag", "start_slot", "end_slot"],
            "values": [list(state_tuple)]
        }]
    }
    
    try:
        response = requests.post(scoring_url, json=payload, headers=headers)
        response.raise_for_status()
        
        result_json = response.json()
        predictions = result_json.get('predictions', [{}])[0].get('values', [[0]])
        recommended_action = predictions[0]
        
        if isinstance(recommended_action, list):
            recommended_action = recommended_action[0]
            
        return int(recommended_action)
        
    except Exception as e:
        print(f"⚠️ IBM recommendation failed: {e}")
        # If token expired, clear it (simple retry logic could be added)
        global _access_token
        _access_token = None
        return None


# --- Test block ---
if __name__ == "__main__":
    print("=" * 50)
    print("IBM Watson ML REST API Test")
    print("=" * 50)
    
    if is_ibm_available():
        test_state = (1, 2, 1, 2, 1, 0, 5)
        print(f"\n📤 Sending test state: {test_state}")
        
        result = get_ibm_recommendation(test_state)
        
        if result is not None:
            print(f"✅ IBM Recommendation: Run appliance in Slot {result}")
        else:
            print("❌ Failed to get recommendation from IBM.")
    else:
        print("❌ IBM Watson ML is not available (check credentials).")
