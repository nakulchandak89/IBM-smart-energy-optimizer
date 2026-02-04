"""
Deploy Q-Learning Model to IBM Watson Machine Learning
This script tests the connection to your existing IBM deployment.
"""

import pickle
import os
from ibm_watson_machine_learning import APIClient

# --- IBM CLOUD CONFIGURATION ---
API_KEY = "WVyuJ5C1vjhx-CwlGEQeZh6USwDJzajGph9_jlQJvwth"
SPACE_ID = "bea9575a-e95e-426b-9689-13f3051bcfa5"
DEPLOYMENT_ID = "4549cb28-e6fa-430b-85df-1f48dbae01af"
WML_URL = "https://eu-gb.ml.cloud.ibm.com"

WML_CREDENTIALS = {
    "url": WML_URL,
    "apikey": API_KEY
}


def test_ibm_connection():
    """Test connection to IBM Watson ML."""
    print("=" * 60)
    print("🔌 IBM Watson ML Connection Test")
    print("=" * 60)
    
    print(f"\n📡 Endpoint: {WML_URL}")
    print(f"📦 Space ID: {SPACE_ID}")
    print(f"🚀 Deployment ID: {DEPLOYMENT_ID}")
    
    try:
        client = APIClient(WML_CREDENTIALS)
        client.set.default_space(SPACE_ID)
        print("\n✅ Connected to IBM Watson ML successfully!")
        return client
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return None


def test_prediction(client):
    """Test making a prediction."""
    print("\n" + "=" * 60)
    print("🧪 Testing Model Prediction")
    print("=" * 60)
    
    # Sample state: (app_id, energy_bin, temp_bin, price_bin, flex_flag, start_slot, end_slot)
    test_state = [1, 2, 1, 2, 1, 0, 5]
    
    payload = {
        "input_data": [{
            "fields": ["app_id", "energy_bin", "temp_bin", "price_bin", "flex_flag", "start_slot", "end_slot"],
            "values": [test_state]
        }]
    }
    
    print(f"\n📤 Sending test state: {test_state}")
    
    try:
        response = client.deployments.score(DEPLOYMENT_ID, payload)
        predictions = response.get('predictions', [{}])[0].get('values', [[0]])
        result = predictions[0]
        
        if isinstance(result, list):
            result = result[0]
        
        slot_labels = ["🌙 Night (0-4h)", "🌅 Early (4-8h)", "☀️ Morning (8-12h)", 
                       "🌤️ Afternoon (12-16h)", "🌇 Evening (16-20h)", "🌃 Late (20-24h)"]
        
        print(f"\n✅ IBM Recommendation: Slot {result}")
        print(f"   {slot_labels[int(result)]}")
        return int(result)
        
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        return None


def list_deployments(client):
    """List all active deployments."""
    print("\n" + "=" * 60)
    print("📋 Active Deployments")
    print("=" * 60)
    
    try:
        deployments = client.deployments.list()
        print(deployments)
    except Exception as e:
        print(f"Error listing deployments: {e}")


if __name__ == "__main__":
    # Test connection
    client = test_ibm_connection()
    
    if client:
        # List deployments
        list_deployments(client)
        
        # Test prediction
        test_prediction(client)
    
    print("\n" + "=" * 60)
    print("✅ IBM Integration Test Complete!")
    print("=" * 60)
