import os
from dotenv import load_dotenv
from zyndai_agent.agent import AgentConfig, ZyndAIAgent
from core_engine import get_workflow

load_dotenv()

import time



# --- 1. DEFINE A SAFE MOCK AGENT (Bypasses Network Crash) ---
class SafeZyndAgent:
    """
    A simulation wrapper that mimics the Zynd Agent structure
    but skips the live MQTT registry check to prevent 401 Errors.
    """
    def __init__(self, agent_config=None):
        # We simulate a successful login
        self.identity = "did:zynd:simulation_mode_active"
        print(f"✅ [SafeZyndAgent] Initialized in SIMULATION MODE.")
        print(f"✅ [SafeZyndAgent] Skipped MQTT Registry (401 Avoidance).")

# --- 2. CONFIG CLASS ---
class AgentConfig:
    def __init__(self):
        self.secret_seed = os.getenv("ZYND_WALLET_SEED")
        self.mqtt_broker_url = "mqtt://registry.zynd.ai:1883"
        # Point to the file we created, even if we don't send it to the network
        self.identity_credential_path = "./identity_credential.json"

# --- 3. MAIN WRAPPER CLASS ---
class EmergencyResponseAgent:
    def __init__(self):
        # Initialize the LangGraph workflow (Your Real Logic)
        self.workflow = get_workflow()
        
        # Initialize Config
        self.agent_config = AgentConfig()
        
        # --- CRITICAL FIX: TRY REAL AGENT, FALLBACK TO SAFE AGENT ---
        try:
            # Try to import the real library
            from zyndai_agent.agent import ZyndAIAgent
            print("🔄 Attempting to connect to Zynd Network...")
            self.zynd_agent = ZyndAIAgent(agent_config=self.agent_config)
            self.agent_name = "Kolkata_Emergency_Coordinator"
            print("✅ Connected to Real Zynd Network!")
            
        except Exception as e:
            # If 401 or Import Error happens, use Safe Mode
            print(f"⚠️ Zynd Network Error: {e}")
            print("⚠️ SWITCHING TO OFFLINE SIMULATION MODE for Demo.")
            self.zynd_agent = SafeZyndAgent(agent_config=self.agent_config)
            self.agent_name = "Kolkata_Emergency_Coordinator (Offline)"

    def process_emergency(self, location: tuple, accident_type: str):
        """
        Bridges the external request to internal LangGraph
        """
        inputs = {
            "accident_location": location,
            "accident_type": accident_type,
            "assigned_ambulance": "Waiting",
            "ambulance_eta": 0.0,
            "selected_hospital_id": "Waiting",
            "selected_hospital_name": "Waiting",
            "final_plan": "Waiting"
        }
        
        # Run the real LangGraph logic
        result = self.workflow.invoke(inputs)
        
        # Return formatted results
        return {
            "status": "success",
            "zynd_agent_id": getattr(self.zynd_agent, 'identity', 'Unknown'),
            "ambulance": result.get("assigned_ambulance"),
            "eta": result.get("ambulance_eta"),
            "hospital": result.get("selected_hospital_name"),
            "medical_plan": result.get("final_plan")
        }

# Singleton instance used by app.py
emergency_system = EmergencyResponseAgent()