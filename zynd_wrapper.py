import os
from dotenv import load_dotenv
from core_engine import get_workflow

load_dotenv()

import time


# --- 1. SIMULATION AGENT (No ZyndAI Dependency) ---
class SimulationAgent:
    """
    Emergency response agent that works without ZyndAI blockchain.
    For production, integrate with actual ZyndAI or other blockchain/identity system.
    """
    def __init__(self):
        self.identity = f"did:zynd:emergency:kolkata:{os.getenv('AGENT_ID', 'demo-001')}"
        self.credentials = ["Credential:ALS_Certified", "Credential:Authorized_Responder"]
        print(f"✅ [EmergencyAgent] Initialized in OFFLINE MODE")
        print(f"✅ [Identity] {self.identity}")

    def verify_handshake(self, target_did: str, required_credential: str) -> bool:
        """
        Simulates the cryptographic handshake and credential verification.
        In a real system, this would verify the VC signature against the ledger.
        """
        print(f"🔐 [Zynd Security] Initiating Handshake with {target_did}...")
        time.sleep(0.5) # Simulate network latency
        
        # Simulation Logic: Check if DID is valid and credential exists
        if not target_did.startswith("did:zynd:"):
            print(f"❌ [Zynd Security] Handshake FAILED: Invalid DID format ({target_did})")
            return False
            
        if required_credential in self.credentials:
            print(f"✅ [Zynd Security] Handshake SUCCESS: Verified {required_credential}")
            return True
        else:
            print(f"❌ [Zynd Security] Handshake FAILED: Missing Credential {required_credential}")
            return False

# --- 2. MAIN WRAPPER CLASS ---
class EmergencyResponseAgent:
    def __init__(self):
        print("--- Initializing Emergency Response System ---")
        
        # Initialize the LangGraph workflow (Your Real Logic)
        self.workflow = get_workflow()
        
        # Use simulation agent (no blockchain dependency)
        self.zynd_agent = SimulationAgent()
        self.agent_name = "Kolkata_Emergency_Coordinator"
        
        print(f"✅ System Ready: {self.agent_name}")
        print(f"✅ Agent ID: {self.zynd_agent.identity}")

    def process_emergency(self, location: tuple, accident_type: str, severity: str = "HIGH", is_child: bool = False, request_id: str = None):
        """
        Bridges the external request to internal LangGraph
        
        Args:
            location: (lat, lng) tuple
            accident_type: Type of emergency
            severity: LOW, MED, HIGH, or CRITICAL
            is_child: Whether it's a pediatric case
            request_id: Unique identifier for the emergency
        """
        import uuid
        
        # --- ZYND SECURITY CHECK ---
        # Simulate verifying the "Citizen App" or "Dispatcher" before processing
        # In a real scenario, the caller would provide their DID.
        caller_did = "did:zynd:citizen:app_v1" 
        if not self.zynd_agent.verify_handshake(caller_did, "Credential:Authorized_Responder"):
             return {
                "status": "error",
                "message": "Security Verification Failed: Unauthorized Agent"
            }
        # ---------------------------
        
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        inputs = {
            "request_id": request_id,
            "accident_location": location,
            "accident_type": accident_type,
            "severity_level": severity,
            "is_child": is_child,
            
            # Initialize placeholders
            "assigned_ambulance": "Waiting",
            "ambulance_eta": 0.0,
            "ambulance_distance": 0.0,
            "ambulance_explanation": "",
            "ambulance_route": {},
            "ambulance_meta": {},
            
            "selected_hospital_id": "Waiting",
            "selected_hospital_name": "Waiting",
            "selected_hospital_lat": 0.0,
            "selected_hospital_lng": 0.0,
            "hospital_distance": 0.0,
            "hospital_explanation": "",
            "hospital_icu_beds": 0,
            "hospital_specialties": "",
            
            "selected_doctor_name": "Waiting",
            "selected_doctor_specialty": "Waiting",
            "doctor_explanation": "",
            
            "final_plan": "Waiting",
            "complete_summary": {}
        }
        
        # Run the real LangGraph logic
        result = self.workflow.invoke(inputs)
        
        # Return formatted results with all details
        return {
            "status": "success",
            "request_id": request_id,
            "accident_type": accident_type,
            "severity": severity,
            "zynd_agent_id": getattr(self.zynd_agent, 'identity', 'Unknown'),
            
            # Ambulance info
            "ambulance": result.get("assigned_ambulance"),
            "eta": result.get("ambulance_eta"),
            "ambulance_distance": result.get("ambulance_distance"),
            "ambulance_explanation": result.get("ambulance_explanation"),
            "ambulance_route": result.get("ambulance_route"),
            
            # Hospital info
            "hospital": result.get("selected_hospital_name"),
            "hospital_id": result.get("selected_hospital_id"),
            "selected_hospital_lat": result.get("selected_hospital_lat"),
            "selected_hospital_lng": result.get("selected_hospital_lng"),
            "hospital_distance": result.get("hospital_distance"),
            "hospital_icu_beds": result.get("hospital_icu_beds"),
            "hospital_explanation": result.get("hospital_explanation"),
            
            # Doctor info
            "doctor_name": result.get("selected_doctor_name"),
            "doctor_specialty": result.get("selected_doctor_specialty"),
            "doctor_explanation": result.get("doctor_explanation"),
            
            # Plans
            "medical_plan": result.get("final_plan"),
            "final_plan": result.get("final_plan"),
            "complete_summary": result.get("complete_summary"),
            
            # Emergency details
            "emergency_summary": {
                "type": accident_type,
                "severity": severity,
                "location": location,
                "is_child": is_child
            }
        }

# Singleton instance used by app.py
emergency_system = EmergencyResponseAgent()