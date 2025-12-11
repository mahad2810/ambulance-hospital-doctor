import streamlit as st
import pandas as pd
import json
import time
import re
import ast

from zynd_wrapper import emergency_system

st.set_page_config(page_title="ZyndAI Emergency Response", page_icon="🚑", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚑 SmartAgent-ER: Autonomous Emergency Orchestrator")
st.caption("Powered by ZyndAI Agents • LangGraph • Vector Search")

# Sidebar - Inputs
with st.sidebar:
    st.header("🚨 Incident Report")
    
    # Simulated Map Input (In a real app, use st_folium for clicks)
    st.subheader("📍 Location")
    lat = st.number_input("Latitude", value=22.5726, format="%.4f")
    lng = st.number_input("Longitude", value=88.3639, format="%.4f")
    
    st.subheader("📝 Details")
    accident_type = st.selectbox(
        "Incident Type",
        ["Cardiac Arrest", "Trauma - Road Accident", "Respiratory Failure", "Stroke", "Burn Injury"]
    )
    severity = st.slider("Severity Level", 1, 10, 9)
    
    st.markdown("---")
    st.info("System Ready. Waiting for dispatch signal.")

# Main Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Live Operation Map")
    # Simple map visualization of the incident
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lng]})
    st.map(map_data, zoom=13)

with col2:
    st.subheader("⚡ Agent Activity Log")
    log_container = st.empty()

# Dispatch Button
if st.button("🚀 DISPATCH EMERGENCY AGENTS"):
    
    # 1. Simulation UI Effects
    with st.spinner('🤖 ZyndAI Agent: analyzing incident...'):
        time.sleep(1) # UX pause
        
    log_container.markdown("✅ **Incident Received**")
    time.sleep(0.5)
    log_container.markdown("✅ **Incident Received**\n\n🔄 **Broadcasting to Ambulances...**")
    
    # 2. CALL THE BACKEND (The scalable part)
    # We call the wrapper, which calls the core engine
    try:
        response = emergency_system.process_emergency((lat, lng), accident_type)
        
        # 3. Display Results
        st.success("🎉 Orchestration Complete!")
        
        # Create 3 metrics columns
        m1, m2, m3 = st.columns(3)
        m1.metric("Ambulance", response['ambulance'], f"{response['eta']} min")
        m2.metric("Hospital", response['hospital'], "Bed Confirmed")
        m3.metric("Agent Status", "Active", "ZyndAI-Core")
        
        # Display the JSON Plan beautifully
        st.subheader("📋 Medical Handover Plan")
        
        # Try to parse the string JSON from the LLM robustly. LLMs often
        # return text with surrounding markdown or additional commentary.
        def extract_json_from_text(text: str):
            # 1) If it's already a dict/list (not a string), return as-is
            if isinstance(text, (dict, list)):
                return text

            # 2) Try direct json.loads
            try:
                return json.loads(text)
            except Exception:
                pass

            # 3) Try to find a JSON object or array within the text using regex
            # This looks for the first {...} or [...] block.
            json_match = re.search(r"(\{(?:.|\n)*\}|\[(?:.|\n)*\])", str(text))
            if json_match:
                candidate = json_match.group(0)
                try:
                    return json.loads(candidate)
                except Exception:
                    # Try python literal eval as a last resort (handles single quotes)
                    try:
                        return ast.literal_eval(candidate)
                    except Exception:
                        return None

            # 4) Try ast.literal_eval on the whole string (handles python dict repr)
            try:
                return ast.literal_eval(str(text))
            except Exception:
                return None

        plan_raw = response.get('medical_plan') or response.get('final_plan') or response
        parsed = extract_json_from_text(plan_raw)

        if parsed is not None:
            st.json(parsed)
        else:
            st.warning("Raw Plan (LLM output not strict JSON):")
            st.write(plan_raw)
            
    except Exception as e:
        st.error(f"System Error: {str(e)}")