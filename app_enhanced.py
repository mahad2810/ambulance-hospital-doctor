import streamlit as st
import pandas as pd
import json
import time
import datetime as dt
import uuid
import os
import numpy as np
import folium
import requests
import polyline
from streamlit_folium import st_folium
from folium.plugins import HeatMap, AntPath
from dotenv import load_dotenv
import openai
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure APIs (Ensure keys are in .env)
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Import the backend system
from zynd_wrapper import emergency_system

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def encode_image_to_base64(pil_image):
    """Encodes a PIL image to base64 string."""
    buffered = BytesIO()
    # Convert RGBA to RGB to avoid JPEG error
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_multimodal_data(image_data=None, voice_text=None, vitals=None):
    """
    Analyzes injury image and voice notes using OpenAI (Vision) or Groq (Llama).
    Returns a structured analysis dictionary.
    """
    analysis = {
        "visual_assessment": "No image provided",
        "voice_summary": "No voice note provided",
        "risk_score": "Unknown",
        "recommendations": []
    }
    
    # 1. Vision Analysis
    # Priority: OpenAI (GPT-4o) -> Groq (Llama Vision) -> Gemini (Fallback)
    if image_data:
        try:
            # Convert PIL image to base64 for API calls
            base64_image = encode_image_to_base64(image_data)
            
            if os.getenv("OPENAI_API_KEY"):
                # Use OpenAI GPT-4o
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this medical injury image. Identify visible trauma, estimated severity (1-10), and immediate first aid needed. Keep it concise."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                analysis["visual_assessment"] = response.choices[0].message.content
                
            elif os.getenv("GROQ_API_KEY"):
                # Use Groq Llama Vision (Preview)
                from groq import Groq
                client = Groq()
                completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this medical injury image. Identify visible trauma, estimated severity (1-10), and immediate first aid needed. Keep it concise."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ],
                        }
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
                analysis["visual_assessment"] = completion.choices[0].message.content
                
            elif os.getenv("GOOGLE_API_KEY"):
                # Fallback to Gemini
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(["Analyze this injury.", image_data])
                analysis["visual_assessment"] = response.text
                
        except Exception as e:
            analysis["visual_assessment"] = f"Error analyzing image: {e}"

    # 2. Voice/Text Analysis
    # Priority: Groq (Llama 3) -> OpenAI (GPT-4o)
    if voice_text:
        try:
            llm = None
            if os.getenv("GROQ_API_KEY"):
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            elif os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
            if llm:
                prompt = f"Summarize this paramedic voice note into medical terms: '{voice_text}'"
                res = llm.invoke(prompt)
                analysis["voice_summary"] = res.content
            else:
                analysis["voice_summary"] = "No LLM API key available."
                
        except Exception as e:
            analysis["voice_summary"] = f"Error processing voice: {e}"
            
    return analysis

def generate_comprehensive_report(static_data, vitals_history, analysis_history):
    """Generates a full report for the hospital/doctor using Llama 3 (Groq) or OpenAI."""
    try:
        llm = None
        if os.getenv("GROQ_API_KEY"):
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
        elif os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
            
        if not llm:
            return "Error: No API Key (Groq or OpenAI) found."
        
        vitals_str = "\n".join([f"{v['time']}: HR={v['pulse']}, BP={v['bp']}, SpO2={v['spo2']}" for v in vitals_history[-5:]])
        
        # Format analysis history
        analysis_str = ""
        if isinstance(analysis_history, list):
            for idx, a in enumerate(analysis_history):
                analysis_str += f"\n--- Update #{idx+1} ({a.get('timestamp', 'Unknown')}) ---\n"
                analysis_str += f"Visual: {a.get('visual_assessment', 'N/A')}\n"
                analysis_str += f"Voice: {a.get('voice_summary', 'N/A')}\n"
        elif isinstance(analysis_history, dict):
             # Fallback for legacy single dict
             analysis_str = f"Visual: {analysis_history.get('visual_assessment')}\nVoice: {analysis_history.get('voice_summary')}"

        prompt = f"""
        Generate a Medical Handover Report for a Hospital Team.
        
        PATIENT INFO:
        - Incident: {static_data.get('accident_type')}
        - Severity: {static_data.get('severity')}
        
        VITALS TREND (Last 5 readings):
        {vitals_str}
        
        MULTIMODAL ASSESSMENT LOG (Chronological):
        {analysis_str}
        
        OUTPUT SECTIONS:
        1. IMMEDIATE PREPARATION (Equipment/Staff needed)
        2. PATIENT STATUS SUMMARY (Evolution of condition)
        3. SUGGESTED TRIAGE CATEGORY
        """
        res = llm.invoke(prompt)
        return res.content
    except Exception as e:
        return "AI Report Generation Failed: " + str(e)

def get_route_coordinates(start_lat, start_lng, end_lat, end_lng):
    """Fetch real driving route from Google Directions API"""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return [[start_lat, start_lng], [end_lat, end_lng]]
    
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lng}",
        "destination": f"{end_lat},{end_lng}",
        "key": api_key,
        "mode": "driving"
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            points = data['routes'][0]['overview_polyline']['points']
            decoded_points = polyline.decode(points)
            return decoded_points
        else:
            # Fallback to straight line if API fails or limit reached
            return [[start_lat, start_lng], [end_lat, end_lng]]
    except Exception as e:
        return [[start_lat, start_lng], [end_lat, end_lng]]

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="SmartAgent-ER: Emergency Response System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data():
    try:
        ambulances = pd.read_csv('ambulances_kolkata.csv')
        hospitals = pd.read_csv('hospitals_kolkata.csv')
        zones = pd.read_csv('zones_kolkata.csv')
        risk = pd.read_csv('risk_profiles_kolkata.csv')
        
        # Merge risk data with zones
        # Assuming risk profile has zone_id and we want the latest risk score
        latest_risk = risk.sort_values('time_window').groupby('zone_id').last().reset_index()
        zones_risk = pd.merge(zones, latest_risk[['zone_id', 'risk_score']], on='zone_id', how='left')
        zones_risk['risk_score'] = zones_risk['risk_score'].fillna(0.1)
        
        return ambulances, hospitals, zones_risk
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_ambulances, df_hospitals, df_zones = load_data()

# ==========================================
# SESSION STATE
# ==========================================

if 'active_emergency' not in st.session_state:
    st.session_state.active_emergency = None

if 'vitals_history' not in st.session_state:
    st.session_state.vitals_history = []

if 'multimodal_analysis' not in st.session_state:
    st.session_state.multimodal_analysis = {}

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'emergency_history' not in st.session_state:
    st.session_state.emergency_history = []

if 'mission_status' not in st.session_state:
    st.session_state.mission_status = "IDLE"  # IDLE, DISPATCHED, EN_ROUTE, AT_SCENE, TRANSPORTING, HANDOVER

if 'user_role' not in st.session_state:
    st.session_state.user_role = "Citizen"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
    <style>
    /* Global Styles */
    .main { 
        background-color: #1a1a1a; 
        color: #e0e0e0;
    }
    
    /* Cards */
    .stCard {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        border: 1px solid #404040;
    }
    
    /* Role Header */
    .role-header {
        padding: 15px 25px;
        background: #2d2d2d;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        justify_content: space-between;
        color: #ffffff;
    }
    
    /* Status Badges */
    .status-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-idle { background: #404040; color: #aaa; }
    .status-active { background: #1e3a8a; color: #60a5fa; }
    .status-critical { background: #7f1d1d; color: #fca5a5; }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, label, span { color: #e0e0e0 !important; }
    
    /* Action Buttons */
    .big-button {
        width: 100%;
        padding: 15px;
        border-radius: 10px;
        border: none;
        color: white;
        font-weight: bold;
        font-size: 18px;
        cursor: pointer;
        transition: transform 0.1s;
    }
    .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .btn-success { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }
    .btn-warning { background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); }
    .btn-danger { background: linear-gradient(135deg, #F44336 0%, #D32F2F 100%); }
    
    .big-button:active { transform: scale(0.98); }
    
    /* Map Container */
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def render_header(title, subtitle, icon="🚑"):
    st.markdown(f"""
        <div class="role-header">
            <div>
                <h2 style="margin:0; color:#ffffff;">{icon} {title}</h2>
                <p style="margin:5px 0 0 0; color:#cccccc; font-size:14px;">{subtitle}</p>
            </div>
            <div class="status-badge status-active">System Online</div>
        </div>
    """, unsafe_allow_html=True)

def get_status_color(status):
    colors = {
        "IDLE": "#9E9E9E",
        "DISPATCHED": "#2196F3",
        "EN_ROUTE": "#FF9800",
        "AT_SCENE": "#9C27B0",
        "TRANSPORTING": "#E91E63",
        "HANDOVER": "#4CAF50"
    }
    return colors.get(status, "#9E9E9E")

# ==========================================
# VIEWS
# ==========================================

def view_citizen():
    st.markdown("""
        <style>
        .emergency-header {
            background-color: #7f1d1d;
            color: white;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
        }
        .report-form {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #334155;
        }
        .status-ticket {
            background-color: #0f172a;
            border-left: 4px solid #10b981;
            padding: 20px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="emergency-header"><h2>🚑 Emergency Incident Reporting</h2><p>Please provide accurate details for immediate assistance.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Incident Details")
        
        with st.form("emergency_report_form"):
            st.markdown("#### Location Coordinates")
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", value=22.5726, format="%.6f")
            with c2:
                lng = st.number_input("Longitude", value=88.3639, format="%.6f")
            
            st.markdown("#### Clinical Information")
            accident_type = st.selectbox("Nature of Emergency", [
                "Trauma - Road Accident", "Cardiac Arrest", "Respiratory Failure", 
                "Stroke", "Burn Injury", "Chest Pain", "Unconscious / Fainting", "Severe Bleeding"
            ])
            
            severity = st.select_slider("Triage Severity Level", options=["LOW", "MED", "HIGH", "CRITICAL"], value="HIGH", help="Critical: Life-threatening, immediate intervention required.")
            
            st.markdown("#### Patient Demographics")
            is_child = st.checkbox("Pediatric Patient (<18 years)")
            
            submitted = st.form_submit_button("🚨 INITIATE EMERGENCY RESPONSE", use_container_width=True, type="primary")
            
            if submitted:
                with st.spinner("Dispatching Emergency Services..."):
                    # Reset previous mission data
                    st.session_state.vitals_history = []
                    st.session_state.multimodal_analysis = {}
                    st.session_state.analysis_history = []
                    st.session_state.patient_vitals = {'pulse': 80, 'bp': 120, 'spo2': 98}

                    # Simulate processing
                    request_id = str(uuid.uuid4())[:8]
                    response = emergency_system.process_emergency(
                        location=(lat, lng),
                        accident_type=accident_type,
                        severity=severity,
                        is_child=is_child,
                        request_id=request_id
                    )
                    
                    # Store in session
                    response['incident_lat'] = lat
                    response['incident_lng'] = lng
                    response['hospital_lat'] = response.get('selected_hospital_lat', 0)
                    response['hospital_lng'] = response.get('selected_hospital_lng', 0)
                    
                    st.session_state.active_emergency = response
                    st.session_state.mission_status = "DISPATCHED"
                    st.session_state.emergency_history.append(response)
                    
                    st.toast("Emergency Signal Broadcasted Successfully", icon="📡")

    with col2:
        st.markdown("### 📡 Dispatch Status")
        if st.session_state.active_emergency:
            data = st.session_state.active_emergency
            
            st.markdown(f"""
                <div class="status-ticket">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <span style="color:#94a3b8; font-size:12px;">INCIDENT ID: #{data.get('request_id', 'UNK')}</span>
                        <span style="background:#10b981; color:black; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold;">DISPATCHED</span>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <h1 style="color:#fff; font-size:36px; margin:0;">{data.get('eta')} <span style="font-size:16px; color:#94a3b8;">MIN ETA</span></h1>
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:15px;">
                        <div style="background:#1e293b; padding:10px; border-radius:4px;">
                            <small style="color:#94a3b8;">AMBULANCE UNIT</small><br>
                            <strong style="color:#fff;">{data.get('ambulance')}</strong>
                        </div>
                        <div style="background:#1e293b; padding:10px; border-radius:4px;">
                            <small style="color:#94a3b8;">DISTANCE</small><br>
                            <strong style="color:#fff;">{data.get('ambulance_distance')} km</strong>
                        </div>
                    </div>
                    <div style="background:#1e293b; padding:10px; border-radius:4px;">
                        <small style="color:#94a3b8;">DESTINATION FACILITY</small><br>
                        <strong style="color:#fff;">{data.get('hospital')}</strong>
                    </div>
                    <hr style="border-color:#334155; margin:15px 0;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <div style="background:#3b82f6; width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center;">👨‍⚕️</div>
                        <div>
                            <small style="color:#94a3b8;">ASSIGNED SPECIALIST</small><br>
                            <strong style="color:#fff;">{data.get('doctor_name', 'Pending Assignment')}</strong>
                            <br><span style="color:#64748b; font-size:12px;">{data.get('doctor_specialty', '')}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Professional Map
            m = folium.Map(location=[data['incident_lat'], data['incident_lng']], zoom_start=14, tiles="CartoDB positron")
            folium.Marker(
                [data['incident_lat'], data['incident_lng']], 
                popup="Incident Location",
                icon=folium.Icon(color="red", icon="exclamation-sign")
            ).add_to(m)
            
            st_folium(m, width="100%", height=300)
        else:
            st.info("System Ready. Awaiting Incident Report.")
            st.markdown("""
                <div style="text-align:center; padding:40px; color:#64748b; border: 2px dashed #334155; border-radius:10px;">
                    <h3>No Active Incidents</h3>
                    <p>Fill out the form to request immediate emergency assistance.</p>
                </div>
            """, unsafe_allow_html=True)

def view_admin():
    render_header("Command Center", "Admin Dashboard", "🏢")
    
    tabs = st.tabs(["🗺️ Live Map", "📊 Fleet Status", "🔥 Risk Analysis"])
    
    with tabs[0]:
        # Initialize Folium Map
        m = folium.Map(location=[22.5726, 88.3639], zoom_start=11, tiles="CartoDB positron")
        
        # 1. Risk Heatmap
        if not df_zones.empty:
            heat_data = [[row['center_lat'], row['center_lng'], row['risk_score']] for index, row in df_zones.iterrows()]
            HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # 2. Hospitals
        if not df_hospitals.empty:
            for _, row in df_hospitals.iterrows():
                folium.Marker(
                    [row['lat'], row['lng']],
                    popup=f"{row['hospital_name']} (ICU: {row['icu_beds_free']})",
                    icon=folium.Icon(color="green", icon="plus-sign"),
                    tooltip=row['hospital_name']
                ).add_to(m)
            
        # 3. Ambulances
        if not df_ambulances.empty:
            for _, row in df_ambulances.iterrows():
                folium.Marker(
                    [row['current_lat'], row['current_lng']],
                    popup=f"{row['ambulance_id']} ({row['status']})",
                    icon=folium.Icon(color="blue", icon="ambulance", prefix="fa"),
                    tooltip=row['ambulance_id']
                ).add_to(m)
            
        # 4. Active Route
        if st.session_state.active_emergency:
            data = st.session_state.active_emergency
            start_coords = [data.get('ambulance_route', {}).get('from_lat'), data.get('ambulance_route', {}).get('from_lng')]
            end_coords = [data['incident_lat'], data['incident_lng']]
            
            if start_coords[0] and end_coords[0]:
                # Get real road route
                route_path = get_route_coordinates(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
                
                folium.PolyLine(
                    locations=route_path,
                    color="orange",
                    weight=5,
                    opacity=0.8,
                    tooltip="Active Emergency Route"
                ).add_to(m)

        st_folium(m, width="100%", height=500)
        
        # Active Emergencies List
        st.markdown("### 🚨 Active Incidents")
        if st.session_state.active_emergency:
            e = st.session_state.active_emergency
            st.info(f"INCIDENT #{e.get('request_id', '???')} | {e.get('accident_type')} | Status: {st.session_state.mission_status}")
            
            # Doctor Assignment Card
            st.markdown(f"""
                <div class="stCard">
                    <h4 style="color: #667eea;">👨‍⚕️ Medical Team Assigned</h4>
                    <p><strong>Doctor:</strong> {e.get('doctor_name', 'Pending')}</p>
                    <p><strong>Specialty:</strong> {e.get('doctor_specialty', 'N/A')}</p>
                    <p><strong>Notes:</strong> {e.get('doctor_explanation', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.write("No active incidents.")

    with tabs[2]:
        st.markdown("### 📈 Predictive Risk & Reallocation")
        
        # Mock Reallocation Data for Visualization
        # In a real scenario, this would come from the AI agent's output
        if not df_ambulances.empty and not df_zones.empty:
            # Get top 3 high risk zones
            high_risk_zones = df_zones.sort_values('risk_score', ascending=False).head(3)
            # Get available ambulances (mocking availability by just taking first 3)
            available_ambulances = df_ambulances.head(3)
            
            col1, col2 = st.columns([1, 2])
            
            realloc_plans = []
            
            with col1:
                st.markdown("#### 🤖 AI Recommendations")
                
                for i in range(min(len(high_risk_zones), len(available_ambulances))):
                    zone = high_risk_zones.iloc[i]
                    amb = available_ambulances.iloc[i]
                    realloc_plans.append((amb, zone))
                    
                    with st.expander(f"Plan #{i+1}: Move {amb['ambulance_id']} -> {zone['zone_name']}", expanded=True):
                        st.write(f"**Reason:** High Risk Score ({zone['risk_score']})")
                        if st.button(f"Apply Plan #{i+1}", key=f"btn_realloc_{i}"):
                            st.toast(f"Dispatching {amb['ambulance_id']} to {zone['zone_name']}")
                
                st.markdown("---")
                st.markdown("**High Risk Zones**")
                st.dataframe(df_zones[['zone_name', 'risk_score']].sort_values('risk_score', ascending=False).head(5), use_container_width=True)

            with col2:
                st.markdown("#### 📍 Reallocation Plan Map")
                
                if realloc_plans:
                    # Center on the first target zone
                    center_zone = realloc_plans[0][1]
                    m_realloc = folium.Map(
                        location=[center_zone['center_lat'], center_zone['center_lng']], 
                        zoom_start=11,
                        tiles="CartoDB positron"
                    )
                    
                    colors = ['red', 'orange', 'purple']
                    
                    for idx, (amb, zone) in enumerate(realloc_plans):
                        color = colors[idx % len(colors)]
                        
                        # 1. Target Zone (Circle)
                        folium.Circle(
                            location=[zone['center_lat'], zone['center_lng']],
                            radius=800,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.2,
                            popup=f"Zone: {zone['zone_name']} (Risk: {zone['risk_score']})"
                        ).add_to(m_realloc)
                        
                        # 2. Ambulance Position
                        folium.Marker(
                            [amb['current_lat'], amb['current_lng']],
                            popup=f"Ambulance: {amb['ambulance_id']}",
                            icon=folium.Icon(color="blue", icon="ambulance", prefix="fa")
                        ).add_to(m_realloc)
                        
                        # 3. Movement Path (Arrow/Line)
                        realloc_path = get_route_coordinates(
                            amb['current_lat'], amb['current_lng'],
                            zone['center_lat'], zone['center_lng']
                        )
                        
                        folium.PolyLine(
                            locations=realloc_path,
                            color=color,
                            weight=3,
                            opacity=0.8,
                            dash_array='10',
                            tooltip=f"Plan #{idx+1}"
                        ).add_to(m_realloc)
                    
                    st_folium(m_realloc, width="100%", height=400)
        else:
            st.warning("Insufficient data for reallocation analysis.")

def view_ambulance():
    amb_id = "Unit: AMB-001 (BLS)"
    if st.session_state.active_emergency and st.session_state.active_emergency.get('ambulance'):
        amb_id = f"Unit: {st.session_state.active_emergency.get('ambulance')}"
        
    render_header("Ambulance Interface", amb_id, "🚑")
    
    if not st.session_state.active_emergency:
        st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <h1 style="color: #ccc; font-size: 60px;">💤</h1>
                <h3>Status: IDLE</h3>
                <p>Waiting for dispatch instructions...</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Active Mission
    data = st.session_state.active_emergency
    status = st.session_state.mission_status
    
    # Progress Bar
    steps = ["DISPATCHED", "EN_ROUTE", "AT_SCENE", "TRANSPORTING", "HANDOVER"]
    try:
        current_idx = steps.index(status)
    except:
        current_idx = 0
    st.progress((current_idx + 1) / len(steps))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Map
        st.markdown("### 🗺️ Navigation")
        
        # Determine route based on status
        start_pos = [data.get('ambulance_route', {}).get('from_lng'), data.get('ambulance_route', {}).get('from_lat')]
        end_pos = [data['incident_lng'], data['incident_lat']]
        
        if status in ["TRANSPORTING", "HANDOVER"]:
            start_pos = [data['incident_lng'], data['incident_lat']]
            end_pos = [data['hospital_lng'], data['hospital_lat']]
            
        # Route Line (Folium)
        m_nav = folium.Map(location=[start_pos[1], start_pos[0]], zoom_start=13, tiles="CartoDB positron")
        
        # Start Marker
        folium.Marker(
            [start_pos[1], start_pos[0]],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m_nav)
        
        # End Marker
        folium.Marker(
            [end_pos[1], end_pos[0]],
            popup="Destination",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m_nav)
        
        # Route Path
        route_path = get_route_coordinates(start_pos[1], start_pos[0], end_pos[1], end_pos[0])
        
        folium.PolyLine(
            locations=route_path,
            color="blue",
            weight=5,
            opacity=0.8
        ).add_to(m_nav)
        
        st_folium(m_nav, width="100%", height=400)
        
    with col2:
        st.markdown("### 📋 Mission Controls")
        st.info(f"Target: {data.get('accident_type', 'Unknown')}")
        
        if data.get('doctor_name'):
            st.markdown(f"""
                <div style="background: #2d2d2d; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #444;">
                    <small style="color: #aaa;">RECEIVING DOCTOR</small><br>
                    <strong style="color: #fff;">{data.get('doctor_name')}</strong><br>
                    <span style="color: #ccc;">{data.get('doctor_specialty')}</span>
                </div>
            """, unsafe_allow_html=True)
        
        # State Machine Buttons
        if status == "DISPATCHED":
            if st.button("✅ ACCEPT MISSION", use_container_width=True):
                st.session_state.mission_status = "EN_ROUTE"
                st.rerun()
        
        elif status == "EN_ROUTE":
            st.warning("🚑 Driving to Scene...")
            if st.button("📍 ARRIVED AT SCENE", use_container_width=True):
                st.session_state.mission_status = "AT_SCENE"
                st.rerun()
                
        elif status == "AT_SCENE":
            st.markdown("#### 🚑 On-Scene Assessment")
            
            # 1. Multimodal Input
            with st.expander("📸 Injury Assessment (Vision & Voice)", expanded=True):
                img_file = st.file_uploader("Upload Injury Image", type=['png', 'jpg', 'jpeg'])
                voice_note = st.text_area("🎙️ Voice Note / Description", placeholder="Describe patient condition...")
                
                if st.button("🔍 Analyze Condition"):
                    with st.spinner("Analyzing with AI..."):
                        # Convert image to bytes if needed
                        img_bytes = None
                        if img_file:
                            import PIL.Image
                            img = PIL.Image.open(img_file)
                            img_bytes = img
                        
                        analysis = analyze_multimodal_data(img_bytes, voice_note)
                        analysis['timestamp'] = dt.datetime.now().strftime("%H:%M:%S")
                        
                        st.session_state.multimodal_analysis = analysis
                        st.session_state.analysis_history.append(analysis)
                        
                        st.success("Analysis Sent to Hospital!")
                        st.write(analysis)

            # 2. Vitals
            st.markdown("#### 🩺 Vitals Log")
            if 'patient_vitals' not in st.session_state:
                st.session_state.patient_vitals = {'pulse': 80, 'bp': 120, 'spo2': 98}
            
            c1, c2, c3 = st.columns(3)
            with c1:
                p = st.number_input("Pulse (BPM)", value=st.session_state.patient_vitals['pulse'], key="v_pulse_init")
            with c2:
                b = st.number_input("BP (Sys)", value=st.session_state.patient_vitals['bp'], key="v_bp_init")
            with c3:
                s = st.number_input("SpO2 (%)", value=st.session_state.patient_vitals['spo2'], key="v_spo2_init")
            
            if st.button("➕ Log Vitals"):
                timestamp = dt.datetime.now().strftime("%H:%M:%S")
                entry = {'time': timestamp, 'pulse': p, 'bp': b, 'spo2': s}
                st.session_state.vitals_history.append(entry)
                st.session_state.patient_vitals = {'pulse': p, 'bp': b, 'spo2': s}
                st.success(f"Logged at {timestamp}")

            if st.button("🛌 PATIENT ONBOARD", use_container_width=True):
                st.session_state.mission_status = "TRANSPORTING"
                st.rerun()
                
        elif status == "TRANSPORTING":
            st.info(f"Destination: {data['hospital']}")
            
            # Multimodal Input (Available during transport too)
            with st.expander("📸 Injury Assessment (Vision & Voice)", expanded=False):
                img_file = st.file_uploader("Upload Injury Image", type=['png', 'jpg', 'jpeg'], key="img_transport")
                voice_note = st.text_area("🎙️ Voice Note / Description", placeholder="Describe patient condition...", key="voice_transport")
                
                if st.button("🔍 Analyze Condition", key="btn_analyze_transport"):
                    with st.spinner("Analyzing with AI..."):
                        # Convert image to bytes if needed
                        img_bytes = None
                        if img_file:
                            import PIL.Image
                            img = PIL.Image.open(img_file)
                            img_bytes = img
                        
                        analysis = analyze_multimodal_data(img_bytes, voice_note)
                        analysis['timestamp'] = dt.datetime.now().strftime("%H:%M:%S")
                        
                        st.session_state.multimodal_analysis = analysis
                        st.session_state.analysis_history.append(analysis)
                        
                        st.success("Analysis Sent to Hospital!")
                        st.write(analysis)

            st.markdown("#### 🩺 Continuous Monitoring")
            
            # Vitals Input
            if 'patient_vitals' not in st.session_state:
                st.session_state.patient_vitals = {'pulse': 80, 'bp': 120, 'spo2': 98}

            c1, c2, c3 = st.columns(3)
            with c1:
                p = st.number_input("Pulse", value=st.session_state.patient_vitals['pulse'], key="v_pulse_live")
            with c2:
                b = st.number_input("BP", value=st.session_state.patient_vitals['bp'], key="v_bp_live")
            with c3:
                s = st.number_input("SpO2", value=st.session_state.patient_vitals['spo2'], key="v_spo2_live")
            
            if st.button("➕ Update Vitals Log"):
                timestamp = dt.datetime.now().strftime("%H:%M:%S")
                entry = {'time': timestamp, 'pulse': p, 'bp': b, 'spo2': s}
                st.session_state.vitals_history.append(entry)
                st.session_state.patient_vitals = {'pulse': p, 'bp': b, 'spo2': s}
                st.toast("Vitals Updated")
            
            # Show History
            if st.session_state.vitals_history:
                st.dataframe(pd.DataFrame(st.session_state.vitals_history).tail(3))
            
            if st.button("🏥 ARRIVED AT HOSPITAL", use_container_width=True):
                st.session_state.mission_status = "HANDOVER"
                st.rerun()
                
        elif status == "HANDOVER":
            st.success("Mission Complete")
            if st.button("🏁 CLEAR & RETURN TO BASE", use_container_width=True):
                st.session_state.active_emergency = None
                st.session_state.mission_status = "IDLE"
                st.rerun()

def view_hospital():
    hospital_name = "MediPoint Hospital QA"
    if st.session_state.active_emergency and st.session_state.active_emergency.get('hospital'):
        hospital_name = st.session_state.active_emergency.get('hospital')
        
    render_header("Hospital Console", hospital_name, "🏥")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚑 Incoming Ambulances")
        if st.session_state.active_emergency and st.session_state.mission_status in ["TRANSPORTING", "HANDOVER"]:
            data = st.session_state.active_emergency
            st.warning(f"INCOMING: {data.get('accident_type', 'Unknown')} (Severity: {data.get('severity', 'HIGH')})")
            st.write(f"ETA: 5 mins | Ambulance: {data['ambulance']}")
            
            st.markdown(f"""
                <div class="stCard">
                    <h4 style="color: #667eea; margin-top:0;">👨‍⚕️ Assigned Specialist</h4>
                    <p><strong>Doctor:</strong> {data.get('doctor_name', 'Pending')}</p>
                    <p><strong>Specialty:</strong> {data.get('doctor_specialty', 'N/A')}</p>
                    <p><strong>AI Notes:</strong> {data.get('doctor_explanation', 'N/A')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Suggestions
            st.markdown("#### 🏥 AI Readiness Suggestions")
            
            # Show latest analysis
            if st.session_state.analysis_history:
                latest = st.session_state.analysis_history[-1]
                if latest.get('visual_assessment'):
                    st.info(f"**Latest Visual Analysis ({latest.get('timestamp')}):** {latest['visual_assessment'][:150]}...")
            
            st.markdown("#### 📋 Dynamic Preparation Checklist")
            
            # Determine context for dynamic checking
            incident_type = data.get('accident_type', '').lower()
            severity = data.get('severity', 'HIGH')
            
            # Check history for keywords from AI analysis
            all_visuals = " ".join([a.get('visual_assessment', '') for a in st.session_state.analysis_history]).lower()
            all_voice = " ".join([a.get('voice_summary', '') for a in st.session_state.analysis_history]).lower()
            combined_context = f"{incident_type} {all_visuals} {all_voice}"
            
            # Logic for checkboxes
            is_trauma = "trauma" in combined_context or "accident" in combined_context or "fracture" in combined_context
            is_cardiac = "cardiac" in combined_context or "chest" in combined_context or "heart" in combined_context or "stemi" in combined_context
            is_resp = "respiratory" in combined_context or "breath" in combined_context
            is_critical = severity in ["HIGH", "CRITICAL"]
            
            c1, c2 = st.columns(2)
            with c1:
                st.checkbox("Trauma Team Alerted", value=is_trauma)
                st.checkbox("Cath Lab Activated", value=is_cardiac)
                st.checkbox("Stroke Team Alerted", value=("stroke" in combined_context))
            with c2:
                st.checkbox("ICU Bed Reserved", value=is_critical)
                st.checkbox("Blood Bank Notified", value=(is_trauma and is_critical))
                st.checkbox("Ventilator Prepped", value=is_resp)

        else:
            st.success("No incoming emergencies.")
            
    with col2:
        st.markdown("### 👨‍⚕️ On-Call Staff Status")
        
        # Mock enhanced data
        doctors_data = [
            {"Name": "Dr. Arindam Roy", "Specialty": "Trauma Surgery", "Status": "Available", "Load": "0/3"},
            {"Name": "Dr. S. Sen", "Specialty": "Cardiology", "Status": "In Surgery", "Load": "2/3"},
            {"Name": "Dr. A. Das", "Specialty": "Neurology", "Status": "On-Call", "Load": "1/3"},
            {"Name": "Dr. P. Gupta", "Specialty": "Emergency Med", "Status": "Available", "Load": "1/5"},
            {"Name": "Dr. R. Khan", "Specialty": "Orthopedics", "Status": "Available", "Load": "0/4"},
        ]
        
        # If an emergency is active, highlight the assigned doctor
        assigned_doc_name = ""
        if st.session_state.active_emergency:
             assigned_doc_name = st.session_state.active_emergency.get('doctor_name', '')
        
        for doc in doctors_data:
            # Determine color based on status
            status_color = "#4CAF50" if doc['Status'] == "Available" else "#FF9800" if doc['Status'] == "On-Call" else "#F44336"
            
            # Highlight assigned
            is_assigned = assigned_doc_name and (doc['Name'] in assigned_doc_name or assigned_doc_name in doc['Name'])
            bg_color = "#2d3748" if is_assigned else "#1a1c24"
            border = "2px solid #667eea" if is_assigned else "1px solid #333"
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: {border}; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: white; font-size: 14px;">{doc['Name']}</strong>
                        <span style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: bold;">{doc['Status']}</span>
                    </div>
                    <div style="font-size: 12px; color: #aaa; margin-top: 4px; display: flex; justify-content: space-between;">
                        <span>{doc['Specialty']}</span>
                        <span>Load: {doc['Load']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def view_doctor():
    doc_name = "Dr. Arindam Roy (Trauma Lead)"
    if st.session_state.active_emergency and st.session_state.active_emergency.get('doctor_name'):
        doc_name = f"{st.session_state.active_emergency.get('doctor_name')} ({st.session_state.active_emergency.get('doctor_specialty')})"
        
    render_header("Doctor Dashboard", doc_name, "👨‍⚕️")
    
    if st.session_state.active_emergency:
        data = st.session_state.active_emergency
        st.markdown(f"### 🩺 Patient: Unknown Male (Incident #{data.get('request_id')})")
        
        # Tabs for Doctor View
        d_tabs = st.tabs(["📊 Vitals & Telemetry", "📸 Visual & Voice Analysis", "📝 AI Handover Report"])
        
        with d_tabs[0]:
            st.markdown("#### 📊 Live Vitals (Telemetry)")
            
            if st.session_state.vitals_history:
                df_vitals = pd.DataFrame(st.session_state.vitals_history)
                st.line_chart(df_vitals.set_index('time')[['pulse', 'bp', 'spo2']])
                st.dataframe(df_vitals.tail(5))
            else:
                st.warning("No vitals history available yet.")
                if 'patient_vitals' in st.session_state:
                    v = st.session_state.patient_vitals
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pulse", f"{v['pulse']} bpm")
                    c2.metric("BP", f"{v['bp']} sys")
                    c3.metric("SpO2", f"{v['spo2']}%")

        with d_tabs[1]:
            st.markdown("#### 🚑 Multimodal Data Timeline")
            
            if st.session_state.analysis_history:
                for idx, analysis in enumerate(reversed(st.session_state.analysis_history)):
                    with st.expander(f"Analysis #{len(st.session_state.analysis_history)-idx} ({analysis.get('timestamp', 'Unknown Time')})", expanded=(idx==0)):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**📸 Visual Assessment**")
                            if analysis.get('visual_assessment'):
                                st.info(analysis['visual_assessment'])
                            else:
                                st.write("No image analysis.")
                        
                        with c2:
                            st.markdown("**🎙️ Voice Note Summary**")
                            if analysis.get('voice_summary'):
                                st.success(analysis['voice_summary'])
                            else:
                                st.write("No voice notes.")
            else:
                st.info("No multimodal data collected yet.")

        with d_tabs[2]:
            st.markdown("#### 🤖 Comprehensive AI Handover Report")
            if st.button("Generate Full Report"):
                with st.spinner("Synthesizing data..."):
                    # Pass the full history to the report generator
                    report = generate_comprehensive_report(
                        data, 
                        st.session_state.vitals_history, 
                        st.session_state.analysis_history
                    )
                    st.markdown(report)
    else:
        st.write("No active patient assignments.")

# ==========================================
# MAIN APP LOGIC
# ==========================================

# Sidebar Navigation
with st.sidebar:
    st.title("🚑 SmartAgent-ER")
    st.markdown("---")
    selected_role = st.radio(
        "Select User Role",
        ["Citizen", "Admin", "Ambulance", "Hospital", "Doctor"],
        index=["Citizen", "Admin", "Ambulance", "Hospital", "Doctor"].index(st.session_state.user_role)
    )
    st.session_state.user_role = selected_role
    
    st.markdown("---")
    st.caption(f"Session ID: {str(uuid.uuid4())[:8]}")
    if st.button("Reset System"):
        st.session_state.active_emergency = None
        st.session_state.mission_status = "IDLE"
        st.session_state.vitals_history = []
        st.session_state.multimodal_analysis = {}
        st.session_state.analysis_history = []
        if 'patient_vitals' in st.session_state:
            del st.session_state.patient_vitals
        st.rerun()

# Render View based on Role
if st.session_state.user_role == "Citizen":
    view_citizen()
elif st.session_state.user_role == "Admin":
    view_admin()
elif st.session_state.user_role == "Ambulance":
    view_ambulance()
elif st.session_state.user_role == "Hospital":
    view_hospital()
elif st.session_state.user_role == "Doctor":
    view_doctor()
