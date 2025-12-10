import os
import requests
from dotenv import load_dotenv

# Load .env with GOOGLE_MAPS_API_KEY inside it
load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    print("❌ API key not found! Please set GOOGLE_MAPS_API_KEY in .env")
    exit()

# Two sample coordinates in Kolkata
origin = "22.5726,88.3639"  # New Market Area
destination = "22.5145,88.3468"  # Ruby Hospital

url = "https://maps.googleapis.com/maps/api/directions/json"
params = {
    "origin": origin,
    "destination": destination,
    "departure_time": "now",  # enable live traffic ETA
    "key": API_KEY,
}

print("🚀 Sending test request to Google Directions API...")
response = requests.get(url, params=params)
data = response.json()

if data.get("status") == "OK":
    leg = data["routes"][0]["legs"][0]
    print("\n🎯 Google API Working Successfully!")
    print("Distance:", leg["distance"]["text"])
    print("Estimated Travel Time:", leg.get("duration_in_traffic", leg["duration"])["text"])
else:
    print("❌ Google API Error:", data.get("status"))
    print("Message:", data.get("error_message"))
