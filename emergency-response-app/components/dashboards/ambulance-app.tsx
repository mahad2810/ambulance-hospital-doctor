"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import Header from "@/components/shared/header"
import { MapPin, Navigation, Heart, Activity, Thermometer, Wind, Upload, Check, X, Clock } from "lucide-react"

interface IncomingRequest {
  id: string
  type: string
  location: string
  coordinates: { lat: number; lng: number }
  priority: "critical" | "high" | "medium"
  patientInfo: string
  distance: string
  eta: string
  timestamp: string
}

interface Assignment {
  id: string
  type: string
  patientLocation: string
  patientCoordinates: { lat: number; lng: number }
  hospitalName?: string
  hospitalLocation?: string
  hospitalCoordinates?: { lat: number; lng: number }
  priority: "critical" | "high" | "medium"
  patientInfo: string
  distance: string
  eta: string
  status: "en-route-to-patient" | "on-scene" | "patient-loaded" | "en-route-to-hospital" | "delivered"
}

interface VitalSign {
  heartRate: string
  bloodPressure: string
  temperature: string
  oxygenSaturation: string
  respiratoryRate: string
  notes: string
}

interface AmbulanceAppProps {
  userName: string
  onLogout: () => void
}

const MOCK_INCOMING_REQUESTS: IncomingRequest[] = [
  {
    id: "req-1",
    type: "Cardiac Emergency",
    location: "Park Street Metro Station",
    coordinates: { lat: 22.5552, lng: 88.3503 },
    priority: "critical",
    patientInfo: "Male, 58y, severe chest pain, difficulty breathing",
    distance: "1.2 km",
    eta: "3 min",
    timestamp: "2 min ago",
  },
  {
    id: "req-2",
    type: "Traffic Accident",
    location: "Esplanade & BBD Bagh",
    coordinates: { lat: 22.5698, lng: 88.3538 },
    priority: "high",
    patientInfo: "Female, 32y, head injury, conscious",
    distance: "2.8 km",
    eta: "7 min",
    timestamp: "5 min ago",
  },
  {
    id: "req-3",
    type: "Medical Emergency",
    location: "Salt Lake City Centre",
    coordinates: { lat: 22.5741, lng: 88.4300 },
    priority: "medium",
    patientInfo: "Child, 8y, high fever, vomiting",
    distance: "4.5 km",
    eta: "12 min",
    timestamp: "8 min ago",
  },
]

const MOCK_HOSPITALS = [
  { name: "AMRI Hospital", location: "Southern Avenue", distance: "3.2 km" },
  { name: "Apollo Gleneagles", location: "Canal Circular Road", distance: "2.8 km" },
  { name: "Fortis Hospital", location: "Anandapur", distance: "5.1 km" },
]

export default function AmbulanceApp({ userName, onLogout }: AmbulanceAppProps) {
  const [incomingRequests, setIncomingRequests] = useState<IncomingRequest[]>(MOCK_INCOMING_REQUESTS)
  const [currentAssignment, setCurrentAssignment] = useState<Assignment | null>(null)
  const [selectedHospital, setSelectedHospital] = useState<string>("")
  const [vitals, setVitals] = useState<VitalSign>({
    heartRate: "",
    bloodPressure: "",
    temperature: "",
    oxygenSaturation: "",
    respiratoryRate: "",
    notes: "",
  })
  const [vitalsUploaded, setVitalsUploaded] = useState(false)

  const priorityBadge = (priority: string) => {
    const badges: Record<string, string> = {
      critical: "bg-red-500 text-white",
      high: "bg-orange-500 text-white",
      medium: "bg-yellow-500 text-white",
    }
    return badges[priority] || "bg-gray-500 text-white"
  }

  const acceptRequest = (request: IncomingRequest) => {
    const newAssignment: Assignment = {
      id: request.id,
      type: request.type,
      patientLocation: request.location,
      patientCoordinates: request.coordinates,
      priority: request.priority,
      patientInfo: request.patientInfo,
      distance: request.distance,
      eta: request.eta,
      status: "en-route-to-patient",
    }
    setCurrentAssignment(newAssignment)
    setIncomingRequests(incomingRequests.filter((r) => r.id !== request.id))
    setVitalsUploaded(false)
    setVitals({
      heartRate: "",
      bloodPressure: "",
      temperature: "",
      oxygenSaturation: "",
      respiratoryRate: "",
      notes: "",
    })
  }

  const rejectRequest = (requestId: string) => {
    setIncomingRequests(incomingRequests.filter((r) => r.id !== requestId))
  }

  const updateAssignmentStatus = (newStatus: Assignment["status"]) => {
    if (currentAssignment) {
      setCurrentAssignment({ ...currentAssignment, status: newStatus })
    }
  }

  const selectHospital = (hospitalName: string) => {
    setSelectedHospital(hospitalName)
    if (currentAssignment) {
      const hospital = MOCK_HOSPITALS.find((h) => h.name === hospitalName)
      if (hospital) {
        setCurrentAssignment({
          ...currentAssignment,
          hospitalName: hospital.name,
          hospitalLocation: hospital.location,
          hospitalCoordinates: { lat: 22.5262, lng: 88.3389 }, // Mock coordinates
        })
      }
    }
  }

  const uploadVitals = () => {
    // Simulate upload
    setVitalsUploaded(true)
    setTimeout(() => {
      alert("Patient vitals successfully sent to hospital and doctors!")
    }, 500)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-red-50 to-orange-50">
      <Header title="Ambulance Crew App" userName={userName} onLogout={onLogout} />

      <main className="p-4 max-w-6xl mx-auto">
        <Tabs defaultValue="incoming" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="incoming" className="text-base">
              📍 Incoming Requests
              {incomingRequests.length > 0 && (
                <Badge className="ml-2 bg-red-500">{incomingRequests.length}</Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="current" className="text-base">
              🚑 Current Assignment
            </TabsTrigger>
          </TabsList>

          {/* INCOMING REQUESTS TAB */}
          <TabsContent value="incoming" className="space-y-4">
            {/* Map Placeholder */}
            <Card className="border-2 border-blue-300">
              <CardHeader className="bg-blue-50">
                <CardTitle className="flex items-center gap-2 text-blue-900">
                  <MapPin className="h-5 w-5" />
                  Live Map - Your Location & Incoming Requests
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-64 bg-gradient-to-br from-blue-100 to-green-100 relative flex items-center justify-center">
                  <div className="absolute top-4 left-4 bg-white px-3 py-2 rounded-lg shadow-md">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-semibold">Your Ambulance</span>
                    </div>
                  </div>
                  {incomingRequests.map((req, idx) => (
                    <div
                      key={req.id}
                      className="absolute bg-red-500 text-white px-2 py-1 rounded-full text-xs font-bold shadow-lg"
                      style={{
                        top: `${20 + idx * 25}%`,
                        left: `${30 + idx * 20}%`,
                      }}
                    >
                      🚨 {req.type.split(" ")[0]}
                    </div>
                  ))}
                  <div className="text-center text-gray-600">
                    <MapPin className="h-16 w-16 mx-auto mb-2 text-blue-500" />
                    <p className="text-sm">Map integration placeholder</p>
                    <p className="text-xs text-gray-500">Shows your location and emergency requests</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Incoming Requests List */}
            <div className="space-y-3">
              {incomingRequests.length === 0 ? (
                <Card>
                  <CardContent className="py-12 text-center text-gray-500">
                    <Clock className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                    <p className="text-lg font-semibold">No incoming requests</p>
                    <p className="text-sm">Waiting for emergency calls...</p>
                  </CardContent>
                </Card>
              ) : (
                incomingRequests.map((request) => (
                  <Card key={request.id} className="border-2 border-orange-300 shadow-lg">
                    <CardHeader className="bg-gradient-to-r from-orange-50 to-red-50">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <CardTitle className="text-lg text-orange-900">{request.type}</CardTitle>
                            <Badge className={priorityBadge(request.priority)}>
                              {request.priority.toUpperCase()}
                            </Badge>
                          </div>
                          <p className="text-sm text-gray-600">{request.timestamp}</p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4 pt-4">
                      <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                        <div className="flex items-start gap-2">
                          <MapPin className="h-5 w-5 text-blue-600 mt-0.5" />
                          <div>
                            <p className="font-semibold text-blue-900">{request.location}</p>
                            <p className="text-sm text-blue-700">
                              {request.distance} away • ETA: {request.eta}
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                        <p className="text-sm font-semibold text-yellow-900 mb-1">Patient Info:</p>
                        <p className="text-sm text-yellow-800">{request.patientInfo}</p>
                      </div>

                      <div className="flex gap-3">
                        <Button
                          onClick={() => acceptRequest(request)}
                          className="flex-1 bg-green-600 hover:bg-green-700 text-white"
                        >
                          <Check className="mr-2 h-4 w-4" />
                          Accept Request
                        </Button>
                        <Button
                          onClick={() => rejectRequest(request.id)}
                          variant="outline"
                          className="flex-1 border-red-300 text-red-600 hover:bg-red-50"
                        >
                          <X className="mr-2 h-4 w-4" />
                          Decline
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </TabsContent>

          {/* CURRENT ASSIGNMENT TAB */}
          <TabsContent value="current" className="space-y-4">
            {!currentAssignment ? (
              <Card>
                <CardContent className="py-12 text-center text-gray-500">
                  <Navigation className="h-12 w-12 mx-auto mb-3 text-gray-400" />
                  <p className="text-lg font-semibold">No active assignment</p>
                  <p className="text-sm">Accept a request to start</p>
                </CardContent>
              </Card>
            ) : (
              <>
                {/* Status Banner */}
                <Card className="border-2 border-blue-400 bg-gradient-to-r from-blue-50 to-indigo-50">
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                        <div>
                          <p className="text-sm text-gray-600">Current Status</p>
                          <p className="text-lg font-bold text-blue-900">
                            {currentAssignment.status === "en-route-to-patient" && "En Route to Patient"}
                            {currentAssignment.status === "on-scene" && "On Scene"}
                            {currentAssignment.status === "patient-loaded" && "Patient Loaded"}
                            {currentAssignment.status === "en-route-to-hospital" && "En Route to Hospital"}
                            {currentAssignment.status === "delivered" && "Patient Delivered"}
                          </p>
                        </div>
                      </div>
                      <Badge className={priorityBadge(currentAssignment.priority)} className="text-base px-4 py-2">
                        {currentAssignment.priority.toUpperCase()}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                {/* Route Map */}
                <Card className="border-2 border-green-300">
                  <CardHeader className="bg-green-50">
                    <CardTitle className="flex items-center gap-2 text-green-900">
                      <Navigation className="h-5 w-5" />
                      {currentAssignment.status === "en-route-to-patient" && "Route to Patient"}
                      {currentAssignment.status === "on-scene" && "At Patient Location"}
                      {(currentAssignment.status === "patient-loaded" ||
                        currentAssignment.status === "en-route-to-hospital") &&
                        "Route to Hospital"}
                      {currentAssignment.status === "delivered" && "Mission Completed"}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <div className="h-72 bg-gradient-to-br from-green-100 via-blue-100 to-purple-100 relative flex items-center justify-center">
                      {/* Route visualization */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="relative w-4/5 h-4/5">
                          {/* Start point (Ambulance) */}
                          <div className="absolute bottom-4 left-4 bg-green-500 text-white px-3 py-2 rounded-lg shadow-lg font-bold">
                            🚑 You
                          </div>
                          
                          {/* Patient location */}
                          {(currentAssignment.status === "en-route-to-patient" ||
                            currentAssignment.status === "on-scene") && (
                            <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-2 rounded-lg shadow-lg font-bold animate-pulse">
                              🏥 Patient
                            </div>
                          )}
                          
                          {/* Hospital location */}
                          {(currentAssignment.status === "patient-loaded" ||
                            currentAssignment.status === "en-route-to-hospital" ||
                            currentAssignment.status === "delivered") && (
                            <div className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-2 rounded-lg shadow-lg font-bold">
                              🏥 Hospital
                            </div>
                          )}

                          {/* Route line */}
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="border-t-4 border-dashed border-blue-500 w-3/4 transform rotate-45"></div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-center text-gray-600 z-10">
                        <Navigation className="h-12 w-12 mx-auto mb-2 text-green-600" />
                        <p className="text-sm font-semibold">GPS Navigation Active</p>
                        <p className="text-xs text-gray-500">ETA: {currentAssignment.eta}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Assignment Details */}
                <Card className="border-2 border-orange-300">
                  <CardHeader className="bg-orange-50">
                    <CardTitle className="text-orange-900">Assignment Details</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-4">
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Emergency Type</p>
                      <p className="text-lg font-bold text-gray-900">{currentAssignment.type}</p>
                    </div>

                    <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                      <p className="text-sm font-semibold text-blue-900 mb-1">Patient Location</p>
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4 text-blue-600" />
                        <p className="text-sm text-blue-800">{currentAssignment.patientLocation}</p>
                      </div>
                    </div>

                    {currentAssignment.hospitalName && (
                      <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                        <p className="text-sm font-semibold text-green-900 mb-1">Destination Hospital</p>
                        <p className="text-base font-bold text-green-800">{currentAssignment.hospitalName}</p>
                        <p className="text-sm text-green-700">{currentAssignment.hospitalLocation}</p>
                      </div>
                    )}

                    <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                      <p className="text-sm font-semibold text-yellow-900 mb-1">Patient Information</p>
                      <p className="text-sm text-yellow-800">{currentAssignment.patientInfo}</p>
                    </div>
                  </CardContent>
                </Card>

                {/* Hospital Selection */}
                {currentAssignment.status === "on-scene" && !currentAssignment.hospitalName && (
                  <Card className="border-2 border-purple-300">
                    <CardHeader className="bg-purple-50">
                      <CardTitle className="text-purple-900">Select Destination Hospital</CardTitle>
                    </CardHeader>
                    <CardContent className="pt-4">
                      <div className="space-y-2">
                        {MOCK_HOSPITALS.map((hospital) => (
                          <button
                            key={hospital.name}
                            onClick={() => selectHospital(hospital.name)}
                            className={`w-full p-4 rounded-lg border-2 text-left transition-all ${
                              selectedHospital === hospital.name
                                ? "border-purple-500 bg-purple-50"
                                : "border-gray-200 hover:border-purple-300 bg-white"
                            }`}
                          >
                            <p className="font-bold text-gray-900">{hospital.name}</p>
                            <p className="text-sm text-gray-600">{hospital.location}</p>
                            <p className="text-xs text-purple-600 mt-1">{hospital.distance} away</p>
                          </button>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Patient Vitals Upload */}
                {(currentAssignment.status === "on-scene" ||
                  currentAssignment.status === "patient-loaded" ||
                  currentAssignment.status === "en-route-to-hospital") && (
                  <Card className="border-2 border-indigo-300">
                    <CardHeader className="bg-indigo-50">
                      <CardTitle className="flex items-center gap-2 text-indigo-900">
                        <Activity className="h-5 w-5" />
                        Patient Health Vitals
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4 pt-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="heartRate" className="flex items-center gap-2 mb-2">
                            <Heart className="h-4 w-4 text-red-500" />
                            Heart Rate (BPM)
                          </Label>
                          <Input
                            id="heartRate"
                            type="number"
                            placeholder="e.g., 75"
                            value={vitals.heartRate}
                            onChange={(e) => setVitals({ ...vitals, heartRate: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>

                        <div>
                          <Label htmlFor="bloodPressure" className="flex items-center gap-2 mb-2">
                            <Activity className="h-4 w-4 text-blue-500" />
                            Blood Pressure (mmHg)
                          </Label>
                          <Input
                            id="bloodPressure"
                            placeholder="e.g., 120/80"
                            value={vitals.bloodPressure}
                            onChange={(e) => setVitals({ ...vitals, bloodPressure: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>

                        <div>
                          <Label htmlFor="temperature" className="flex items-center gap-2 mb-2">
                            <Thermometer className="h-4 w-4 text-orange-500" />
                            Temperature (°F)
                          </Label>
                          <Input
                            id="temperature"
                            type="number"
                            step="0.1"
                            placeholder="e.g., 98.6"
                            value={vitals.temperature}
                            onChange={(e) => setVitals({ ...vitals, temperature: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>

                        <div>
                          <Label htmlFor="oxygenSaturation" className="flex items-center gap-2 mb-2">
                            <Wind className="h-4 w-4 text-cyan-500" />
                            Oxygen Saturation (%)
                          </Label>
                          <Input
                            id="oxygenSaturation"
                            type="number"
                            placeholder="e.g., 98"
                            value={vitals.oxygenSaturation}
                            onChange={(e) => setVitals({ ...vitals, oxygenSaturation: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>

                        <div>
                          <Label htmlFor="respiratoryRate" className="flex items-center gap-2 mb-2">
                            <Activity className="h-4 w-4 text-green-500" />
                            Respiratory Rate (breaths/min)
                          </Label>
                          <Input
                            id="respiratoryRate"
                            type="number"
                            placeholder="e.g., 16"
                            value={vitals.respiratoryRate}
                            onChange={(e) => setVitals({ ...vitals, respiratoryRate: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>

                        <div className="md:col-span-2">
                          <Label htmlFor="notes" className="mb-2">
                            Additional Notes
                          </Label>
                          <Input
                            id="notes"
                            placeholder="Any additional observations..."
                            value={vitals.notes}
                            onChange={(e) => setVitals({ ...vitals, notes: e.target.value })}
                            disabled={vitalsUploaded}
                          />
                        </div>
                      </div>

                      {vitalsUploaded ? (
                        <div className="bg-green-100 border border-green-400 text-green-800 px-4 py-3 rounded-lg flex items-center gap-2">
                          <Check className="h-5 w-5" />
                          <p className="font-semibold">Vitals uploaded and sent to hospital & doctors!</p>
                        </div>
                      ) : (
                        <Button
                          onClick={uploadVitals}
                          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white"
                          disabled={!vitals.heartRate && !vitals.bloodPressure}
                        >
                          <Upload className="mr-2 h-4 w-4" />
                          Upload & Send Vitals to Hospital & Doctors
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                )}

                {/* Action Buttons */}
                <Card className="border-2 border-blue-300">
                  <CardContent className="pt-6">
                    <div className="space-y-3">
                      {currentAssignment.status === "en-route-to-patient" && (
                        <Button
                          onClick={() => updateAssignmentStatus("on-scene")}
                          className="w-full bg-orange-600 hover:bg-orange-700 text-white text-lg py-6"
                        >
                          ✓ Arrived at Patient Location
                        </Button>
                      )}

                      {currentAssignment.status === "on-scene" && currentAssignment.hospitalName && (
                        <Button
                          onClick={() => updateAssignmentStatus("patient-loaded")}
                          className="w-full bg-blue-600 hover:bg-blue-700 text-white text-lg py-6"
                        >
                          ✓ Patient Loaded - Start to Hospital
                        </Button>
                      )}

                      {currentAssignment.status === "patient-loaded" && (
                        <Button
                          onClick={() => updateAssignmentStatus("en-route-to-hospital")}
                          className="w-full bg-purple-600 hover:bg-purple-700 text-white text-lg py-6"
                        >
                          🚑 En Route to Hospital
                        </Button>
                      )}

                      {currentAssignment.status === "en-route-to-hospital" && (
                        <Button
                          onClick={() => updateAssignmentStatus("delivered")}
                          className="w-full bg-green-600 hover:bg-green-700 text-white text-lg py-6"
                        >
                          ✓ Patient Delivered to Hospital
                        </Button>
                      )}

                      {currentAssignment.status === "delivered" && (
                        <div className="space-y-3">
                          <div className="bg-green-100 border-2 border-green-400 text-green-900 px-4 py-6 rounded-lg text-center">
                            <p className="text-2xl mb-2">🎉</p>
                            <p className="text-xl font-bold">Mission Completed!</p>
                            <p className="text-sm mt-1">Patient successfully delivered to hospital</p>
                          </div>
                          <Button
                            onClick={() => setCurrentAssignment(null)}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                          >
                            Return to Incoming Requests
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
