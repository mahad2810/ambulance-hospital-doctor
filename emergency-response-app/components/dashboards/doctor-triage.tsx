"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Header from "@/components/shared/header"

interface TriagePatient {
  id: string
  name: string
  age: string
  chief_complaint: string
  vitals: {
    bp: string
    hr: string
    temp: string
    o2: string
  }
  priority: "resuscitation" | "emergent" | "urgent" | "non-urgent"
  assessments: string[]
  wait_time: string
}

interface DoctorTriageProps {
  userName: string
  onLogout: () => void
}

const MOCK_TRIAGE_QUEUE: TriagePatient[] = [
  {
    id: "1",
    name: "Robert Williams",
    age: "54",
    chief_complaint: "Severe chest pain, shortness of breath",
    vitals: { bp: "160/95", hr: "125", temp: "36.8°C", o2: "91%" },
    priority: "resuscitation",
    assessments: ["EKG ordered", "Troponin test pending"],
    wait_time: "5 min",
  },
  {
    id: "2",
    name: "Jennifer Davis",
    age: "31",
    chief_complaint: "Abdominal pain, nausea",
    vitals: { bp: "128/82", hr: "98", temp: "37.2°C", o2: "98%" },
    priority: "emergent",
    assessments: ["Ultrasound scheduled", "Blood work ordered"],
    wait_time: "15 min",
  },
  {
    id: "3",
    name: "Thomas Anderson",
    age: "67",
    chief_complaint: "Dizziness and weakness",
    vitals: { bp: "142/88", hr: "110", temp: "37.0°C", o2: "95%" },
    priority: "urgent",
    assessments: ["Vitals monitored"],
    wait_time: "25 min",
  },
]

export default function DoctorTriage({ userName, onLogout }: DoctorTriageProps) {
  const [queue, setQueue] = useState<TriagePatient[]>(MOCK_TRIAGE_QUEUE)
  const [selectedPatient, setSelectedPatient] = useState<string>("1")
  const [notes, setNotes] = useState<Record<string, string>>({})

  const priorityColor = (priority: string) => {
    const colors: Record<string, string> = {
      resuscitation: "bg-red-200 text-red-800",
      emergent: "bg-orange-200 text-orange-800",
      urgent: "bg-amber-200 text-amber-800",
      "non-urgent": "bg-green-200 text-green-800",
    }
    return colors[priority] || "bg-stone-200"
  }

  const moveToTreatment = (patientId: string) => {
    setQueue(queue.filter((p) => p.id !== patientId))
    if (selectedPatient === patientId && queue.length > 1) {
      const nextId = queue.find((p) => p.id !== patientId)?.id
      if (nextId) setSelectedPatient(nextId)
    }
  }

  const current = queue.find((p) => p.id === selectedPatient)

  return (
    <div className="min-h-screen bg-amber-50">
      <Header title="Doctor - Triage Interface" userName={userName} onLogout={onLogout} />

      <main className="p-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Triage Queue */}
          <Card className="bg-white border-orange-200 shadow-md">
            <CardHeader>
              <CardTitle className="text-orange-900">Triage Queue ({queue.length})</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {queue.map((patient) => (
                  <button
                    key={patient.id}
                    onClick={() => setSelectedPatient(patient.id)}
                    className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                      selectedPatient === patient.id
                        ? "border-blue-400 bg-blue-50"
                        : "border-orange-200 hover:border-blue-400 bg-gradient-to-r from-orange-50 to-amber-50"
                    }`}
                  >
                    <div className="font-semibold text-orange-900 text-sm">{patient.name}</div>
                    <div className="text-xs text-orange-700">{patient.chief_complaint}</div>
                    <div className="flex justify-between items-center mt-2">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${priorityColor(patient.priority)}`}>
                        {patient.priority.toUpperCase()}
                      </span>
                      <span className="text-xs text-orange-600">{patient.wait_time}</span>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Patient Details */}
          {current && (
            <div className="lg:col-span-2 space-y-6">
              {/* Patient Info */}
              <Card className="bg-white border-orange-200 shadow-md">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-orange-900">{current.name}</CardTitle>
                      <p className="text-sm text-orange-700">Age: {current.age}</p>
                    </div>
                    <span className={`text-xs px-3 py-1 rounded ${priorityColor(current.priority)}`}>
                      {current.priority.toUpperCase()}
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="text-sm text-orange-700 mb-1">Chief Complaint</div>
                    <p className="text-orange-900">{current.chief_complaint}</p>
                  </div>

                  {/* Vitals */}
                  <div className="bg-gradient-to-r from-orange-50 to-amber-50 p-4 rounded-lg border border-orange-200">
                    <div className="text-sm text-orange-700 mb-3">Vitals</div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div>
                        <div className="text-xs text-orange-600">BP</div>
                        <div className="font-semibold text-orange-900">{current.vitals.bp}</div>
                      </div>
                      <div>
                        <div className="text-xs text-orange-600">HR</div>
                        <div className="font-semibold text-orange-900">{current.vitals.hr}</div>
                      </div>
                      <div>
                        <div className="text-xs text-orange-600">Temp</div>
                        <div className="font-semibold text-orange-900">{current.vitals.temp}</div>
                      </div>
                      <div>
                        <div className="text-xs text-orange-600">O2</div>
                        <div className="font-semibold text-orange-900">{current.vitals.o2}</div>
                      </div>
                    </div>
                  </div>

                  {/* Assessments */}
                  <div>
                    <div className="text-sm text-orange-700 mb-2">Assessments & Orders</div>
                    <div className="space-y-1">
                      {current.assessments.map((assessment, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm text-orange-800">
                          <span className="text-green-600">✓</span> {assessment}
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Clinical Notes */}
              <Card className="bg-white border-orange-200 shadow-md">
                <CardHeader>
                  <CardTitle className="text-orange-900">Clinical Notes</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <textarea
                    placeholder="Add clinical notes and observations..."
                    value={notes[current.id] || ""}
                    onChange={(e) =>
                      setNotes({
                        ...notes,
                        [current.id]: e.target.value,
                      })
                    }
                    rows={5}
                    className="w-full bg-orange-50 text-orange-900 rounded px-3 py-2 border border-orange-200 focus:border-orange-400 focus:outline-none text-sm"
                  />
                  <Button
                    onClick={() => moveToTreatment(current.id)}
                    className="w-full bg-green-500 hover:bg-green-600 text-white"
                  >
                    Move to Treatment
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
