"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Header from "@/components/shared/header"

interface Patient {
  id: string
  name: string
  condition: string
  severity: "critical" | "high" | "medium"
  room: string
  doctor: string
  admitted: string
  status: "awaiting" | "in-treatment" | "stable" | "discharged"
}

interface HospitalConsoleProps {
  userName: string
  onLogout: () => void
}

const MOCK_PATIENTS: Patient[] = [
  {
    id: "1",
    name: "Michael Brown",
    condition: "Chest Pain",
    severity: "critical",
    room: "ICU-01",
    doctor: "Dr. Anderson",
    admitted: "30 min ago",
    status: "in-treatment",
  },
  {
    id: "2",
    name: "Sarah Johnson",
    condition: "Leg Fracture",
    severity: "medium",
    room: "OR-02",
    doctor: "Dr. Smith",
    admitted: "1 hour ago",
    status: "in-treatment",
  },
  {
    id: "3",
    name: "David Lee",
    condition: "Head Injury",
    severity: "high",
    room: "ICU-03",
    doctor: "Dr. Mitchell",
    admitted: "45 min ago",
    status: "awaiting",
  },
]

export default function HospitalConsole({ userName, onLogout }: HospitalConsoleProps) {
  const [patients, setPatients] = useState<Patient[]>(MOCK_PATIENTS)

  const severityBadge = (severity: string) => {
    const badges: Record<string, string> = {
      critical: "bg-red-200 text-red-800",
      high: "bg-orange-200 text-orange-800",
      medium: "bg-amber-200 text-amber-800",
    }
    return badges[severity] || "bg-stone-200"
  }

  const statusBadge = (status: string) => {
    const badges: Record<string, string> = {
      awaiting: "bg-blue-100 text-blue-700",
      "in-treatment": "bg-amber-100 text-amber-700",
      stable: "bg-green-100 text-green-700",
      discharged: "bg-stone-100 text-stone-700",
    }
    return badges[status] || "bg-stone-100"
  }

  const updateStatus = (patientId: string, newStatus: string) => {
    setPatients(
      patients.map((p) =>
        p.id === patientId
          ? {
              ...p,
              status: newStatus as Patient["status"],
            }
          : p,
      ),
    )
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <Header title="Hospital Console" userName={userName} onLogout={onLogout} />

      <main className="p-6 max-w-7xl mx-auto">
        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-red-600">{patients.length}</div>
              <div className="text-orange-700 text-sm">Total Patients</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-orange-600">
                {patients.filter((p) => p.severity === "critical").length}
              </div>
              <div className="text-orange-700 text-sm">Critical</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-blue-600">2</div>
              <div className="text-orange-700 text-sm">Waiting Rooms</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-green-600">87%</div>
              <div className="text-orange-700 text-sm">Bed Occupancy</div>
            </CardContent>
          </Card>
        </div>

        {/* Patients List */}
        <Card className="bg-white border-orange-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-orange-900">Active Patients</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {patients.map((patient) => (
                <div
                  key={patient.id}
                  className="p-4 rounded-lg border border-orange-100 bg-gradient-to-r from-orange-50 to-amber-50"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-orange-900">{patient.name}</h3>
                      <p className="text-sm text-orange-700">{patient.condition}</p>
                    </div>
                    <div className="flex gap-2">
                      <span className={`text-xs px-2 py-1 rounded ${severityBadge(patient.severity)}`}>
                        {patient.severity.toUpperCase()}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${statusBadge(patient.status)}`}>
                        {patient.status.toUpperCase().replace("-", " ")}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3 text-sm">
                    <div>
                      <div className="text-orange-600">Room</div>
                      <div className="text-orange-900 font-semibold">{patient.room}</div>
                    </div>
                    <div>
                      <div className="text-orange-600">Doctor</div>
                      <div className="text-orange-900 font-semibold">{patient.doctor}</div>
                    </div>
                    <div>
                      <div className="text-orange-600">Admitted</div>
                      <div className="text-orange-900 font-semibold">{patient.admitted}</div>
                    </div>
                    <div className="text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-orange-300 text-orange-700 hover:bg-orange-50 bg-transparent"
                      >
                        View Details
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
