"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Header from "@/components/shared/header"

interface Incident {
  id: string
  type: string
  location: string
  severity: "critical" | "high" | "medium" | "low"
  reported: string
  responders: number
  status: "active" | "resolved"
}

interface Resource {
  id: string
  type: string
  available: number
  total: number
  status: "operational" | "limited"
}

const MOCK_INCIDENTS: Incident[] = [
  {
    id: "1",
    type: "Traffic Accident",
    location: "5th Ave & Main St",
    severity: "critical",
    reported: "5 min ago",
    responders: 4,
    status: "active",
  },
  {
    id: "2",
    type: "Medical Emergency",
    location: "Downtown Plaza",
    severity: "high",
    reported: "12 min ago",
    responders: 2,
    status: "active",
  },
  {
    id: "3",
    type: "Fire Alarm",
    location: "Central Business District",
    severity: "medium",
    reported: "23 min ago",
    responders: 1,
    status: "active",
  },
]

const MOCK_RESOURCES: Resource[] = [
  { id: "1", type: "Ambulances", available: 8, total: 12, status: "operational" },
  { id: "2", type: "Fire Trucks", available: 3, total: 5, status: "limited" },
  { id: "3", type: "Police Units", available: 12, total: 15, status: "operational" },
]

interface AdminDashboardProps {
  userName: string
  onLogout: () => void
}

export default function AdminDashboard({ userName, onLogout }: AdminDashboardProps) {
  const [incidents, setIncidents] = useState<Incident[]>(MOCK_INCIDENTS)

  const severityBadge = (severity: string) => {
    const badges: Record<string, string> = {
      critical: "bg-red-200 text-red-800",
      high: "bg-orange-200 text-orange-800",
      medium: "bg-amber-200 text-amber-800",
      low: "bg-green-200 text-green-800",
    }
    return badges[severity] || "bg-stone-200"
  }

  const statusBadge = (status: string) => {
    return status === "active" ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <Header title="Admin Dashboard" userName={userName} onLogout={onLogout} />

      <main className="p-6 max-w-7xl mx-auto">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-red-600">{incidents.length}</div>
              <div className="text-orange-700 text-sm">Active Incidents</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-orange-600">23</div>
              <div className="text-orange-700 text-sm">Total Responders</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-green-600">8</div>
              <div className="text-orange-700 text-sm">Available Units</div>
            </CardContent>
          </Card>
          <Card className="bg-white border-orange-200 shadow-md">
            <CardContent className="pt-6">
              <div className="text-3xl font-bold text-amber-600">94%</div>
              <div className="text-orange-700 text-sm">System Status</div>
            </CardContent>
          </Card>
        </div>

        {/* Active Incidents */}
        <div className="mb-8">
          <Card className="bg-white border-orange-200 shadow-md">
            <CardHeader>
              <CardTitle className="text-orange-900">Active Incidents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {incidents.map((incident) => (
                  <div
                    key={incident.id}
                    className="p-4 rounded-lg border border-orange-200 bg-gradient-to-r from-orange-50 to-amber-50"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-semibold text-orange-900">{incident.type}</h3>
                          <span className={`text-xs px-2 py-1 rounded ${severityBadge(incident.severity)}`}>
                            {incident.severity.toUpperCase()}
                          </span>
                          <span className={`text-xs px-2 py-1 rounded ${statusBadge(incident.status)}`}>
                            {incident.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-sm text-orange-700">📍 {incident.location}</div>
                        <div className="text-xs text-orange-600 mt-1">
                          Reported: {incident.reported} • Responders: {incident.responders}
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        className="ml-4 bg-white border-orange-300 text-orange-700 hover:bg-orange-50"
                      >
                        Details
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Resource Status */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {MOCK_RESOURCES.map((resource) => (
            <Card key={resource.id} className="bg-white border-orange-200 shadow-md">
              <CardHeader>
                <CardTitle className="text-orange-900 text-lg">{resource.type}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end justify-between mb-4">
                  <div>
                    <div className="text-3xl font-bold text-orange-600">{resource.available}</div>
                    <div className="text-xs text-orange-700">of {resource.total} available</div>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      resource.status === "operational"
                        ? "bg-green-100 text-green-700"
                        : "bg-yellow-100 text-yellow-700"
                    }`}
                  >
                    {resource.status.toUpperCase()}
                  </span>
                </div>
                <div className="w-full bg-orange-100 rounded-full h-2">
                  <div
                    className="bg-orange-500 h-2 rounded-full"
                    style={{ width: `${(resource.available / resource.total) * 100}%` }}
                  />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  )
}
