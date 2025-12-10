"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Header from "@/components/shared/header"

interface Report {
  id: string
  type: string
  description: string
  status: "submitted" | "in-progress" | "resolved"
  created: string
}

interface CitizenPortalProps {
  userName: string
  onLogout: () => void
}

export default function CitizenPortal({ userName, onLogout }: CitizenPortalProps) {
  const [reports, setReports] = useState<Report[]>([
    {
      id: "1",
      type: "Medical Emergency",
      description: "Person collapsed at shopping center",
      status: "in-progress",
      created: "2 hours ago",
    },
    {
      id: "2",
      type: "Fire Alarm",
      description: "Smoke detected in office building",
      status: "resolved",
      created: "5 hours ago",
    },
  ])

  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({
    type: "medical",
    description: "",
    location: "",
  })

  const handleSubmit = () => {
    if (formData.description && formData.location) {
      const newReport: Report = {
        id: (reports.length + 1).toString(),
        type: formData.type === "medical" ? "Medical Emergency" : "Other",
        description: formData.description,
        status: "submitted",
        created: "just now",
      }
      setReports([newReport, ...reports])
      setFormData({ type: "medical", description: "", location: "" })
      setShowForm(false)
    }
  }

  const statusColor = (status: string) => {
    const colors: Record<string, string> = {
      submitted: "bg-blue-100 text-blue-800",
      "in-progress": "bg-amber-100 text-amber-800",
      resolved: "bg-green-100 text-green-800",
    }
    return colors[status] || "bg-stone-100"
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <Header title="Citizen Portal" userName={userName} onLogout={onLogout} />

      <main className="p-6 max-w-2xl mx-auto">
        {/* Emergency Button */}
        <Card className="bg-white border-red-200 shadow-lg mb-8">
          <CardContent className="pt-8">
            <Button
              onClick={() => setShowForm(!showForm)}
              className="w-full bg-red-500 hover:bg-red-600 text-white font-bold text-lg py-6 rounded-lg"
            >
              {showForm ? "Cancel" : "Report Emergency"}
            </Button>
          </CardContent>
        </Card>

        {/* Report Form */}
        {showForm && (
          <Card className="bg-white border-orange-200 shadow-md mb-8">
            <CardHeader>
              <CardTitle className="text-orange-900">New Emergency Report</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm text-orange-800 mb-2 font-medium">Emergency Type</label>
                <select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                  className="w-full bg-orange-50 text-orange-900 rounded px-3 py-2 border border-orange-200 focus:border-orange-400 focus:outline-none"
                >
                  <option value="medical">Medical Emergency</option>
                  <option value="fire">Fire</option>
                  <option value="accident">Traffic Accident</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-orange-800 mb-2 font-medium">Location</label>
                <input
                  type="text"
                  placeholder="Enter your location"
                  value={formData.location}
                  onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                  className="w-full bg-orange-50 text-orange-900 rounded px-3 py-2 border border-orange-200 focus:border-orange-400 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-orange-800 mb-2 font-medium">Description</label>
                <textarea
                  placeholder="Describe the emergency"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={4}
                  className="w-full bg-orange-50 text-orange-900 rounded px-3 py-2 border border-orange-200 focus:border-orange-400 focus:outline-none"
                />
              </div>
              <Button onClick={handleSubmit} className="w-full bg-orange-500 hover:bg-orange-600 text-white">
                Submit Report
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Report History */}
        <Card className="bg-white border-orange-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-orange-900">Your Reports</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {reports.map((report) => (
                <div
                  key={report.id}
                  className="p-4 rounded-lg border border-orange-100 bg-gradient-to-r from-orange-50 to-amber-50"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-orange-900">{report.type}</h3>
                    <span className={`text-xs px-2 py-1 rounded ${statusColor(report.status)}`}>
                      {report.status.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-sm text-orange-700 mb-2">{report.description}</p>
                  <p className="text-xs text-orange-600">{report.created}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
