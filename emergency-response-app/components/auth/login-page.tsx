"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const DEMO_USERS = [
  { role: "admin", name: "Sarah Chen", title: "Emergency Coordinator" },
  { role: "citizen", name: "John Doe", title: "Report Emergency" },
  { role: "ambulance", name: "Alex Rivera", title: "Ambulance Crew" },
  { role: "hospital", name: "Dr. Patricia Wong", title: "Hospital Staff" },
  { role: "doctor", name: "Dr. James Mitchell", title: "Emergency Doctor" },
]

interface LoginPageProps {
  onLogin: (role: string, name: string) => void
}

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [selectedRole, setSelectedRole] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-red-50 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-orange-600 mb-2">SmartAgent ER</h1>
          <p className="text-orange-800 text-lg">Emergency Response Coordination System</p>
        </div>

        <Card className="bg-white border-orange-200 shadow-lg">
          <CardHeader>
            <CardTitle className="text-center text-orange-900">Select Your Role</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {DEMO_USERS.map((user) => (
                <button
                  key={user.role}
                  onClick={() => {
                    setSelectedRole(user.role)
                    setTimeout(() => onLogin(user.role, user.name), 300)
                  }}
                  className={`p-4 rounded-lg border-2 transition-all duration-300 text-left ${
                    selectedRole === user.role
                      ? "border-orange-500 bg-orange-100"
                      : "border-orange-200 hover:border-orange-400 bg-amber-50"
                  }`}
                >
                  <div className="font-semibold text-orange-900">{user.name}</div>
                  <div className="text-sm text-orange-700">{user.title}</div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <p className="text-center text-orange-600 text-sm mt-6">Demo Mode: Click any role to access that dashboard</p>
      </div>
    </div>
  )
}
