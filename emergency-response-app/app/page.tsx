"use client"

import { useState } from "react"
import LoginPage from "@/components/auth/login-page"
import AdminDashboard from "@/components/dashboards/admin-dashboard"
import CitizenPortal from "@/components/dashboards/citizen-portal"
import AmbulanceApp from "@/components/dashboards/ambulance-app"
import HospitalConsole from "@/components/dashboards/hospital-console"
import DoctorTriage from "@/components/dashboards/doctor-triage"

export default function Home() {
  const [currentUser, setCurrentUser] = useState<{
    role: string
    name: string
  } | null>(null)

  const handleLogin = (role: string, name: string) => {
    setCurrentUser({ role, name })
  }

  const handleLogout = () => {
    setCurrentUser(null)
  }

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} />
  }

  const renderDashboard = () => {
    switch (currentUser.role) {
      case "admin":
        return <AdminDashboard userName={currentUser.name} onLogout={handleLogout} />
      case "citizen":
        return <CitizenPortal userName={currentUser.name} onLogout={handleLogout} />
      case "ambulance":
        return <AmbulanceApp userName={currentUser.name} onLogout={handleLogout} />
      case "hospital":
        return <HospitalConsole userName={currentUser.name} onLogout={handleLogout} />
      case "doctor":
        return <DoctorTriage userName={currentUser.name} onLogout={handleLogout} />
      default:
        return <LoginPage onLogin={handleLogin} />
    }
  }

  return renderDashboard()
}
