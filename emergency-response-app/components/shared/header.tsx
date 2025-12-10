"use client"

import { Button } from "@/components/ui/button"

interface HeaderProps {
  title: string
  userName: string
  onLogout: () => void
}

export default function Header({ title, userName, onLogout }: HeaderProps) {
  return (
    <header className="bg-gradient-to-r from-orange-100 to-amber-100 border-b border-orange-200 px-6 py-4 shadow-sm">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-orange-900">{title}</h1>
          <p className="text-sm text-orange-700">Logged in as {userName}</p>
        </div>
        <Button
          onClick={onLogout}
          variant="outline"
          className="text-orange-700 border-orange-300 hover:bg-orange-50 bg-white"
        >
          Logout
        </Button>
      </div>
    </header>
  )
}
