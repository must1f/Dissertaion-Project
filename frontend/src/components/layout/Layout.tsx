import { Outlet } from "react-router-dom"
import { Sidebar } from "./Sidebar"
import { Header } from "./Header"
import { useAppStore } from "../../stores/appStore"
import { cn } from "../../lib/utils"

export function Layout() {
  const { sidebarOpen } = useAppStore()

  return (
    <div className="min-h-screen bg-background relative">
      {/* Subtle ambient gradient */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-primary/5 blur-3xl" />
      </div>

      <Sidebar />

      <div
        className={cn(
          "transition-all duration-300",
          sidebarOpen ? "ml-64" : "ml-16"
        )}
      >
        <Header />

        <main className="p-6 animate-fade-in">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
