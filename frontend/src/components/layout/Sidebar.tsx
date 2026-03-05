import { NavLink } from "react-router-dom"
import { cn } from "../../lib/utils"
import { useAppStore } from "../../stores/appStore"
import {
  LayoutDashboard,
  LineChart,
  Brain,
  GitCompare,
  TrendingUp,
  History,
  Dices,
  Book,
  GraduationCap,
  BarChart3,
  Database,
  Atom,
  Bot,
  Settings,
  ChevronLeft,
  ChevronRight,
  HardDrive,
  Layers,
  FolderSync,
  Activity,
  FileText,
  Waves,
} from "lucide-react"
import { Button } from "../ui/button"

const navGroups = [
  {
    label: "Overview",
    items: [
      { name: "Dashboard", href: "/", icon: LayoutDashboard },
      { name: "Data Management", href: "/data-management", icon: FolderSync },
    ],
  },
  {
    label: "Models & Training",
    items: [
      { name: "Model Comparison", href: "/models", icon: GitCompare },
      { name: "Training", href: "/training", icon: GraduationCap },
      { name: "Batch Training", href: "/batch-training", icon: Layers },
      { name: "Model Manager", href: "/model-manager", icon: HardDrive },
      { name: "Predictions", href: "/predictions", icon: TrendingUp },
    ],
  },
  {
    label: "Analysis",
    items: [
      { name: "PINN Analysis", href: "/pinn", icon: Brain },
      { name: "Volatility", href: "/volatility", icon: Activity },
      { name: "Spectral", href: "/spectral", icon: Waves },
      { name: "Dissertation", href: "/dissertation", icon: FileText },
      { name: "Comprehensive", href: "/comprehensive", icon: BarChart3 },
      { name: "Monte Carlo", href: "/monte-carlo", icon: Dices },
      { name: "Backtesting", href: "/backtesting", icon: History },
      { name: "Leaderboard", href: "/leaderboard", icon: BarChart3 },
    ],
  },
  {
    label: "Data & System",
    items: [
      { name: "Data Explorer", href: "/data", icon: Database },
      { name: "Physics Params", href: "/physics", icon: Atom },
      { name: "Trading Agent", href: "/trading", icon: Bot },
      { name: "Methodology", href: "/methodology", icon: Book },
      { name: "Settings", href: "/settings", icon: Settings },
    ],
  },
]

export function Sidebar() {
  const { sidebarOpen, toggleSidebar } = useAppStore()

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 h-screen transition-all duration-300 flex flex-col",
        "bg-gradient-to-b from-card via-card to-card/80 border-r border-border/50",
        "dark:from-[hsl(222,47%,7%)] dark:via-[hsl(222,40%,8%)] dark:to-[hsl(222,40%,6%)]",
        sidebarOpen ? "w-64" : "w-16"
      )}
    >
      {/* Header */}
      <div className="flex h-16 items-center justify-between border-b border-border/50 px-4 flex-shrink-0">
        {sidebarOpen && (
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
              <Brain className="h-5 w-5 text-primary" />
            </div>
            <span className="font-semibold tracking-tight bg-gradient-to-r from-primary to-cyan-400 bg-clip-text text-transparent">
              PINN Finance
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className={cn(
            "h-8 w-8 rounded-lg hover:bg-accent/80",
            !sidebarOpen && "mx-auto"
          )}
        >
          {sidebarOpen ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden py-3 px-2 [&::-webkit-scrollbar]:w-0">
        {navGroups.map((group) => (
          <div key={group.label} className="mb-1">
            {sidebarOpen && (
              <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                {group.label}
              </div>
            )}
            {!sidebarOpen && group.label !== "Overview" && (
              <div className="mx-auto my-2 h-px w-6 bg-border/50" />
            )}
            <div className="space-y-0.5">
              {group.items.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200",
                      "hover:bg-accent/80",
                      isActive
                        ? "bg-primary/10 text-primary border-l-2 border-primary shadow-sm shadow-primary/5"
                        : "text-muted-foreground hover:text-foreground border-l-2 border-transparent",
                      !sidebarOpen && "justify-center px-2 border-l-0"
                    )
                  }
                  title={!sidebarOpen ? item.name : undefined}
                >
                  <item.icon
                    className={cn(
                      "h-[18px] w-[18px] flex-shrink-0 transition-transform duration-200",
                      "group-hover:scale-110"
                    )}
                  />
                  {sidebarOpen && <span className="truncate">{item.name}</span>}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Sidebar footer */}
      {sidebarOpen && (
        <div className="flex-shrink-0 border-t border-border/50 p-4">
          <div className="text-[10px] text-muted-foreground/50 text-center">
            PINN Finance v1.0
          </div>
        </div>
      )}
    </aside>
  )
}
