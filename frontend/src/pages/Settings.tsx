import { useState, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { useAppStore } from "../stores/appStore"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import { Settings as SettingsIcon, Moon, Sun, Monitor, Database, Server, Bell, CheckCircle, XCircle, AlertCircle } from "lucide-react"
import api from "../services/api"

interface AppSettings {
  apiUrl: string
  defaultDevice: string
  defaultSequenceLength: number
  mcDropoutSamples: number
  trainingNotifications: boolean
  predictionAlerts: boolean
  dataUpdateNotifications: boolean
}

const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: import.meta.env.VITE_API_URL || "http://localhost:8000",
  defaultDevice: "cpu",
  defaultSequenceLength: 60,
  mcDropoutSamples: 50,
  trainingNotifications: true,
  predictionAlerts: true,
  dataUpdateNotifications: false,
}

export default function Settings() {
  const { theme, setTheme } = useAppStore()
  const [settings, setSettings] = useState<AppSettings>(() => {
    // Load settings from localStorage
    const saved = localStorage.getItem("appSettings")
    return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS
  })
  const [isSaving, setIsSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<{ type: "success" | "error"; text: string } | null>(null)

  // Check API connection
  const { data: healthData, isLoading: healthLoading, error: healthError, refetch: refetchHealth } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const start = Date.now()
      const response = await api.get<{ status: string; app: string; version: string }>("/health")
      const latency = Date.now() - start
      return { ...response.data, latency }
    },
    refetchInterval: 30000, // Check every 30 seconds
  })

  const handleThemeChange = (newTheme: "light" | "dark" | "system") => {
    setTheme(newTheme)
    if (newTheme === "dark") {
      document.documentElement.classList.add("dark")
    } else if (newTheme === "light") {
      document.documentElement.classList.remove("dark")
    } else {
      // System preference
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        document.documentElement.classList.add("dark")
      } else {
        document.documentElement.classList.remove("dark")
      }
    }
  }

  const handleSaveSettings = () => {
    setIsSaving(true)
    setSaveMessage(null)

    try {
      localStorage.setItem("appSettings", JSON.stringify(settings))
      setSaveMessage({ type: "success", text: "Settings saved successfully!" })
    } catch (error) {
      setSaveMessage({ type: "error", text: "Failed to save settings" })
    } finally {
      setIsSaving(false)
      setTimeout(() => setSaveMessage(null), 3000)
    }
  }

  const handleResetSettings = () => {
    setSettings(DEFAULT_SETTINGS)
    localStorage.removeItem("appSettings")
    setSaveMessage({ type: "success", text: "Settings reset to defaults" })
    setTimeout(() => setSaveMessage(null), 3000)
  }

  const handleTestConnection = async () => {
    await refetchHealth()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Configure application preferences
        </p>
      </div>

      {/* Save Status Message */}
      {saveMessage && (
        <Card className={saveMessage.type === "success" ? "border-green-500" : "border-destructive"}>
          <CardContent className="flex items-center gap-2 pt-6">
            {saveMessage.type === "success" ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : (
              <AlertCircle className="h-5 w-5 text-destructive" />
            )}
            <span className={saveMessage.type === "success" ? "text-green-500" : "text-destructive"}>
              {saveMessage.text}
            </span>
          </CardContent>
        </Card>
      )}

      {/* Appearance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <SettingsIcon className="h-5 w-5" />
            Appearance
          </CardTitle>
          <CardDescription>Customize the look and feel</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">Theme</label>
              <div className="flex gap-2">
                <Button
                  variant={theme === "light" ? "default" : "outline"}
                  onClick={() => handleThemeChange("light")}
                >
                  <Sun className="mr-2 h-4 w-4" />
                  Light
                </Button>
                <Button
                  variant={theme === "dark" ? "default" : "outline"}
                  onClick={() => handleThemeChange("dark")}
                >
                  <Moon className="mr-2 h-4 w-4" />
                  Dark
                </Button>
                <Button
                  variant={theme === "system" ? "default" : "outline"}
                  onClick={() => handleThemeChange("system")}
                >
                  <Monitor className="mr-2 h-4 w-4" />
                  System
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* API Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            API Configuration
          </CardTitle>
          <CardDescription>Backend connection settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium">API URL</label>
              <Input
                value={settings.apiUrl}
                onChange={(e) => setSettings({ ...settings, apiUrl: e.target.value })}
                placeholder="http://localhost:8000"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Changes require app restart to take effect
              </p>
            </div>
            <div className="flex items-center gap-4">
              {healthLoading ? (
                <Badge variant="secondary" className="flex items-center gap-1">
                  <LoadingSpinner size="sm" />
                  Checking...
                </Badge>
              ) : healthError ? (
                <Badge variant="destructive" className="flex items-center gap-1">
                  <XCircle className="h-3 w-3" />
                  Disconnected
                </Badge>
              ) : (
                <Badge variant="default" className="flex items-center gap-1 bg-green-500">
                  <CheckCircle className="h-3 w-3" />
                  Connected
                </Badge>
              )}
              {healthData && !healthError && (
                <span className="text-sm text-muted-foreground">
                  Latency: {healthData.latency}ms | {healthData.app} v{healthData.version}
                </span>
              )}
              <Button variant="outline" size="sm" onClick={handleTestConnection}>
                Test Connection
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Model Settings</CardTitle>
          <CardDescription>Default model configuration</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <label className="mb-2 block text-sm font-medium">Default Device</label>
                <select
                  value={settings.defaultDevice}
                  onChange={(e) => setSettings({ ...settings, defaultDevice: e.target.value })}
                  className="w-full rounded-md border border-input bg-background px-3 py-2"
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (GPU)</option>
                  <option value="mps">MPS (Apple Silicon)</option>
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Default Sequence Length</label>
                <Input
                  type="number"
                  value={settings.defaultSequenceLength}
                  onChange={(e) => setSettings({ ...settings, defaultSequenceLength: Number(e.target.value) })}
                  min={10}
                  max={200}
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">MC Dropout Samples</label>
                <Input
                  type="number"
                  value={settings.mcDropoutSamples}
                  onChange={(e) => setSettings({ ...settings, mcDropoutSamples: Number(e.target.value) })}
                  min={10}
                  max={200}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
          <CardDescription>Alert preferences</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Training Complete</div>
                <div className="text-sm text-muted-foreground">
                  Notify when model training finishes
                </div>
              </div>
              <input
                type="checkbox"
                checked={settings.trainingNotifications}
                onChange={(e) => setSettings({ ...settings, trainingNotifications: e.target.checked })}
                className="h-4 w-4"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Prediction Alerts</div>
                <div className="text-sm text-muted-foreground">
                  Notify on significant price predictions
                </div>
              </div>
              <input
                type="checkbox"
                checked={settings.predictionAlerts}
                onChange={(e) => setSettings({ ...settings, predictionAlerts: e.target.checked })}
                className="h-4 w-4"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Data Updates</div>
                <div className="text-sm text-muted-foreground">
                  Notify when new data is fetched
                </div>
              </div>
              <input
                type="checkbox"
                checked={settings.dataUpdateNotifications}
                onChange={(e) => setSettings({ ...settings, dataUpdateNotifications: e.target.checked })}
                className="h-4 w-4"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex gap-4">
        <Button onClick={handleSaveSettings} disabled={isSaving}>
          {isSaving ? (
            <>
              <LoadingSpinner size="sm" className="mr-2" />
              Saving...
            </>
          ) : (
            "Save Settings"
          )}
        </Button>
        <Button variant="outline" onClick={handleResetSettings}>
          Reset to Defaults
        </Button>
      </div>
    </div>
  )
}
