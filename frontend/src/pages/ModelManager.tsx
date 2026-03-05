import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Badge } from "../components/ui/badge"
import { LoadingSpinner } from "../components/common/LoadingSpinner"
import {
  AlertCircle,
  Pencil,
  Trash2,
  Save,
  X,
  FileBox,
  HardDrive,
  Calendar,
} from "lucide-react"
import api from "../services/api"

interface Checkpoint {
  filename: string
  name: string
  path: string
  size_bytes: number
  size_mb: number
  modified: string
  created: string
  is_best: boolean
}

export default function ModelManager() {
  const queryClient = useQueryClient()
  const [editingName, setEditingName] = useState<string | null>(null)
  const [newName, setNewName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["checkpoints"],
    queryFn: async () => {
      const response = await api.get("/api/models/checkpoints/list")
      return response.data as { checkpoints: Checkpoint[]; total: number }
    },
  })

  const renameMutation = useMutation({
    mutationFn: async ({ oldName, newName }: { oldName: string; newName: string }) => {
      const response = await api.post(
        `/api/models/checkpoints/rename?old_name=${encodeURIComponent(oldName)}&new_name=${encodeURIComponent(newName)}`
      )
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checkpoints"] })
      setEditingName(null)
      setNewName("")
      setError(null)
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || "Failed to rename checkpoint")
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (name: string) => {
      const response = await api.delete(`/api/models/checkpoints/${encodeURIComponent(name)}`)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["checkpoints"] })
      setDeleteConfirm(null)
      setError(null)
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || "Failed to delete checkpoint")
    },
  })

  const startEditing = (checkpoint: Checkpoint) => {
    setEditingName(checkpoint.name)
    setNewName(checkpoint.name)
    setError(null)
  }

  const cancelEditing = () => {
    setEditingName(null)
    setNewName("")
    setError(null)
  }

  const handleRename = () => {
    if (editingName && newName && editingName !== newName) {
      renameMutation.mutate({ oldName: editingName, newName })
    }
  }

  const handleDelete = (name: string) => {
    deleteMutation.mutate(name)
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  const parseModelInfo = (name: string) => {
    // Try to parse: {model_type}_{ticker}_{date}_{epochs}ep_best
    const parts = name.split("_")
    if (parts.length >= 4) {
      const isBest = parts[parts.length - 1] === "best"
      const epochPart = isBest ? parts[parts.length - 2] : parts[parts.length - 1]
      const epochs = epochPart.replace("ep", "")

      // Find model type and ticker
      let modelType = parts[0]
      let ticker = ""
      let dateStr = ""

      // Handle model types like "pinn_gbm_ou"
      for (let i = 1; i < parts.length - 2; i++) {
        const part = parts[i]
        // Check if it looks like a ticker (2-5 uppercase letters)
        if (/^[A-Z]{2,5}$/.test(part)) {
          ticker = part
          break
        } else if (/^\d{8}$/.test(part)) {
          dateStr = part
          break
        } else {
          modelType += "_" + part
        }
      }

      // Find date after ticker
      for (let i = 0; i < parts.length; i++) {
        if (/^\d{8}$/.test(parts[i])) {
          dateStr = parts[i]
          break
        }
      }

      return { modelType, ticker, dateStr, epochs, isBest }
    }
    return null
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Model Manager</h1>
          <p className="text-muted-foreground">
            Manage saved model checkpoints - rename, view, or delete
          </p>
        </div>
        <Button onClick={() => refetch()} variant="outline">
          Refresh
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-2 pt-6">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-destructive">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setError(null)}
              className="ml-auto"
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-4 pt-6">
            <FileBox className="h-8 w-8 text-primary" />
            <div>
              <div className="text-2xl font-bold">{data?.total || 0}</div>
              <div className="text-sm text-muted-foreground">Total Checkpoints</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-4 pt-6">
            <HardDrive className="h-8 w-8 text-primary" />
            <div>
              <div className="text-2xl font-bold">
                {data?.checkpoints
                  ? (data.checkpoints.reduce((sum, c) => sum + c.size_mb, 0)).toFixed(1)
                  : 0} MB
              </div>
              <div className="text-sm text-muted-foreground">Total Size</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-4 pt-6">
            <Calendar className="h-8 w-8 text-primary" />
            <div>
              <div className="text-2xl font-bold">
                {data?.checkpoints?.[0]
                  ? new Date(data.checkpoints[0].modified).toLocaleDateString()
                  : "--"}
              </div>
              <div className="text-sm text-muted-foreground">Last Modified</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Checkpoints List */}
      <Card>
        <CardHeader>
          <CardTitle>Saved Checkpoints</CardTitle>
          <CardDescription>
            Click the pencil icon to rename a checkpoint. Names should be descriptive
            (e.g., pinn_gbm_AAPL_20260210_100ep).
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <LoadingSpinner size="lg" />
            </div>
          ) : data?.checkpoints?.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No checkpoints found. Train a model to create checkpoints.
            </div>
          ) : (
            <div className="space-y-4">
              {data?.checkpoints?.map((checkpoint) => {
                const info = parseModelInfo(checkpoint.name)
                const isEditing = editingName === checkpoint.name
                const isDeleting = deleteConfirm === checkpoint.name

                return (
                  <div
                    key={checkpoint.filename}
                    className="flex items-center justify-between rounded-lg border p-4"
                  >
                    <div className="flex-1">
                      {isEditing ? (
                        <div className="flex items-center gap-2">
                          <Input
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            className="max-w-md"
                            placeholder="Enter new name"
                            autoFocus
                          />
                          <Button
                            size="sm"
                            onClick={handleRename}
                            disabled={renameMutation.isPending || newName === editingName}
                          >
                            {renameMutation.isPending ? (
                              <LoadingSpinner size="sm" />
                            ) : (
                              <Save className="h-4 w-4" />
                            )}
                          </Button>
                          <Button size="sm" variant="ghost" onClick={cancelEditing}>
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ) : (
                        <>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{checkpoint.name}</span>
                            {checkpoint.is_best && (
                              <Badge variant="success">Best</Badge>
                            )}
                          </div>
                          {info && (
                            <div className="flex gap-2 mt-1">
                              <Badge variant="outline">{info.modelType}</Badge>
                              {info.ticker && (
                                <Badge variant="secondary">{info.ticker}</Badge>
                              )}
                              {info.epochs && (
                                <Badge variant="secondary">{info.epochs} epochs</Badge>
                              )}
                            </div>
                          )}
                          <div className="text-sm text-muted-foreground mt-1">
                            {checkpoint.size_mb} MB | Modified: {formatDate(checkpoint.modified)}
                          </div>
                        </>
                      )}
                    </div>

                    {!isEditing && (
                      <div className="flex items-center gap-2">
                        {isDeleting ? (
                          <>
                            <span className="text-sm text-destructive mr-2">Delete?</span>
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={() => handleDelete(checkpoint.name)}
                              disabled={deleteMutation.isPending}
                            >
                              {deleteMutation.isPending ? (
                                <LoadingSpinner size="sm" />
                              ) : (
                                "Yes"
                              )}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setDeleteConfirm(null)}
                            >
                              No
                            </Button>
                          </>
                        ) : (
                          <>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => startEditing(checkpoint)}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => setDeleteConfirm(checkpoint.name)}
                              className="text-destructive hover:text-destructive"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
