"""WebSocket endpoints for real-time updates."""

import asyncio
import logging
from typing import Dict, Set
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.dependencies import get_training_service, get_prediction_service
from backend.app.services.training_service import TrainingService
from backend.app.schemas.training import TrainingStatus
from backend.app.schemas.predictions import PredictionRequest

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept and track a new connection."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove a connection."""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        """Broadcast message to all connections for a job."""
        if job_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)

            # Clean up disconnected
            for conn in disconnected:
                self.active_connections[job_id].discard(conn)


manager = ConnectionManager()


@router.websocket("/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for training progress updates."""
    await manager.connect(websocket, job_id)

    training_service = get_training_service()

    try:
        # Send initial status
        status = training_service.get_job_status(job_id)
        if status:
            await websocket.send_json({
                "type": "status",
                "job_id": job_id,
                "status": status.status.value,
                "current_epoch": status.current_epoch,
                "total_epochs": status.total_epochs,
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Job {job_id} not found",
            })
            return

        # Poll for updates
        last_epoch = 0
        while True:
            status = training_service.get_job_status(job_id)

            if status is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Job not found",
                })
                break

            # Send update if epoch changed
            if status.current_epoch > last_epoch:
                last_epoch = status.current_epoch

                history = training_service.get_job_history(job_id)

                await websocket.send_json({
                    "type": "training_update",
                    "job_id": job_id,
                    "epoch": status.current_epoch,
                    "total_epochs": status.total_epochs,
                    "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                    "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                    "best_val_loss": status.best_val_loss,
                    "progress_percent": status.progress_percent,
                })

            # Check if completed
            if status.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.STOPPED,
            ]:
                await websocket.send_json({
                    "type": "training_complete",
                    "job_id": job_id,
                    "status": status.status.value,
                    "final_metrics": {
                        "best_val_loss": status.best_val_loss,
                        "total_epochs": status.current_epoch,
                    },
                    "saved_model_name": status.saved_model_name,
                })
                break

            # Wait before next poll (2s interval reduces overhead)
            await asyncio.sleep(2.0)

            # Check for client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1,
                )
                if data.get("action") == "stop":
                    training_service.stop_training(job_id)
                    await websocket.send_json({
                        "type": "info",
                        "message": "Stop requested",
                    })
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, job_id)


@router.websocket("/predictions/{ticker}")
async def predictions_websocket(websocket: WebSocket, ticker: str):
    """WebSocket endpoint for real-time prediction updates."""
    await websocket.accept()

    try:
        while True:
            # Wait for client request
            data = await websocket.receive_json()

            if data.get("action") == "subscribe":
                await websocket.send_json({
                    "type": "subscribed",
                    "ticker": ticker,
                    "message": f"Subscribed to {ticker} predictions",
                })

            elif data.get("action") == "predict":
                # Trigger prediction via the prediction service
                model_key = data.get("model_key", "pinn_gbm")

                try:
                    prediction_service = get_prediction_service()
                    request = PredictionRequest(
                        ticker=ticker,
                        model_key=model_key,
                        estimate_uncertainty=True,
                        generate_signal=True,
                    )
                    result = prediction_service.predict(request)

                    await websocket.send_json({
                        "type": "prediction",
                        "ticker": ticker,
                        "model_key": model_key,
                        "predicted_return": result.prediction.predicted_return,
                        "predicted_price": result.prediction.predicted_price,
                        "current_price": result.prediction.current_price,
                        "confidence": result.prediction.confidence_score,
                        "signal": result.prediction.signal_action.value if result.prediction.signal_action else "HOLD",
                        "uncertainty_std": result.prediction.uncertainty_std,
                    })
                except Exception as e:
                    logger.warning("Prediction failed for %s/%s: %s", ticker, model_key, e)
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Prediction failed: {str(e)}",
                    })

    except WebSocketDisconnect:
        pass


@router.websocket("/live")
async def live_updates_websocket(websocket: WebSocket):
    """WebSocket endpoint for general live updates."""
    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to live updates",
        })

        while True:
            data = await websocket.receive_json()

            # Echo back for testing
            await websocket.send_json({
                "type": "echo",
                "data": data,
            })

    except WebSocketDisconnect:
        pass


@router.websocket("/batch-training/{batch_id}")
async def batch_training_websocket(websocket: WebSocket, batch_id: str):
    """WebSocket endpoint for batch training progress updates."""
    await manager.connect(websocket, batch_id)

    training_service = get_training_service()

    try:
        # Send initial status
        status = training_service.get_batch_status(batch_id)
        if status:
            await websocket.send_json({
                "type": "batch_status",
                "batch_id": batch_id,
                "status": status.status.value,
                "total_models": status.total_models,
                "completed_models": status.completed_models,
                "current_model": status.current_model,
                "overall_progress": status.overall_progress,
                "models": [m.model_dump(mode='json') for m in status.models],
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Batch job {batch_id} not found",
            })
            return

        # Poll for updates
        last_update_hash = ""
        while True:
            status = training_service.get_batch_status(batch_id)

            if status is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Batch job not found",
                })
                break

            # Create update hash to detect changes (includes batch-level progress)
            current_hash = f"{status.completed_models}:{status.overall_progress:.1f}"
            for model in status.models:
                # Include current_batch for more frequent updates during long epochs
                current_hash += f":{model.model_key}:{model.current_epoch}:{model.current_batch}"

            if current_hash != last_update_hash:
                # Debug: log when we detect a change and send an update
                logger.debug("Hash changed: sending update (batch=%s)", status.models[0].current_batch if status.models else 0)
                last_update_hash = current_hash

                # Find the currently training model for detailed update
                current_model_status = None
                for model in status.models:
                    if model.status == TrainingStatus.RUNNING:
                        current_model_status = model
                        break

                if current_model_status:
                    await websocket.send_json({
                        "type": "batch_training_update",
                        "batch_id": batch_id,
                        "model_key": current_model_status.model_key,
                        "model_name": current_model_status.model_name,
                        "epoch": current_model_status.current_epoch,
                        "total_epochs": current_model_status.total_epochs,
                        # Batch-level progress for real-time updates within each epoch
                        "current_batch": current_model_status.current_batch,
                        "total_batches": current_model_status.total_batches,
                        "batch_loss": current_model_status.batch_loss,
                        "train_loss": current_model_status.train_loss,
                        "val_loss": current_model_status.val_loss,
                        "best_val_loss": current_model_status.best_val_loss,
                        "data_loss": current_model_status.data_loss,
                        "physics_loss": current_model_status.physics_loss,
                        "learning_rate": current_model_status.learning_rate,
                        "overall_progress": status.overall_progress,
                        "completed_models": status.completed_models,
                        "total_models": status.total_models,
                    })

                # Send model completion updates
                await websocket.send_json({
                    "type": "batch_progress",
                    "batch_id": batch_id,
                    "status": status.status.value,
                    "current_model": status.current_model,
                    "overall_progress": status.overall_progress,
                    "completed_models": status.completed_models,
                    "failed_models": status.failed_models,
                    "total_models": status.total_models,
                    "models": [
                        {
                            "model_key": m.model_key,
                            "model_name": m.model_name,
                            "model_type": m.model_type,
                            "status": m.status.value,
                            "current_epoch": m.current_epoch,
                            "total_epochs": m.total_epochs,
                            # Batch-level progress
                            "current_batch": m.current_batch,
                            "total_batches": m.total_batches,
                            "batch_loss": m.batch_loss,
                            "train_loss": m.train_loss,
                            "val_loss": m.val_loss,
                            "best_val_loss": m.best_val_loss,
                            "data_loss": m.data_loss,
                            "physics_loss": m.physics_loss,
                            "progress_percent": m.progress_percent,
                        }
                        for m in status.models
                    ],
                })

            # Check if completed
            if status.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.STOPPED,
            ]:
                # Send final summary
                summary = []
                for model in status.models:
                    summary.append({
                        "model_key": model.model_key,
                        "model_name": model.model_name,
                        "status": model.status.value,
                        "best_val_loss": model.best_val_loss,
                        "epochs_trained": model.current_epoch,
                    })

                await websocket.send_json({
                    "type": "batch_training_complete",
                    "batch_id": batch_id,
                    "status": status.status.value,
                    "total_models": status.total_models,
                    "completed_models": status.completed_models,
                    "failed_models": status.failed_models,
                    "summary": summary,
                })
                break

            # Wait before next poll (2s interval reduces overhead while maintaining responsiveness)
            await asyncio.sleep(2.0)

            # Check for client messages
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1,
                )
                if data.get("action") == "stop":
                    training_service.stop_batch_training(batch_id)
                    await websocket.send_json({
                        "type": "info",
                        "message": "Batch stop requested",
                    })
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, batch_id)
