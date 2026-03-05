"""Training API endpoints."""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException

from backend.app.dependencies import get_training_service
from backend.app.services.training_service import TrainingService, get_training_mode_info
from backend.app.schemas.training import (
    TrainingRequest,
    TrainingStartResponse,
    TrainingStopResponse,
    TrainingStatusResponse,
    TrainingHistoryResponse,
    TrainingRunListResponse,
    TrainingStatus,
    # Batch training schemas
    BatchTrainingRequest,
    BatchTrainingStartResponse,
    BatchTrainingStatusResponse,
    BatchTrainingJobInfo,
    AvailableModelsResponse,
    AVAILABLE_MODELS,
)
from backend.app.core.exceptions import TrainingError

router = APIRouter()


@router.get("/mode")
async def get_training_mode():
    """
    Get current training mode information.

    Returns whether the backend is using real neural network training
    or simulated training (for when dependencies are missing).
    """
    return get_training_mode_info()


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
):
    """Start a new training job."""
    try:
        job_id = training_service.start_training(request)

        return TrainingStartResponse(
            success=True,
            job_id=job_id,
            message=f"Training job {job_id} started for {request.model_type}",
            websocket_url=f"/api/ws/training/{job_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{job_id}", response_model=TrainingStopResponse)
async def stop_training(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Stop a running training job."""
    success = training_service.stop_training(job_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = training_service.get_job_status(job_id)

    return TrainingStopResponse(
        success=True,
        job_id=job_id,
        message=f"Training job {job_id} stopped",
        final_status=status.status if status else TrainingStatus.STOPPED,
    )


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Get status of a training job."""
    job_info = training_service.get_job_status(job_id)

    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    history = training_service.get_job_history(job_id)

    return TrainingStatusResponse(
        job=job_info,
        history=history,
    )


@router.get("/history/{job_id}", response_model=TrainingHistoryResponse)
async def get_training_history(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Get detailed training history for a job."""
    job_info = training_service.get_job_status(job_id)

    if job_info is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    epochs = training_service.get_epoch_metrics(job_id)
    history = training_service.get_job_history(job_id)

    best_epoch = 1
    best_val_loss = float("inf")
    for i, epoch in enumerate(epochs):
        if epoch.val_loss < best_val_loss:
            best_val_loss = epoch.val_loss
            best_epoch = i + 1

    return TrainingHistoryResponse(
        job_id=job_id,
        model_type=job_info.model_type,
        epochs=epochs,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_metrics={
            "final_train_loss": epochs[-1].train_loss if epochs else 0,
            "final_val_loss": epochs[-1].val_loss if epochs else 0,
            "best_val_loss": best_val_loss,
        },
    )


@router.get("/history", response_model=TrainingRunListResponse)
async def list_training_runs(
    status: Optional[TrainingStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    training_service: TrainingService = Depends(get_training_service),
):
    """List all training runs."""
    result = training_service.list_jobs(
        status=status,
        page=page,
        page_size=page_size,
    )
    return TrainingRunListResponse(**result)


@router.get("/active")
async def get_active_jobs(
    training_service: TrainingService = Depends(get_training_service),
):
    """Get currently running training jobs."""
    result = training_service.list_jobs(status=TrainingStatus.RUNNING)
    return {
        "active_jobs": result["runs"],
        "count": len(result["runs"]),
    }


# ============== Batch Training Endpoints ==============


@router.get("/batch/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get all available models for batch training with their configurations."""
    by_type = {}
    for model_info in AVAILABLE_MODELS.values():
        model_type = model_info["type"]
        by_type[model_type] = by_type.get(model_type, 0) + 1

    return AvailableModelsResponse(
        models=AVAILABLE_MODELS,
        total=len(AVAILABLE_MODELS),
        by_type=by_type,
    )


@router.post("/batch/start", response_model=BatchTrainingStartResponse)
async def start_batch_training(
    request: BatchTrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
):
    """Start batch training of multiple models."""
    try:
        # Filter enabled models
        enabled_models = [m for m in request.models if m.enabled]

        if not enabled_models:
            raise HTTPException(status_code=400, detail="No models enabled for training")

        # Validate model keys
        invalid_models = [
            m.model_key for m in enabled_models
            if m.model_key not in AVAILABLE_MODELS
        ]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model keys: {', '.join(invalid_models)}"
            )

        batch_id = training_service.start_batch_training(request, enabled_models)

        model_keys = [m.model_key for m in enabled_models]

        return BatchTrainingStartResponse(
            success=True,
            batch_id=batch_id,
            message=f"Batch training started for {len(enabled_models)} models",
            total_models=len(enabled_models),
            model_keys=model_keys,
            websocket_url=f"/api/ws/batch-training/{batch_id}",
        )
    except RuntimeError as e:
        # Most common case: a batch is already running
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/stop/{batch_id}")
async def stop_batch_training(
    batch_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Stop a running batch training job."""
    success = training_service.stop_batch_training(batch_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

    return {
        "success": True,
        "batch_id": batch_id,
        "message": f"Batch training {batch_id} stopped",
    }


@router.get("/batch/status/{batch_id}", response_model=BatchTrainingStatusResponse)
async def get_batch_training_status(
    batch_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Get status of a batch training job."""
    batch_info = training_service.get_batch_status(batch_id)

    if batch_info is None:
        raise HTTPException(status_code=404, detail=f"Batch job {batch_id} not found")

    history = training_service.get_batch_history(batch_id)

    return BatchTrainingStatusResponse(
        batch=batch_info,
        history=history,
    )


@router.get("/batch/list")
async def list_batch_jobs(
    status: Optional[TrainingStatus] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    training_service: TrainingService = Depends(get_training_service),
):
    """List all batch training jobs."""
    result = training_service.list_batch_jobs(
        status=status,
        page=page,
        page_size=page_size,
    )
    return result
