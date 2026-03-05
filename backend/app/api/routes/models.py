"""Model API endpoints."""

from typing import Optional, List

from fastapi import APIRouter, Depends, Query, HTTPException

from backend.app.dependencies import get_model_service
from backend.app.services.model_service import ModelService
from backend.app.schemas.models import (
    ModelListResponse,
    ModelInfo,
    ModelDetailResponse,
    ModelWeightsInfo,
    ModelComparisonResponse,
)
from backend.app.core.exceptions import ModelNotFoundError, ModelNotTrainedError

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_service: ModelService = Depends(get_model_service),
):
    """List all available models."""
    models = model_service.get_all_models()
    trained = [m for m in models if m.status.value == "trained"]
    pinn = [m for m in models if m.is_pinn]

    return ModelListResponse(
        models=models,
        total=len(models),
        trained_count=len(trained),
        pinn_count=len(pinn),
    )


@router.get("/trained", response_model=ModelListResponse)
async def list_trained_models(
    model_service: ModelService = Depends(get_model_service),
):
    """List only trained models."""
    models = model_service.get_trained_models()
    pinn = [m for m in models if m.is_pinn]

    return ModelListResponse(
        models=models,
        total=len(models),
        trained_count=len(models),
        pinn_count=len(pinn),
    )


@router.get("/types")
async def get_model_types(
    model_service: ModelService = Depends(get_model_service),
):
    """Get available model types for training."""
    return {
        "model_types": model_service.get_available_model_types(),
        "categories": {
            "baseline": ["lstm", "gru", "bilstm", "attention_lstm", "transformer"],
            "pinn": [
                "pinn_baseline", "pinn_gbm", "pinn_ou",
                "pinn_black_scholes", "pinn_gbm_ou", "pinn_global",
            ],
            "advanced": ["stacked_pinn", "residual_pinn"],
        },
    }


@router.get("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    model_keys: str = Query(..., description="Comma-separated model keys"),
    model_service: ModelService = Depends(get_model_service),
):
    """Compare multiple models."""
    keys = [k.strip() for k in model_keys.split(",")]

    comparison = model_service.compare_models(keys)

    return ModelComparisonResponse(
        models=[
            {
                "model_key": m["model_key"],
                "display_name": m["display_name"],
                "is_pinn": m["is_pinn"],
                "metrics": m.get("metrics", {}),
                "physics_parameters": m.get("physics_parameters"),
            }
            for m in comparison["models"]
        ],
        metric_names=comparison["metric_names"],
        best_by_metric=comparison["best_by_metric"],
    )


@router.get("/{model_key}", response_model=ModelInfo)
async def get_model(
    model_key: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Get information about a specific model."""
    try:
        return model_service.get_model_info(model_key)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{model_key}/weights", response_model=ModelWeightsInfo)
async def get_model_weights(
    model_key: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Get information about model weights."""
    try:
        return model_service.get_model_weights_info(model_key)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_key}/load")
async def load_model(
    model_key: str,
    device: Optional[str] = Query(None, description="Device to load model on"),
    model_service: ModelService = Depends(get_model_service),
):
    """Pre-load a model into memory."""
    try:
        model_service.load_model(model_key, device=device)
        return {"message": f"Model {model_key} loaded successfully"}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_key}/unload")
async def unload_model(
    model_key: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Unload a model from memory."""
    model_service.unload_model(model_key)
    return {"message": f"Model {model_key} unloaded"}


@router.get("/checkpoints/list")
async def list_checkpoints(
    model_service: ModelService = Depends(get_model_service),
):
    """List all saved model checkpoint files."""
    checkpoints = model_service.list_saved_checkpoints()
    return {
        "checkpoints": checkpoints,
        "total": len(checkpoints),
    }


@router.post("/checkpoints/rename")
async def rename_checkpoint(
    old_name: str = Query(..., description="Current checkpoint name"),
    new_name: str = Query(..., description="New checkpoint name"),
    model_service: ModelService = Depends(get_model_service),
):
    """Rename a saved checkpoint file."""
    try:
        result = model_service.rename_checkpoint(old_name, new_name)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/checkpoints/{checkpoint_name}")
async def delete_checkpoint(
    checkpoint_name: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a saved checkpoint file."""
    try:
        result = model_service.delete_checkpoint(checkpoint_name)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
