"""HTTP endpoints the dashboard page calls. Each maps to one ScenarioManager action."""

from fastapi import APIRouter, HTTPException, Request

from experiments.fault_injector import FAULTS
from experiments.scenario_manager import ExperimentConfig, InvalidTransition

router = APIRouter(prefix="/api")


def _run(request: Request, action, *args):
    """Call a manager action; an action not allowed in the current state becomes HTTP 409."""
    try:
        action(*args)
    except (InvalidTransition, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return request.app.state.manager.status()


@router.get("/status")
def status(request: Request):
    return request.app.state.manager.status()


@router.get("/faults")
def faults():
    return FAULTS


@router.get("/map")
def map_info(request: Request, town: str = "Town04", spawn: int = 0):
    """Road lines + start + suggested destinations for the dashboard map (may load the town in CARLA)."""
    try:
        return request.app.state.manager.map_info(town, spawn)
    except (InvalidTransition, ConnectionError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:  # CARLA errors (e.g. unknown map name)
        raise HTTPException(status_code=409, detail=f"CARLA: {exc}")


@router.get("/route")
def route_info(request: Request):
    return request.app.state.manager.route_info()


@router.post("/config")
def configure(request: Request, config: dict):
    try:
        cfg = ExperimentConfig(**config)
    except TypeError as exc:  # unknown or missing field
        raise HTTPException(status_code=422, detail=str(exc))
    return _run(request, request.app.state.manager.configure, cfg)


@router.post("/start")
def start(request: Request):
    return _run(request, request.app.state.manager.start)


@router.post("/fault")
def inject_fault(request: Request):
    return _run(request, request.app.state.manager.inject_fault)


@router.post("/stop")
def stop(request: Request):
    return _run(request, request.app.state.manager.stop)


@router.post("/reset")
def reset(request: Request):
    return _run(request, request.app.state.manager.reset)
