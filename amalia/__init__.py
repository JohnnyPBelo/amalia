"""amalia-v1 — a training-free local re-implementation of Sakana's Conductor."""
from .workers import Worker, WorkerPool, WorkerResult
from .parser import Workflow, parse_workflow, WorkflowParseError
from .engine import WorkflowEngine, WorkflowResult, StepTrace
from .conductor import Conductor, ConductorConfig, ConductorTrace
from .config import AmaliaConfig, load_config

__version__ = "0.1.0"
__all__ = [
    "Worker", "WorkerPool", "WorkerResult",
    "Workflow", "parse_workflow", "WorkflowParseError",
    "WorkflowEngine", "WorkflowResult", "StepTrace",
    "Conductor", "ConductorConfig", "ConductorTrace",
    "AmaliaConfig", "load_config",
]
