"""
FastAPI application for the InvoiceGuard Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import InvoiceGuardAction, InvoiceGuardObservation
    from .invoice_guard_environment import InvoiceGuardEnvironment
except (ImportError, ModuleNotFoundError):
    from models import InvoiceGuardAction, InvoiceGuardObservation
    from server.invoice_guard_environment import InvoiceGuardEnvironment


app = create_app(
    InvoiceGuardEnvironment,
    InvoiceGuardAction,
    InvoiceGuardObservation,
    env_name="invoice_guard",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
