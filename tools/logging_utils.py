"""
logging_utils.py
================
Shared logging utilities for the orchestrator and tools.
Ensures consistent prefixing with the current agent's role in Docker logs.
"""

import os
from rich import print as _rprint

# Use Orchestrator as default for initial setup logs
os.environ.setdefault("CREWAI_CURRENT_AGENT", "Orchestrator")

def rprint(*args, **kwargs):
    """Wrapped rich.print that prefixes the current agent name."""
    agent = os.getenv("CREWAI_CURRENT_AGENT", "Orchestrator")
    prefix = f"[bold cyan][{agent}][/bold cyan]"
    _rprint(prefix, *args, **kwargs)
