"""Response envelope for agent-friendly JSON output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class AgentResponse:
    """Standardized response envelope for all CLI commands.

    This provides a consistent structure for LLM agents to parse,
    including success status, data, errors, and actionable suggestions.
    """

    success: bool
    command: str
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    human_summary: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "command": self.command,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "human_summary": self.human_summary,
            "timestamp": datetime.now().isoformat(),
            "schema_version": self.schema_version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def create_response(
    command: str,
    success: bool = True,
    data: Optional[dict[str, Any]] = None,
    errors: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
    suggestions: Optional[list[str]] = None,
    human_summary: str = "",
) -> AgentResponse:
    """Create an AgentResponse with defaults.

    Args:
        command: The command that was executed
        success: Whether the command succeeded
        data: Command-specific result data
        errors: Error messages
        warnings: Non-fatal warning messages
        suggestions: Actionable suggestions for next steps
        human_summary: One-line description for humans

    Returns:
        AgentResponse instance
    """
    return AgentResponse(
        success=success,
        command=command,
        data=data or {},
        errors=errors or [],
        warnings=warnings or [],
        suggestions=suggestions or [],
        human_summary=human_summary,
    )


def error_response(
    command: str,
    error: str,
    suggestions: Optional[list[str]] = None,
) -> AgentResponse:
    """Create an error response.

    Args:
        command: The command that failed
        error: Error message
        suggestions: Suggestions for fixing the error

    Returns:
        AgentResponse with success=False
    """
    return AgentResponse(
        success=False,
        command=command,
        errors=[error],
        suggestions=suggestions or [],
        human_summary=f"Error: {error}",
    )
