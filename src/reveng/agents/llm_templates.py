"""Pydantic models for LLM judge scoring responses."""

from pydantic import BaseModel, Field, field_validator

from reveng.datatypes import Action


class ActionResponse(BaseModel):
    """Action choice for a single step."""

    action: Action = Field(
        description="The chosen action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)"
    )

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v):
        """Convert int or string to Action enum if needed."""
        if isinstance(v, int):
            return Action(v)
        if isinstance(v, str):
            # Try to convert numeric string to int first
            if v.isdigit():
                return Action(int(v))
            # Try to match enum name (case-insensitive)
            return Action[v.upper()]
        return v
