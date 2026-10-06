from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


ValueDefinition = Literal["string", "number", "integer", "boolean"]


class ParameterValidator(BaseModel):
    """Describes the type of one parameter of a function definition.

    Attributes:
        type: One of "string", "number", "integer" or "boolean".
    """
    model_config = ConfigDict(extra="forbid")
    type: ValueDefinition


class FunctionValidator(BaseModel):
    """Describes one callable function from functions_definition.json.

    Attributes:
        name: Function name. Must start with "fn_".
        description: Human readable description of the function.
        parameters: Mapping of parameter name to its type definition.
        returns: Description of the return type (kept as-is).
    """
    name: str = Field(min_length=3)
    description: str = Field(..., description="Function's description")
    parameters: dict[str, ParameterValidator]
    returns: dict[str, Any]

    @field_validator("name")
    @classmethod
    def check_function_name(cls, v: str) -> str:
        """Ensure the function name starts with "fn_".

        Args:
            v: The candidate function name.

        Returns:
            The unchanged name if it is valid.

        Raises:
            ValueError: If the name does not start with "fn_".
        """
        if not v.startswith("fn_"):
            raise ValueError("Invalid function name. Must start with 'fn_'")
        return v


class PromptValidator(BaseModel):
    """One natural-language request from function_calling_tests.json.

    Attributes:
        prompt: The user's request.
    """
    prompt: str


class OutputValidator(BaseModel):
    """One entry of the output file.

    Attributes:
        prompt: The original natural-language request.
        name: Name of the chosen function. Must start with "fn_".
        parameters: Arguments extracted for the function call.
    """
    prompt: str
    name: str = Field(min_length=3)
    parameters: dict[str, Any]

    @field_validator("name")
    @classmethod
    def check_function_name(cls, v: str) -> str:
        """Ensure the function name starts with "fn_".

        Args:
            v: The candidate function name.

        Returns:
            The unchanged name if it is valid.

        Raises:
            ValueError: If the name does not start with "fn_".
        """
        if not v.startswith("fn_"):
            raise ValueError("Invalid function name. Must start with 'fn_'")
        return v
