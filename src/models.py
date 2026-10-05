from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


ValueDefinition = Literal["string", "number", "integer", "boolean"]


class ParameterValidator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ValueDefinition


class FunctionValidator(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(..., description="Function's description")
    parameters: dict[str, ParameterValidator]
    returns: dict[str, Any]

    @field_validator("name")
    @classmethod
    def check_function_name(cls, v: str) -> str:
        if not v.startswith("fn_"):
            raise ValueError("Invalid function name. Must start with 'fn_'")
        return v


class PromptValidator(BaseModel):
    prompt: str


class OutputValidator(BaseModel):
    prompt: str
    name: str = Field(min_length=3)
    parameters: dict[str, Any]

    @field_validator("name")
    @classmethod
    def check_function_name(cls, v: str) -> str:
        if not v.startswith("fn_"):
            raise ValueError("Invalid function name. Must start with 'fn_'")
        return v
