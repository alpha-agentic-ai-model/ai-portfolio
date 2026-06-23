"""LLM Structured Output Validation Engine."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar, Type, Optional, get_type_hints

from pydantic import BaseModel, Field, ValidationError


T = TypeVar("T", bound=BaseModel)


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    category: str = Field(description="Entity type")
    confidence: float = Field(ge=0, le=1, description="Extraction confidence")
    relations: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    summary: str
    token_count: int = 0


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_factor: float = 2.0
    partial_recovery: bool = True


@dataclass
class ValidationResult:
    success: bool
    data: Optional[Any] = None
    errors: list[str] = field(default_factory=list)
    attempts: int = 0
    latency_ms: float = 0.0


class SchemaCompiler:
    """Convert Pydantic models to LLM-friendly schema prompts."""

    @staticmethod
    def to_prompt(model: Type[BaseModel]) -> str:
        schema = model.model_json_schema()
        return json.dumps(schema, indent=2)

    @staticmethod
    def to_function_schema(model: Type[BaseModel]) -> dict:
        schema = model.model_json_schema()
        return {
            "name": model.__name__,
            "description": schema.get("description", ""),
            "parameters": schema,
        }


class PartialParser:
    """Recover partial JSON from truncated or malformed LLM output."""

    @staticmethod
    def extract_json(text: str) -> Optional[str]:
        # Try to find JSON block in response
        start = text.find("{")
        if start == -1:
            start = text.find("[")
        if start == -1:
            return None

        # Try progressive closing
        bracket_stack = []
        for i in range(start, len(text)):
            if text[i] in "{[":
                bracket_stack.append("}" if text[i] == "{" else "]")
            elif text[i] in "}]":
                if bracket_stack:
                    bracket_stack.pop()
                if not bracket_stack:
                    return text[start:i + 1]

        # Attempt to close unclosed brackets
        if bracket_stack:
            repaired = text[start:] + "".join(reversed(bracket_stack))
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def try_parse(text: str, model: Type[T]) -> Optional[T]:
        json_str = PartialParser.extract_json(text)
        if not json_str:
            return None
        try:
            data = json.loads(json_str)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            return None


class ConfidenceCalibrator:
    """Score confidence of structured outputs."""

    def __init__(self):
        self.field_weights: dict[str, float] = {}

    def calibrate(self, result: BaseModel, raw_response: str) -> float:
        scores = []
        data = result.model_dump()

        for field_name, value in data.items():
            completeness = 1.0 if value is not None else 0.0
            if isinstance(value, str):
                completeness = min(len(value) / 10, 1.0)
            elif isinstance(value, list):
                completeness = min(len(value) / 3, 1.0)
            weight = self.field_weights.get(field_name, 1.0)
            scores.append(completeness * weight)

        return sum(scores) / max(len(scores), 1)


class MockLLM:
    """Simulated LLM for testing structured output."""

    def __init__(self, fail_rate: float = 0.0):
        self.fail_rate = fail_rate
        self.call_count = 0

    def generate(self, prompt: str, schema: dict) -> str:
        self.call_count += 1
        import random
        if random.random() < self.fail_rate:
            return '{"partial": true, "entities": [{"name": "test"'

        return json.dumps({
            "entities": [
                {"name": "OpenAI", "category": "company",
                 "confidence": 0.95, "relations": ["GPT-4", "DALL-E"]},
                {"name": "Transformer", "category": "architecture",
                 "confidence": 0.88, "relations": ["attention"]},
            ],
            "summary": "Extracted 2 entities from input text",
            "token_count": 150,
        })


class StructuredOutputEngine:
    """Main engine for validated structured LLM outputs."""

    def __init__(self, llm: Optional[MockLLM] = None,
                 retry_config: Optional[RetryConfig] = None):
        self.llm = llm or MockLLM()
        self.config = retry_config or RetryConfig()
        self.parser = PartialParser()
        self.calibrator = ConfidenceCalibrator()
        self.compiler = SchemaCompiler()

    def generate(self, prompt: str, model: Type[T]) -> ValidationResult:
        start = time.monotonic()
        schema = self.compiler.to_function_schema(model)
        full_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(schema['parameters'], indent=2)}"
        )

        last_errors = []
        for attempt in range(1, self.config.max_retries + 1):
            raw = self.llm.generate(full_prompt, schema)
            try:
                data = json.loads(raw)
                result = model.model_validate(data)
                confidence = self.calibrator.calibrate(result, raw)
                latency = (time.monotonic() - start) * 1000
                return ValidationResult(
                    success=True, data=result,
                    attempts=attempt, latency_ms=latency,
                )
            except (json.JSONDecodeError, ValidationError) as e:
                last_errors.append(str(e))
                if self.config.partial_recovery:
                    recovered = self.parser.try_parse(raw, model)
                    if recovered:
                        latency = (time.monotonic() - start) * 1000
                        return ValidationResult(
                            success=True, data=recovered,
                            attempts=attempt, latency_ms=latency,
                        )
                time.sleep(self.config.backoff_base *
                           (self.config.backoff_factor ** (attempt - 1)))

        latency = (time.monotonic() - start) * 1000
        return ValidationResult(
            success=False, errors=last_errors,
            attempts=self.config.max_retries, latency_ms=latency,
        )


if __name__ == "__main__":
    engine = StructuredOutputEngine(llm=MockLLM(fail_rate=0.3))
    result = engine.generate(
        "Extract entities from: OpenAI released GPT-4 using Transformer architecture",
        ExtractionResult,
    )
    print(f"Success: {result.success}, Attempts: {result.attempts}")
    if result.data:
        print(f"Entities: {len(result.data.entities)}")
        for e in result.data.entities:
            print(f"  {e.name} ({e.category}) — confidence: {e.confidence}")
