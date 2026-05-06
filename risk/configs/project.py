import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from risk.configs.models import RITSModels


class ApiKeys(BaseModel):
    rits: Optional[str] = os.getenv("RITS_API_KEY")
    watsonx: Optional[str] = os.getenv("WATSONX_APIKEY")
    openai: Optional[str] = os.getenv("OPENAI_API_KEY")

class ModelConfigSettings(BaseSettings):
    model_name: RITSModels = RITSModels.LLAMA_3_3_70B_INSTRUCT
    temperature: float = 0.0
    max_tokens: int = 120_000
    api_keys: ApiKeys = ApiKeys()
    base_url: str | None = Field(
        default=os.environ.get(
            "RITS_BASE_URL",
            "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model}/v1",
        ),
        description="Base URL for the model provider API",
    )
    default_no_answer: str = "Answer not found"


class ProjectSettings(BaseSettings):
    model_config_settings: ModelConfigSettings = ModelConfigSettings()

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.prod"),
        env_file_encoding="utf-8",
        extra="allow",
        env_nested_delimiter="__",
    )


@lru_cache
def get_project_settings() -> ProjectSettings:
    return ProjectSettings()


CONFIG = get_project_settings()
