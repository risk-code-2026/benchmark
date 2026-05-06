from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar, Dict, Type, Union

import instructor
from litellm import acompletion
from litellm.utils import trim_messages
from loguru import logger
from pydantic import BaseModel, Field, create_model, model_validator
import openai

from risk.configs.models import ModelProvider, RITSModels, WatsonxModels
from risk.configs.project import CONFIG

# === Base Schemas ===


class InputSchema(BaseModel):
    """Structured input for the builder."""


class OutputSchema(BaseModel):
    """Structured output from the builder."""


# === Configuration ===


class BaseBuilderConfig(BaseModel):
    """Configuration for LiteLLM-based agents."""

    model_name: Union[RITSModels, WatsonxModels, str] | None = Field(
        default=CONFIG.model_config_settings.model_name,
    )
    temperature: float = Field(
        default=CONFIG.model_config_settings.temperature,
        ge=0.0,
        le=2.0,
    )
    system_prompt: str = Field(..., description="System prompt for builder")
    max_tokens: int = Field(
        default=CONFIG.model_config_settings.max_tokens,
        ge=5,
        le=1_000_000,
    )
    trim_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    enable_trimming: bool = Field(default=True)
    debug: bool = Field(default=False)

    provider: ModelProvider | None = None
    base_url: str | None = None
    api_key: str | None = None

    PROVIDER_INFO: ClassVar[Dict[ModelProvider, Dict[str, str | None]]] = {
        ModelProvider.RITS: {
            "env_url": "RITS_BASE_URL",
            "env_key": "RITS_API_KEY",
            "default_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model}/v1",
        },
        ModelProvider.WATSONX: {
            "env_url": "WATSONX_URL",
            "env_key": "WATSONX_APIKEY",
            "default_url": "https://us-south.ml.cloud.ibm.com/",
        },
        ModelProvider.OPENAI: {
            "env_url": None,
            "env_key": "OPENAI_API_KEY",
            "default_url": None,
        },
    }

    @model_validator(mode="after")
    def infer_provider_and_fill(self) -> "BaseBuilderConfig":
        """Infer provider from model_name and fill provider-dependent fields."""

        if self.model_name is None:
            raise ValueError("model_name must be specified")

        if isinstance(self.model_name, RITSModels):
            provider = ModelProvider.RITS
        elif isinstance(self.model_name, WatsonxModels):
            provider = ModelProvider.WATSONX
        elif isinstance(self.model_name, str):
            provider = ModelProvider.OPENAI
        else:
            raise ValueError(
                f"Unsupported model type: {type(self.model_name).__name__}"
            )

        self.provider = provider
        info = self.PROVIDER_INFO[provider]

        env_url = info["env_url"]
        env_key = info["env_key"]
        default_url = info["default_url"]

        if not self.base_url and env_url:
            self.base_url = os.getenv(env_url, default_url)

        if not self.api_key and env_key:
            self.api_key = os.getenv(env_key)

        # Optional provider-specific fallbacks
        if not self.api_key and provider == ModelProvider.RITS:
            self.api_key = CONFIG.model_config_settings.api_keys.rits

        if not self.api_key and provider == ModelProvider.WATSONX:
            self.api_key = CONFIG.model_config_settings.api_keys.watsonx

        if not self.api_key and provider == ModelProvider.OPENAI:
            self.api_key = CONFIG.model_config_settings.api_keys.openai

        if not self.api_key:
            raise ValueError(f"Missing API key for provider '{provider.value}'.")

        if provider is not ModelProvider.OPENAI and not self.base_url:
            raise ValueError(f"Missing base URL for provider '{provider.value}'.")

        return self


class BaseBuilderConfig_(BaseModel):
    """Configuration for LiteLLM-based agents."""

    # stuff that user can override
    model_name: Union[RITSModels, WatsonxModels, str] | None = Field(
        default=CONFIG.model_config_settings.model_name,
    )
    temperature: float = Field(
        default=CONFIG.model_config_settings.temperature,
        ge=0.0,
        le=2.0,
    )
    system_prompt: str = Field(..., description="System prompt for builder")
    max_tokens: int = Field(
        default=CONFIG.model_config_settings.max_tokens,
        ge=5,
        le=1_000_000,
    )
    trim_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    enable_trimming: bool = Field(
        default=True,
        description="Enable automatic message trimming",
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    # stuff that gets set automatically
    provider: ModelProvider | None = None
    base_url: str | None = None
    api_key: str | None = None

    # --- Static provider info ---
    PROVIDER_INFO: ClassVar[Dict[ModelProvider, Dict[str, str]]] = {
        ModelProvider.RITS: {
            "env_url": "RITS_BASE_URL",
            "env_key": "RITS_API_KEY",
            "default_url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model}/v1",
        },
        ModelProvider.WATSONX: {
            "env_url": "WATSONX_URL",
            "env_key": "WATSONX_APIKEY",
            "default_url": "https://us-south.ml.cloud.ibm.com/",
        },
        ModelProvider.OPENAI: { #dummy values here
            "env_url": "",
            "env_key": "",
            "default_url": "",
        },
    }

    @model_validator(mode="after")
    def infer_provider_and_fill(cls, values: BaseBuilderConfig) -> BaseBuilderConfig:
        """Infer provider from model_name and fill provider-dependent fields."""
        model_name = values.model_name
        if model_name is None:
            raise ValueError("model_name must be specified")

        # --- Infer provider from model type ---
        if isinstance(model_name, RITSModels):
            provider = ModelProvider.RITS
        elif isinstance(model_name, WatsonxModels):
            provider = ModelProvider.WATSONX
        elif isinstance(model_name, str):
            provider = ModelProvider.OPENAI
        else:
            raise ValueError(f"Unsupported model type: {type(model_name).__name__}")

        values.provider = provider

        # --- Lookup provider metadata ---
        info = cls.PROVIDER_INFO[provider]

        # Base URL
        if not values.base_url:
            values.base_url = os.getenv(info["env_url"], info["default_url"])

        # API key
        if not values.api_key:
            env_key = os.getenv(info["env_key"])
            values.api_key = (
                env_key or CONFIG.model_config_settings.api_keys.rits
            )  # fallback

        # Final sanity checks
        if not values.api_key:
            raise ValueError(f"Missing API key for provider '{provider.value}'.")
        if not values.base_url:
            raise ValueError(f"Missing base URL for provider '{provider.value}'.")

        return values


class LiteLLMInstructorBaseBuilder:
    """
    Self-contained builder integrating Instructor with LiteLLM.
    Provides structured async calls, automatic message trimming,
    and Pydantic-based input/output handling.
    """

    input_schema: Type[BaseModel] = InputSchema
    output_schema: Type[BaseModel] = OutputSchema
    config_class: Type[BaseModel] = BaseBuilderConfig

    def __init__(
        self,
        config: BaseBuilderConfig | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        self.config = config or self.config_class(**kwargs)
        self.debug = debug or self.config.debug
        if self.config.provider in [ModelProvider.RITS, ModelProvider.WATSONX]:
            self.client = instructor.from_litellm(acompletion)
        else:
            self.client = instructor.from_openai(openai.AsyncOpenAI(api_key=self.config.api_key))

        if self.debug:
            logger.debug(
                f"Initialized {self.__class__.__name__} with config: {self.config}",
            )

    def _create_instructor_compatible_model(
        self,
        response_model: Type[BaseModel],
    ) -> Type[BaseModel]:
        """Rebuilds a model compatible with Instructor, bypassing IOSchema internals."""
        fields = {
            name: (field.annotation, field)
            for name, field in response_model.model_fields.items()
        }
        model = create_model(response_model.__name__, __base__=BaseModel, **fields)
        model.__doc__ = response_model.__doc__
        return model

    async def get_response(
        self,
        messages: list[dict[str, str]],
        response_model: Type[BaseModel] | None = None,
        max_retries: int = 5,
        retry_delay: float = 2,
    ) -> BaseModel:
        """
        Handles message trimming, model invocation, and response casting.

        Args:
            messages: List of role-content dicts for the chat completion.
            response_model: Optional Pydantic model type for structured output.

        Returns:
            Instance of the response model containing structured data.
        """
        cfg = self.config

        if cfg.enable_trimming:
            messages = trim_messages(
                messages,
                model=cfg.model_name,
                max_tokens=cfg.max_tokens,
                trim_ratio=cfg.trim_ratio,
            )

        response_model = response_model or self.output_schema
        instructor_model = self._create_instructor_compatible_model(response_model)

        if self.debug:
            logger.debug(
                f"[{self.__class__.__name__}] Sending {len(messages)} messages to {cfg.model_name}",
            )

        extra_headers = {}
        if cfg.provider == ModelProvider.RITS:
            extra_headers["RITS_API_KEY"] = cfg.api_key
            if cfg.base_url:
                api_base = cfg.base_url.rstrip("/").format(
                    model=cfg.model_name.split("/")[-1],
                )
            else:
                None
        elif cfg.provider == ModelProvider.WATSONX:
            api_base = cfg.base_url

        for attempt in range(1, max_retries + 1):
            logger.info(f"Attempt {attempt} of {max_retries}.")
            try:
                if cfg.provider in [ModelProvider.WATSONX, ModelProvider.RITS]:
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        model=cfg.model_name,
                        temperature=cfg.temperature,
                        response_model=instructor_model,
                        api_base=api_base,
                        api_key=cfg.api_key,
                        extra_headers=extra_headers,
                    )
                    break
                elif cfg.provider in [ModelProvider.OPENAI]:
                    kwargs = {
                        "model": cfg.model_name,
                        "messages": messages,
                        "response_model": instructor_model,
                    }

                    if cfg.model_name not in ["gpt-5-nano"]:
                        kwargs["temperature"] = 0.0

                    response = await self.client.chat.completions.create(**kwargs)
                    break
            except Exception as e:
                err_type = type(e).__name__
                logger.warning(
                    f"[Attempt {attempt}/{max_retries}] Error in get_response ({err_type}): {e}",
                )

                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * attempt)
                else:
                    logger.error(
                        f"get_response_async failed after {max_retries} attempts: {e}",
                    )
                    raise e

        response_data = response.model_dump()
        return response_model(**response_data)

    async def _arun(self, params: InputSchema) -> OutputSchema:
        """
        Default async implementation.
        Subclasses can override this for custom behavior, e.g. custom message building.
        """
        if not isinstance(params, self.input_schema):
            raise TypeError(
                f"Expected {self.input_schema.__name__}, got {type(params).__name__}",
            )

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": params.model_dump_json()},
        ]

        return await self.get_response(messages, response_model=self.output_schema)

    async def arun(self, params: InputSchema) -> OutputSchema:
        """Public entrypoint for running the builder asynchronously."""
        try:
            return await self._arun(params)
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise
