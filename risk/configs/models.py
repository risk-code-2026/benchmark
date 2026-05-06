from enum import Enum


class ModelProvider(str, Enum):
    """Enum for supported model providers"""

    RITS = "rits"
    WATSONX = "watsonx"  
    OPENAI = "openai"


class RITSModels(str, Enum):
    """Enum for RITS-hosted models."""

    # DeepSeek
    DEEPSEEK_V3 = "openai/deepseek-ai/DeepSeek-V3"

    # Granite
    GRANITE_3_0_8B_INSTRUCT = "openai/ibm-granite/granite-3.0-8b-instruct"
    GRANITE_3_1_8B_INSTRUCT = "openai/ibm-granite/granite-3.1-8b-instruct"
    GRANITE_3_2_8B_INSTRUCT = "openai/ibm-granite/granite-3.2-8b-instruct"
    GRANITE_3_3_8B_INSTRUCT = "openai/ibm-granite/granite-3.3-8b-instruct"
    GRANITE_GUARDIAN_3_2_5B = "openai/ibm-granite/granite-guardian-3.2-5b"
    GRANITE_GUARDIAN_3_2_3B_A800M = "openai/ibm-granite/granite-guardian-3.2-3b-a800m"

    # Llama series
    LLAMA_3_1_8B_INSTRUCT = "openai/ibm-granite/Meta-Llama-3.1-8B-Instruct"
    LLAMA_3_1_405B_INSTRUCT = "openai/meta-llama/llama-3-1-405b-instruct-fp8"
    LLAMA_3_2_11B_INSTRUCT = "openai/meta-llama/Llama-3.2-11B-Vision-Instruct"
    LLAMA_3_2_90B_INSTRUCT = "openai/meta-llama/Llama-3.2-90B-Vision-Instruct"
    LLAMA_3_3_70B_INSTRUCT = "openai/meta-llama/llama-3-3-70b-instruct"
    LLAMA_4_MAVERICK_17B_128E_FP8 = (
        "openai/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    )
    LLAMA_4_SCOUT_17B_16E = "openai/meta-llama/llama-4-scout-17b-16e"

    # Mixtral
    MIXTRAL_8X7B_INSTRUCT = "openai/mistralai/mixtral-8x7B-instruct-v0.1"
    MIXTRAL_8X22B_INSTRUCT = "openai/mistralai/mixtral-8x22B-instruct-v0.1"
    MIXTRAL_8X22B_INSTRUCT_PRIORITY = "openai/mistralai/mixtral-8x22B-instruct-v0.1"
    MIXTRAL_SMALL_3_1=  "openai/mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    # Phi and GPT-OSS
    PHI_4 = "openai/microsoft/phi-4"
    GPT_OSS_20B = "openai/openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/openai/gpt-oss-120b"


class WatsonxModels(str, Enum):
    """Enum for watsonx-hosted models (served via watsonx.ai endpoints)."""

    # Llama series
    LLAMA_3_3_70B_INSTRUCT = "watsonx/meta-llama/llama-3-3-70b-instruct"
    LLAMA_3_2_1B_INSTRUCT = "watsonx/meta-llama/llama-3-2-1b-instruct"
    LLAMA_3_2_3B_INSTRUCT = "watsonx/meta-llama/llama-3-2-3b-instruct"
    LLAMA_3_2_11B_VISION_INSTRUCT = "watsonx/meta-llama/llama-3-2-11b-vision-instruct"
    LLAMA_3_2_90B_VISION_INSTRUCT = "watsonx/meta-llama/llama-3-2-90b-vision-instruct"
    LLAMA_3_405B_INSTRUCT = "watsonx/meta-llama/llama-3-405b-instruct"
    LLAMA_4_MAVERICK_17B_128E_INSTRUCT_FP8 = (
        "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    )
    LLAMA_2_13B_CHAT = "watsonx/meta-llama/llama-2-13b-chat"
    LLAMA_GUARD_3_11B_VISION = "watsonx/meta-llama/llama-guard-3-11b-vision"

    # Granite series
    GRANITE_3_3_8B_INSTRUCT = "watsonx/ibm/granite-3.3-8b-instruct"
    GRANITE_4H_SMALL = "watsonx/ibm/granite-4-h-small"

    # Allam
    ALLAM_1_13B_INSTRUCT = "watsonx/sdaia/allam-1-13b-instruct"

    # Jais
    JAIS_13B_CHAT = "watsonx/core42/jais-13b-chat"

    # GPT-OSS
    GPT_OSS_120B = "watsonx/openai/gpt-oss-120b"

    # Mistral family
    MISTRAL_LARGE = "watsonx/mistralai/mistral-large"
    MISTRAL_MEDIUM_2505 = "watsonx/mistralai/mistral-medium-2505"
    MISTRAL_SMALL_3_1_24B_INSTRUCT_2503 = (
        "watsonx/mistralai/mistral-small-3-1-24b-instruct-2503"
    )

    # BigScience
    MT0_XXL_13B = "watsonx/bigscience/mt0-xxl-13b"


MODEL_PROVIDER_MAP = {
    "rits": RITSModels,
    "watsonx": WatsonxModels,
}
