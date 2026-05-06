from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

DEEPEVAL_SYSTEM_PROMPT = (
    """IMPORTANT: Please make sure to only return in JSON format, with the \'output\' key as the output from the instructions.\n"""
    """Example JSON:\n{\n    "output": "your output goes here"\n}\n**\n\nJSON:\n"""
)

def get_risk_root() -> Path:
    """
    Returns the root directory of the risk project.
    """
    return Path(__file__).parent.parent.resolve()


def apply_partial_config(base_config: BaseModel, overrides: BaseModel) -> BaseModel:
    """
    Return a copy of `base_config` with non-None attributes from `overrides`.
    Does not modify the original.
    """
    override_data = overrides.model_dump(exclude_none=True)
    base_data = base_config.model_dump()

    if "model_name" in override_data and override_data["model_name"] != base_data.get(
        "model_name",
    ):
        for dep_field in ("provider", "base_url", "api_key"):
            base_data.pop(dep_field, None)

    base_data.update(override_data)
    return base_config.__class__(**base_data)


class CustomChat(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        model_name="Custom Model",
    ):
        self.model = model
        self.model_name = model_name

    def load_model(self):
        return self.model

    def format_messages(self, prompt, system_prompt = DEEPEVAL_SYSTEM_PROMPT):

        system_message = SystemMessage(
            content=system_prompt,
        )
        human_message = HumanMessage(content=prompt)

        return [system_message, human_message]

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(self.format_messages(prompt)).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()

        res = await chat_model.ainvoke(
            self.format_messages(prompt)
        )
        return res.content

    def get_model_name(self):
        return self.model_name
