from typing import Any, Dict, List, Optional

import importlib_resources
from langchain.prompts import BasePromptTemplate, load_prompt
from pydantic import BaseModel, PrivateAttr
import json


class PromptFormatter(BaseModel):
    expertise: str = ""
    initialization: str = ""
    objective: str = "No specific objective, just chatting..."
    definitions: Dict[str, str] = {}
    answer_schema: Dict[str, str] = {}
    conclusion: str = ""
    examples_path: Optional[str] = None
    _loaded_prompt: Optional[BasePromptTemplate] = PrivateAttr(default=None)
    _definition_separators: Optional[List[str]] = PrivateAttr(default=None)
    _answer_schema: Optional[str] = PrivateAttr(default=None)
    _definition_list: Optional[str] = PrivateAttr(default=[None])
    _examples_list: Optional[str] = PrivateAttr(default="")

    def definitions_to_string(
        self, definition_intialization_key: str = "Initialization"
    ) -> str:
        """Generate a formatted definition list

        Args:
            definition_intialization_key: Key of the intialization text. Defaults to "Initialization".

        Raises:
            Warning: If the definition_initializatio_key does not exist in the definition dictionary

        Returns:
            a formatted definition list
        """
        if self.definitions == {}:
            return ""
        self._definition_separators: List[str] = [
            definition_key
            for definition_key in self.definitions.keys()
            if definition_intialization_key not in definition_key
        ]
        definitions: List[str] = []
        if definition_intialization_key not in self.definitions:
            (
                "Warning: The definitions were provided without a valid definition_initializatio_key"
            )
        else:
            definitions.append(f"{self.definitions[definition_intialization_key]}\n")
        for definition in self._definition_separators:
            definitions.append(f"-'{definition}' : {self.definitions[definition]}\n")
        return "".join(definitions)

    def answer_schema_to_string(
        self, definition_intialization_key: str = "Initialization"
    ) -> str:
        """Generate a formatted answer schema

        Args:
            definition_intialization_key: Key of the intialization text. Defaults to "Initialization".

        Raises:
            Warning: If the definition_initializatio_key does not exist in the definition dictionary

        Returns:
            a formatted answer schema
        """
        if self.answer_schema == {}:
            return ""
        schema_list: List[str] = [
            schema_key
            for schema_key in self.answer_schema.keys()
            if definition_intialization_key not in schema_key
        ]
        answer_schema_list: List[str] = []
        if definition_intialization_key not in self.answer_schema:
            (
                "Warning: The answer schema was provided without a valid definition_initialization_key"
            )
        else:
            answer_schema_list.append(
                f"{self.answer_schema[definition_intialization_key]}\n"
            )
        for schema in schema_list:
            answer_schema_list.append(f"{self.answer_schema[schema]}\n")
        return "".join(answer_schema_list)

    def model_post_init(self, __context: Any) -> None:
        if __context is None:
            __context = str(
                importlib_resources.files("xlmexlab")
                / "resources/template"
                / "organic_synthesis_actions_last_template.json"
            )
            loaded_prompt: BasePromptTemplate = load_prompt(__context)
            self._loaded_prompt = loaded_prompt
        else:
            loaded_prompt = load_prompt(__context)
            self._loaded_prompt = loaded_prompt
        definition_list: str = self.definitions_to_string()
        self._definition_list = definition_list
        answer_schema: str = self.answer_schema_to_string()
        self._answer_schema = answer_schema
        if self.examples_path is not None:
            self._examples_list = "\n Here are some examples to help you understand the task:\n"
            with open(self.examples_path, "r") as f:
                examples_dict: Dict[str, Any] = json.load(f)
            for example in examples_dict["examples"]:
                self._examples_list += f"Input: {example['text']}\nExpected Output: {example['output']}\n"
        if self.expertise != "":
            self.expertise = self.expertise + "\n"
        if self.initialization != "":
            self.initialization = self.initialization + "\n"
        if self.objective != "":
            self.objective = self.objective + "\n"

    def format_prompt(
        self,
        context: str = "",
    ) -> str:
        """Generate a formatted prompt.

        Args:
            context: Text to be analysed. Defaults to "".
            prompt_template_path: Path to the prompt template file. Defaults to "".

        Returns:
            a formatted prompt.
        """
        if self._loaded_prompt is None:
            raise AttributeError(
                "There is no prompt loaded, you need to post init the object."
            )
        formatted_prompt: str = self._loaded_prompt.format(
            expertise=self.expertise,
            initialization=self.initialization,
            objective=self.objective,
            context=f"{context}\n",
            definitions=self._definition_list,
            answer_schema=self._answer_schema,
            conclusion=self.conclusion,
            examples=self._examples_list,
        )
        return formatted_prompt


TEMPLATE_REGISTRY: Dict[str, str] = {
    "Llama-2-7b-chat-hf": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama2_default_chat_template.json"
    ),
    "Llama-2-13b-chat-hf": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama2_default_chat_template.json"
    ),
    "Llama-2-70b-chat-hf": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama2_default_chat_template.json"
    ),
    "vicuna-13b-v1.5-16k": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "vicuna_default_chat_template.json"
    ),
    "vicuna-33b-v1.3": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "vicuna_default_chat_template.json"
    ),
    "Mistral-7B-Instruct-v0.1": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_instruct_default_template.json"
    ),
    "Mistral-Small-3.2-24B-Instruct-2506": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_instruct_default_template.json"
    ),
    "Mistral-7B-Instruct-v0.3": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_instruct_default_template.json"
    ),
    "Mistral-7B-v0.1": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_default_template.json"
    ),
    "tora-13b-v1.0": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_default_template.json"
    ),
    "tora-70b-v1.0": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_default_template.json"
    ),
    "open_llama_7b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "open_llama_default_template.json"
    ),
    "open_llama_13b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "open_llama_default_template.json"
    ),
    "mpt-30b-chat": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mpt_default_chat_template.json"
    ),
    "mpt-30b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mpt_default_template.json"
    ),
    "Mistral-7B-Instruct-v0.2": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_default_template.json"
    ),
    "openchat_3.5": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "open_chat_default_template.json"
    ),
    "openchat-3.5-0106": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "open_chat_default_template.json"
    ),
    "openchat-3.6-8b-20240522": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "open_chat_3.6_default_template.json"
    ),
    "Mixtral-8x7B-Instruct-v0.1-AWQ": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_default_template.json"
    ),
    "Mixtral-8x7B-Instruct-v0.1": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "mistral_instruct_default_template.json"
    ),
    "Meta-Llama-3-8B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama3_default_instruct_template.json"
    ),
    "Llama-3.3-70B-Instruct-4bit": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama31_default_instruct_template.json"
    ),
    "Llama-3.3-70B-Instruct-NVFP4": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama31_default_instruct_template.json"
    ),
    "Phi-3-medium-128k-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi_default_instruct_template.json"
    ),
    "Phi-3-medium-4k-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi_default_instruct_template.json"
    ),
    "Phi-3-mini-4k-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi_default_instruct_template.json"
    ),
    "Qwen1.5-7B-Chat": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    "Qwen3-14B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    "Qwen3-4B-Instruct-2507": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    "Qwen3-30B-A3B-Instruct-2507": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    "Qwen3-4B-Thinking-2507": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    "Qwen2.5-7B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template_lnp.json"
    ),
    "Starling-LM-7B-alpha": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "starlinglm_default_template.json"
    ),
    "Kunoichi-DPO-v2-7B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "kunoichi-dp_default_chat_template.json"
    ),
    "zephyr-7b-beta": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "zephyr_default_chat_template.json"
    ),
    "gemma-1.1-7b-it": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "gemma_default_it_template.json"
    ),
    "gemma-1.1-2b-it": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "gemma_default_it_template.json"
    ),
    "gemma-3-12b-it": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "gemma_default_it_template.json"
    ),
    "gemma-3-1b-it": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "gemma_default_it_template.json"
    ),
    "WizardLM-13B-V1.2": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "wizardlm_default_template.json"
    ),
    "granite-7b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite_default_template.json"
    ),
    "granite-3.0-8b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "granite-3.1-8b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "granite-3.1-2b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "granite-3.2-8b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "granite-3.3-8b-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "granite-4.0-tiny-preview": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite3_default_template.json"
    ),
    "merlinite-7b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ibm_granite_default_template.json"
    ),
    "Meta-Llama-3.1-8B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llama31_default_instruct_template.json"
    ),
    "SOLAR-10.7B-Instruct-v1.0": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "solar_default_template.json"
    ),
    "llava-1.5-7b-hf": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llava1.5_default_template.json"
    ),
    "deepseek-llm-7b-chat": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "deepseek_chat_default_instruct_template.json"
    ),
    "DeepSeek-V2-Lite-Chat": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "deepseek_chat_default_instruct_template.json"
    ),
    "Ovis2-4B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ovis_default_instruct_template.json"
    ),
    "Ovis2-16B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ovis_chat_default_instruct_template.json"
    ),
    "Ovis1.6-Gemma2-9B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ovis_chat_default_instruct_template.json"
    ),
    "Ovis1.5-Llama3-8B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ovis_chat_default_instruct_template.json"
    ),
    "Ovis1.5-Llama3-8B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "ovis_chat_default_instruct_template.json"
    ),
    "Phi-3-vision-128k-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi3_default_instruct_template.json"
    ),
    "Phi-3.5-vision-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi3_default_instruct_template.json"
    ),
    "Qwen-VL": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen_vl_template.json"
    ),
    "Qwen2.5-VL-7B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen_default_instruct_template.json"
    ),
    "Qwen2.5-VL-3B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen_default_instruct_template.json"
    ),
    "POINTS-Qwen-2-5-7B-Chat": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen_default_instruct_template.json"
    ),
    "pixtral-12b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "pixtral-12b_default_instruct_template.json"
    ),
    "InternVL2-8B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "internvl_default_instruct_template.json"
    ),
    "InternVL3-8B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "internvl_default_instruct_template.json"
    ),
    "InternVL2_5-4B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "internvl_default_instruct_template.json"
    ),
    "InternVL2_5-8B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "internvl_default_instruct_template.json"
    ),
    "MiniCPM-o-2_6": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "minicpm_default_instruct_template.json"
    ),
    "SmolVLM2-2.2B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "smolvlm2_default_instruct_template.json"
    ),
    "gemma-3-4b-it": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "gemma-3_default_instruct_template.json"
    ),
    "glm-4v-9b": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "glm_default_instruct_template.json"
    ),
    "Idefics3-8B-Llama3": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "idefics3_llama_default_instruct_template.json"
    ),
    "Molmo-7B-D-0924": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "molmo_default_instruct_template.json"
    ),
    "Phi-4-mini-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi4_text_default_instruct_template.json"
    ),
    "Phi-4-multimodal-instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "phi4_default_instruct_template.json"
    ),
    "llava-onevision-qwen2-7b-ov-hf": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "llava-onevision-qwen_default_instruct_template.json"
    ),
    "gpt_4o_aiedu": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "blank_template.json"
    ),
    "Qwen3.5-9B": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "qwen1.5_default_chat_template.json"
    ),
    #qwen1.5_default_chat_template.json"
    #"qwen_vl_template.json"
    "DeepSeek-OCR-2": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "deepseek_ocr_default_template.json"
    ),
    "InternVL3-8B-Instruct": str(
        importlib_resources.files("xlmexlab")
        / "resources/template"
        / "internvl_default_instruct_template.json"
    ),

}