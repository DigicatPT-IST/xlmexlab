import json
from typing import Any, Dict, Optional, Tuple
import requests
import uuid
import os

from langchain_community.llms import VLLM
from PIL import Image, ImageFile
from pydantic import BaseModel
from vllm import LLM, RequestOutput, SamplingParams, TextPrompt
from vllm.sampling_params import BeamSearchParams
import time

from openai import OpenAI, RateLimitError
from xlmexlab.randomization import seed_everything

import json
import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI, RateLimitError
from pydantic import BaseModel

# Fixed, non-configurable endpoint. This is intentionally not exposed as a
# constructor field, config value, or input prompt, so it cannot be changed
# at runtime or by editing the config file.
OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAILLM(BaseModel):
    """LLM wrapper for the OpenAI API.

    Combines:
    - AIeduLLM's pattern of persisting connection secrets (api_key)
      to a local config file so the user is only prompted once.
    - ModelLLM's pattern of loading generation parameters (temperature, seed,
      max_tokens, etc.) from a separate JSON parameter file.
    """

    model_name: str = "gpt-5.2"
    config_path: str = "openai_config.json"
    api_key: Optional[str] = None
    model_parameters: Dict[str, Any] = {}
    client: Any = None

    def model_post_init(self, context) -> None:
        # --- Config file handling (from AIeduLLM) ---
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
                self.api_key = config.get("api_key")
        else:
            self.api_key = input("Enter the OpenAI API key: ")

            config_dict = {"api_key": self.api_key}
            with open(self.config_path, "w") as f:
                json.dump(config_dict, f, indent=4)

        if self.api_key is None:
            raise ValueError("An API key must be set (via config file or input prompt)")

        self.client = OpenAI(api_key=self.api_key, base_url=OPENAI_BASE_URL)

    def load_model_parameters(self, file_path: str) -> None:
        """Load generation parameters (temperature, seed, max_tokens, etc.)
        from a JSON file (mirrors ModelLLM.load_model_parameters).

        Args:
            file_path (str): Path to the json file containing the model parameters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_single_prompt(self, prompt: str) -> str:
        """Run a single prompt against the OpenAI API.

        Args:
            prompt (str): prompt to send to the model

        Returns:
            str: the model's text response
        """
        if self.client is None:
            raise AttributeError("The OpenAI client is not initialized")

        # Only pass through parameters that were actually loaded, so
        # OpenAI's own defaults apply to anything left unset.
        allowed_params = {
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "stop",
        }
        call_kwargs = {
            k: v for k, v in self.model_parameters.items() if k in allowed_params
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **call_kwargs,
            )
        except RateLimitError:
            time.sleep(2)  # small fallback, mirrors AIeduLLM's retry behavior
            return self.run_single_prompt(prompt)

        return response.choices[0].message.content

class AIeduLLM(BaseModel):
    model_name: str = "gpt_4o_aiedu"
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    channel_id: Optional[str] = None

    def model_post_init(self, context):
        if os.path.exists("aiedu_config.json"):
            with open("aiedu_config.json", "r") as f:
                config = json.load(f)
                self.endpoint_url = config.get("endpoint_url")
                self.api_key = config.get("api_key")
                self.channel_id = config.get("channel_id")
        else:
            self.endpoint_url = input("Enter the endpoint URL: ")
            self.api_key = input("Enter the API key: ")
            self.channel_id = input("Enter the channel ID: ")
            config_dict = {
                "endpoint_url": self.endpoint_url,
                "api_key": self.api_key,
                "channel_id": self.channel_id,
            }
            config_json = json.dumps(config_dict, indent=4)
            with open("aiedu_config.json", "w") as f:
                f.write(config_json)

    def extract_dicts_with_type_message(self, response_text: str, response_status: Any) -> str:
        s = response_text       
        results = {}
        depth = 0
        start = None

        for i, ch in enumerate(s):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1

            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    block = s[start:i + 1]
                    if '"type": "message"' in block:
                        results = block
                    start = None
        if results == {}:
            print(f"Response status: {response_status}")
            print(response_text)
            return ""
        else:
            result_json = json.loads(results)
            return result_json["content"]["content"]
    
    def run_single_prompt(self, prompt: str) -> str:
        """Run a single prompt on the loaded model

        Args:
            prompt (str): prompt to the loaded model

        Returns:
            str: a string containing the model response
        """
        if self.endpoint_url is None or self.api_key is None or self.channel_id is None:
            raise ValueError("Endpoint URL, API key, and Channel ID must be set")

        payload = {'channel_id': self.channel_id,
                    'thread_id': str(uuid.uuid4()),
                    'message': prompt,
                    'user_info': '{}'}
        headers = {"x-api-key": self.api_key}
        response = requests.post(self.endpoint_url, data=payload, headers=headers)

        if "Rate limit reached (429)" in response.text:
            time.sleep(2)  # small fallback
            return self.run_single_prompt(prompt)

        else:
            return self.extract_dicts_with_type_message(response.text, response.status_code)


class ModelLLM(BaseModel):
    model_name: str
    model_parameters: Dict[str, Any] = {}
    model_library: str = "vllm"
    model: Any = None
    params: Any = None

    def vllm_load_model(self) -> None:
        """Load a model using vllm library"""
        if self.model_parameters == {}:
            self.model = LLM(model=self.model_name)
        else:
            print("Start Model Loading")
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=self.model_parameters["tensor_parallel_size"],
                dtype=self.model_parameters["dtype"],
                enforce_eager=self.model_parameters["enforce-eager"],
                trust_remote_code=self.model_parameters["trust_remote_code"],
                quantization=self.model_parameters["quantization"],
                max_model_len=self.model_parameters["max_model_len"],
                gpu_memory_utilization=self.model_parameters["gpu_memory_utilization"],
                seed=self.model_parameters["seed"],
            )
            print("Ended Model Loading")
            if self.model_parameters["best_of"] is None:
                self.model_parameters["best_of"] = self.model_parameters["n"]
            if self.model_parameters["use_beam_search"] is False:
                print("sampling_params")
                self.params = SamplingParams(
                    presence_penalty=self.model_parameters["presence_penalty"],
                    frequency_penalty=self.model_parameters["frequency_penalty"],
                    temperature=self.model_parameters["temperature"],
                    top_p=self.model_parameters["top_p"],
                    top_k=self.model_parameters["top_k"],
                    n=self.model_parameters["n"],
                    max_tokens=self.model_parameters["max_new_tokens"],
                    stop=self.model_parameters["stop"],
                    logprobs=self.model_parameters["logprobs"],
                    ignore_eos=self.model_parameters["ignore_eos"],
                    seed=self.model_parameters["seed"],
                )
            else:
                print("beam_search_params")
                self.params = BeamSearchParams(
                    beam_width=self.model_parameters["n"],
                    max_tokens=self.model_parameters["max_new_tokens"],
                    ignore_eos=self.model_parameters["ignore_eos"],
                )

    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path (str): Path to the json file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_single_prompt(self, prompt: str) -> str:
        """Run a single prompt on the loaded model

        Args:
            prompt (str): prompt to the loaded model

        Raises:
            AttributeError: if no model is loaded

        Returns:
            str: a string containing the model response
        """
        if self.model is None:
            raise AttributeError("The LLM model is not loaded")
        if self.model_parameters == {}:
            output: list[RequestOutput] = self.model.generate(prompt)[0]
        elif self.model_parameters["use_beam_search"] is False:
            output = self.model.generate(prompt, self.params)[0]
            generated_text: str = output.outputs[0].text
        return generated_text

        
class ModelVLM(BaseModel):
    model_name: str
    model_parameters: Dict[str, Any] = {}
    model_library: str = "vllm"
    model: Optional[VLLM] = None

    def vllm_load_model(self) -> None:
        """Load a model using vllm library"""
        if self.model_parameters == {}:
            self.model = VLLM(model=self.model_name)
        else:
            self.model = VLLM(
                model=self.model_name,
                best_of=self.model_parameters["best_of"],
                cache=self.model_parameters["cache"],
                callback_manager=self.model_parameters["callback_manager"],
                callbacks=self.model_parameters["callbacks"],
                download_dir=self.model_parameters["download_dir"],
                dtype=self.model_parameters["dtype"],
                frequency_penalty=self.model_parameters["frequency_penalty"],
                ignore_eos=self.model_parameters["ignore_eos"],
                logprobs=self.model_parameters["logprobs"],
                max_new_tokens=self.model_parameters["max_new_tokens"],
                metadata=self.model_parameters["metadata"],
                n=self.model_parameters["n"],
                presence_penalty=self.model_parameters["presence_penalty"],
                stop=self.model_parameters["stop"],
                stop_token_ids=self.model_parameters["stop_token_ids"],
                tags=self.model_parameters["tags"],
                temperature=self.model_parameters["temperature"],
                tensor_parallel_size=self.model_parameters["tensor_parallel_size"],
                top_k=self.model_parameters["top_k"],
                top_p=self.model_parameters["top_p"],
                trust_remote_code=self.model_parameters["trust_remote_code"],
                image_input_type="pixel_values",
                image_token_id=self.model_parameters["image_token_id"],
                image_input_shape=self.model_parameters["image_input_shape"],
                image_feature_size=self.model_parameters["image_feature_size"],
                vllm_kwargs={
                    "gpu_memory_utilization": self.model_parameters[
                        "gpu_memory_utilization"
                    ],
                    "seed": self.model_parameters["seed"],
                    "enforce_eager": self.model_parameters["enforce-eager"],
                    "quantization": self.model_parameters["quantization"],
                    "max_model_len": self.model_parameters["max_model_len"],
                },
            )

    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path: Path to the json file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_image_single_prompt(self, prompt: str, image_path: str) -> str:
        """Run a single prompt on the loaded vision language model

        Args:
            prompt (str): prompt to the loaded model
            image_path (str): file path for the image

        Returns:
            str: a string containing the model response
        """
        pil_image: ImageFile = Image.open(image_path)
        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]
        outputs: list[RequestOutput] = self.model.generate(new_prompt)
        final_response: str = ""
        for o in outputs:
            final_response += o[1][0][0].text
            break
        return final_response

    def run_image_single_prompt_rescale(
        self, prompt: str, image_path: str, scale: float = 1.0
    ) -> str:
        """Run a single prompt on the loaded vision language model with the option to rescale the image

        Args:
            prompt (str): prompt to the loaded model
            image_path (str): file path for the image
            scale (float, optional): rescale factor to use. Defaults to 1.0.

        Returns:
            str: a string containing the model response
        """
        pil_image: ImageFile = Image.open(image_path)
        if scale < 1.0:
            new_size: Tuple[int, int] = (
                int(pil_image.width * scale),
                int(pil_image.height * scale),
            )
            pil_image = pil_image.resize(new_size, Image.BILINEAR)

        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]

        outputs: list[RequestOutput] = self.model.generate(prompts=new_prompt)
        final_response: str = ""
        for o in outputs:
            completion += o[1][0][0]
            final_response += completion.text
            finish_reason = completion.finish_reason
            print(f"Finish reason: {finish_reason}")
            break
        return final_response

class ModelVLM2(BaseModel):
    model_name: str
    model_parameters: Dict[str, Any] = {}
    model_library: str = "vllm"
    model: Optional[VLLM] = None
    params: Any = None

    def vllm_load_model(self) -> None:
        """Load a model using vllm library"""
        if self.model_parameters == {}:
            self.model = VLLM(model=self.model_name)
        else:
            self.model = VLLM(
                model=self.model_name,
                tensor_parallel_size=self.model_parameters["tensor_parallel_size"],
                dtype=self.model_parameters["dtype"],
                enforce_eager=self.model_parameters["enforce-eager"],
                trust_remote_code=self.model_parameters["trust_remote_code"],
                quantization=self.model_parameters["quantization"],
                max_model_len=self.model_parameters["max_model_len"],
                gpu_memory_utilization=self.model_parameters["gpu_memory_utilization"],
                seed=self.model_parameters["seed"],
                max_seq_len_to_capture=self.model_parameters["max_seq_len_to_capture"]
            )

            print("Ended Model Loading")
            if self.model_parameters["best_of"] is None:
                self.model_parameters["best_of"] = self.model_parameters["n"]
            if self.model_parameters["use_beam_search"] is False:
                print("sampling_params")
                self.params = SamplingParams(
                    presence_penalty=self.model_parameters["presence_penalty"],
                    frequency_penalty=self.model_parameters["frequency_penalty"],
                    temperature=self.model_parameters["temperature"],
                    top_p=self.model_parameters["top_p"],
                    top_k=self.model_parameters["top_k"],
                    n=self.model_parameters["n"],
                    max_tokens=self.model_parameters["max_new_tokens"],
                    stop=self.model_parameters["stop"],
                    logprobs=self.model_parameters["logprobs"],
                    ignore_eos=self.model_parameters["ignore_eos"],
                    seed=self.model_parameters["seed"],
                )
            else:
                print("beam_search_params")
                self.params = BeamSearchParams(
                    beam_width=self.model_parameters["n"],
                    max_tokens=self.model_parameters["max_new_tokens"],
                    ignore_eos=self.model_parameters["ignore_eos"],
                )
                
                
    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path: Path to the json file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_image_single_prompt(self, prompt: str, image_path: str) -> str:
        """Run a single prompt on the loaded vision language model

        Args:
            prompt (str): prompt to the loaded model
            image_path (str): file path for the image

        Returns:
            str: a string containing the model response
        """
        pil_image: ImageFile = Image.open(image_path)
        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]
        outputs: list[RequestOutput] = self.model.generate(new_prompt)
        final_response: str = ""
        for o in outputs:
            final_response = o.outputs[0].text
            break
        return final_response

    def run_image_single_prompt_rescale(
        self, prompt: str, image_path: str, scale: float = 1.0
    ) -> str:
        """Run a single prompt on the loaded vision language model with the option to rescale the image

        Args:
            prompt (str): prompt to the loaded model
            image_path (str): file path for the image
            scale (float, optional): rescale factor to use. Defaults to 1.0.

        Returns:
            str: a string containing the model response
        """
        pil_image: ImageFile = Image.open(image_path)
        if scale < 1.0:
            new_size: Tuple[int, int] = (
                int(pil_image.width * scale),
                int(pil_image.height * scale),
            )
            pil_image = pil_image.resize(new_size, Image.BILINEAR)

        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]

        outputs: list[RequestOutput] = self.model.generate(prompts = new_prompt, sampling_params = self.params)
        print(outputs)
        final_response: str = ""
        for o in outputs:
            final_response += o[1][0][0].text
            break
        return final_response
