import json
from typing import Any, Dict, Optional, Tuple

from langchain_community.llms import VLLM
from PIL import Image, ImageFile
from pydantic import BaseModel
from vllm import LLM, RequestOutput, SamplingParams, TextPrompt
from vllm.sampling_params import BeamSearchParams

from xlmexlab.randomization import seed_everything


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
            final_response += o[1][0][0].text
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
