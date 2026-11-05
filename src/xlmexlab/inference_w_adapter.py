import json
import torch
import importlib
from pydantic import BaseModel, PrivateAttr
from typing import Any, Dict, List, Optional

from xlmexlab.parser import ImageParser


class ModelWithAdapter(BaseModel):
    base_model_path: str
    imports_config_path: str
    adapter_path: Optional[str] = None

    _device: torch.device = PrivateAttr()
    _model: Any = PrivateAttr()
    _processor: Any = PrivateAttr()
    _process_vision_info: Any = PrivateAttr()
    _ModelClass: Any = PrivateAttr()
    _ProcessorClass: Any = PrivateAttr()
    _imports_config: Dict = PrivateAttr()
    _image_parser: Optional[ImageParser] = PrivateAttr(default=None)


    def model_post_init(self, __context: Any) -> None:
        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load imports configuration
        with open(self.imports_config_path) as f:
            self._imports_config = json.load(f)

        # Dynamically load classes/functions
        self._ModelClass = self._load_class(self._imports_config["model_class"])
        self._ProcessorClass = self._load_class(self._imports_config["processor_class"])
        self._process_vision_info = self._load_function(self._imports_config["vision_utils"])

        torch_dtype = torch.float16

        # Load model
        self._model = self._ModelClass.from_pretrained(
            self.base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.generation_config.temperature = None

        # Load processor
        self._processor = self._ProcessorClass.from_pretrained(self.base_model_path)

        # Load adapter if provided
        if self.adapter_path:
            self._model.load_adapter(self.adapter_path)

    def _load_class(self, dotted_path: str) -> Any:
        module_name, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _load_function(self, dotted_path: str) -> Any:
        module_name, func_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def generate(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Apply chat template
        print("STEP 1 - chat template")
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process images/videos

        print("STEP 2 - process vision")
        image_inputs, video_inputs = self._process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Generate output
        print("STEP 4 - generate")
        with torch.no_grad():
            print("START GENERATE")
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        print("END GENERATE")  

        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output = self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._image_parser = ImageParser()
        self._image_parser.parse(output[0])
        parsed_output = self._image_parser.get_data_dict()
        print (parsed_output)
        return parsed_output