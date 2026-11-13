from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from pydantic import BaseModel, PrivateAttr, validator
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class EmbeddingExtractor(BaseModel):
    _device: str = PrivateAttr()
    _model: Qwen2_5_VLForConditionalGeneration = PrivateAttr()
    _processor: AutoProcessor = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        model_id: str = "/projects/F202407080CPCAA1/Lea/models/Qwen2.5-VL-7B-Instruct"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="eager",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._model.to(self._device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(model_id)

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract the embeddings from an image

        Args:
            image_path (str): file path for the image file

        Returns:
            np.ndarray: a numpy array conatining all the image embeddings
        """
        img: ImageFile = Image.open(image_path).convert("RGB")
        inputs: Any = self._processor.image_processor(images=img, return_tensors="pt")
        pixel_values: Any = inputs["pixel_values"].to(self._device)
        grid_thw: Any = inputs["image_grid_thw"].to(self._device)

        with torch.no_grad():
            vision_outputs: Any = self._model.visual(pixel_values, grid_thw)
            visual_embeds: Any = vision_outputs.squeeze(0).cpu()
            pooled = visual_embeds[0]
            #pooled = torch.cat([visual_embeds.mean(dim=0), visual_embeds.max(dim=0).values])
            #pooled = visual_embeds.mean(dim=0)
            #pooled,_ = visual_embeds.max(dim=0)
            normalized = pooled / pooled.norm(p=2)
        return normalized.numpy()
