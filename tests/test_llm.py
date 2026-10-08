import importlib_resources
import pytest
import torch

from xlmexlab.llm import ModelLLM


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="The vllm models only run on GPU"
)
def test_ModelLLM():
    name = "facebook/opt-2.7b"
    llm = ModelLLM(model_name=name)
    params_path = str(
        importlib_resources.files("xlmexlab")
        / "resources"
        / "vllm_default_params.json"
    )
    llm.load_model_parameters(params_path)
    llm.vllm_load_model()
    prompt = "Suggest TV show similar to pokemon"
    result = llm.run_single_prompt(prompt)
    assert llm.model_parameters != {}
    assert llm.model is not None
    assert len(result) > 0
