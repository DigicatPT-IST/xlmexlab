import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
import copy
import click
import importlib_resources
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, PrivateAttr, validator

from html_table_extractor.extractor import Extractor

from xlmexlab.actions import (
    ACTION_REGISTRY,
    AQUEOUS_REGISTRY,
    BANNED_CHEMICALS_REGISTRY,
    BANNED_TRANSFER_REGISTRY,
    CENTRIFUGATION_REGISTRY,
    CUSTOM_ACTION_REGISTRY,
    ELEMENTARY_ACTION_REGISTRY,
    EVAPORATION_REGISTRY,
    FILTER_REGISTRY,
    FILTRATE_REGISTRY,
    MATERIAL_ACTION_REGISTRY,
    MICROWAVE_REGISTRY,
    ORGANIC_ACTION_REGISTRY,
    ORGANIC_REGISTRY,
    PH,
    PH_REGISTRY,
    PISTACHIO_ACTION_REGISTRY,
    PRECIPITATE_REGISTRY,
    SAC_ACTION_REGISTRY,
    Add,
    CollectLayer,
    Crystallization,
    DrySolution,
    Filter,
    MakeSolution,
    NewSolution,
    PhaseSeparation,
    Quench,
    ReduceTemperature,
    Separate,
    SetTemperature,
    Stir,
    ThermalTreatment,
    Transfer,
    Wash,
)
from xlmexlab.llm import ModelLLM, ModelVLM2, AIeduLLM, OpenAILLM
from xlmexlab.parser import (
    MOLAR_RATIO_REGISTRY,
    ActionsParser,
    ComplexConditions,
    ComplexParametersParser,
    Conditions,
    EquationFinder,
    ImageParser,
    KeywordSearching,
    ListParametersParser,
    MolarRatioFinder,
    NumberFinder,
    ParametersParser,
    SchemaParser,
    TableParser,
    VariableFinder,
    LaTeXTableParser #Adicionado
)
from xlmexlab.prompt import PromptFormatter


class ActionExtractorFromText(BaseModel):
    actions_type: str = "All"
    action_prompt_template_path: Optional[str] = None
    chemical_prompt_template_path: Optional[str] = None
    action_prompt_schema_path: Optional[str] = None
    chemical_prompt_schema_path: Optional[str] = None
    wash_chemical_prompt_schema_path: Optional[str] = None
    add_chemical_prompt_schema_path: Optional[str] = None
    solution_chemical_prompt_schema_path: Optional[str] = (None,)
    transfer_prompt_schema_path: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    elementar_actions: bool = False
    examples_path:  Optional[str] = None
    post_processing: bool = True
    banned_chemicals: bool = True
    inference_system: str = "vllm"
    _action_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _wash_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _add_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _solution_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _transfer_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)
    _action_parser: Optional[ActionsParser] = PrivateAttr(default=None)
    _condition_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _complex_parser: Optional[ComplexParametersParser] = PrivateAttr(default=None)
    _quantity_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _schema_parser: Optional[SchemaParser] = PrivateAttr(default=None)
    _transfer_schema_parser: Optional[SchemaParser] = PrivateAttr(default=None)
    _filtrate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _precipitate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _filter_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _centri_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _evaporation_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _aqueous_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _organic_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _microwave_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _ph_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _banned_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _transfer_banned_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _molar_ratio_parser: Optional[MolarRatioFinder] = PrivateAttr(default=None)
    _action_dict: Dict[str, Any] = PrivateAttr(default=ACTION_REGISTRY)

    def model_post_init(self, __context: Any) -> None:
        if self.actions_type == "pistachio":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "organic_synthesis_actions_schema.json"
                )
            self._action_dict = PISTACHIO_ACTION_REGISTRY
            ph_keywords: List[str] = PH_REGISTRY
            atributes: List[str] = [
                "type",
                "name",
                "dropwise",
                "concentration",
                "amount",
            ]
        elif self.actions_type == "organic":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "organic_synthesis_actions_schema.json"
                )
            self._action_dict = ORGANIC_ACTION_REGISTRY
            ph_keywords = ["&^%#@&#@(*)"]
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        elif self.actions_type == "materials":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "material_synthesis_actions_schema.json"
                )
            self._action_dict = MATERIAL_ACTION_REGISTRY
            ph_keywords = PH_REGISTRY
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        elif self.actions_type == "sac":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "sac_synthesis_actions_schema.json"
                )
            self._action_dict = SAC_ACTION_REGISTRY
            ph_keywords = PH_REGISTRY
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        elif self.actions_type == "custom":
            if self.action_prompt_schema_path is None:
                raise AttributeError(
                    "You need to give a schema file for custom actions"
                )
            self._action_dict = CUSTOM_ACTION_REGISTRY
            ph_keywords = PH_REGISTRY
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        elif self.actions_type == "elementary":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("xlmexlab")
                    / "resources/schemas"
                    / "material_synthesis_actions_schema.json"
                )
            self._action_dict = ELEMENTARY_ACTION_REGISTRY
            ph_keywords = PH_REGISTRY
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        if self.chemical_prompt_schema_path is None:
            self.chemical_prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "chemicals_from_actions_schema.json"
            )
        with open(self.chemical_prompt_schema_path, "r") as f:
            chemical_prompt_dict: Dict[str, Any] = json.load(f)
        self._chemical_prompt = PromptFormatter(**chemical_prompt_dict)
        self._chemical_prompt.model_post_init(self.chemical_prompt_template_path)
        if self.wash_chemical_prompt_schema_path is None:
            self._wash_chemical_prompt = self._chemical_prompt
        else:
            with open(self.wash_chemical_prompt_schema_path, "r") as f:
                wash_chemical_prompt_dict: Dict[str, Any] = json.load(f)
            self._wash_chemical_prompt = PromptFormatter(**wash_chemical_prompt_dict)
            self._wash_chemical_prompt.model_post_init(
                self.chemical_prompt_template_path
            )
        if self.add_chemical_prompt_schema_path is None:
            self._add_chemical_prompt = self._chemical_prompt
        else:
            with open(self.add_chemical_prompt_schema_path, "r") as f:
                add_chemical_prompt_dict: Dict[str, Any] = json.load(f)
            self._add_chemical_prompt = PromptFormatter(**add_chemical_prompt_dict)
            self._add_chemical_prompt.model_post_init(
                self.chemical_prompt_template_path
            )
        if self.solution_chemical_prompt_schema_path is None:
            self._solution_chemical_prompt = self._chemical_prompt
        else:
            with open(self.solution_chemical_prompt_schema_path, "r") as f:
                solution_chemical_prompt_dict: Dict[str, Any] = json.load(f)
            self._solution_chemical_prompt = PromptFormatter(
                **solution_chemical_prompt_dict
            )
            self._solution_chemical_prompt.model_post_init(
                self.chemical_prompt_template_path
            )
        self.transfer_prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/schemas"
            / "transfer_schema.json"
        )
        with open(self.transfer_prompt_schema_path, "r") as f:
            transfer_prompt_dict: Dict[str, Any] = json.load(f)
        self._transfer_prompt = PromptFormatter(**transfer_prompt_dict)
        self._transfer_prompt.model_post_init(self.chemical_prompt_template_path)
        if self.llm_model_parameters_path is None:
            llm_param_path: str = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "vllm_default_params.json"
            )
        else:
            llm_param_path = self.llm_model_parameters_path
        if self.llm_model_name is None:
            self._llm_model = ModelLLM(model_name="microsoft/Phi-3-medium-4k-instruct")
        else:
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
        #print(self.action_prompt_schema_path)
        with open(self.action_prompt_schema_path, "r") as f:
            action_prompt_dict: Dict[str, Any] = json.load(f)
        #print(action_prompt_dict)
        self._action_prompt = PromptFormatter(**action_prompt_dict, examples_path = self.examples_path)
        self._action_prompt.model_post_init(self.action_prompt_template_path)
        #print(self._action_prompt)
        if self.llm_model_name == "gpt_4o_aiedu" and self.inference_system == "iaedu":
            self._llm_model = AIeduLLM()
        elif self.inference_system == "vllm":
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
            self._llm_model.load_model_parameters(llm_param_path)
            self._llm_model.vllm_load_model()
        elif self.inference_system == "openai":
            self._llm_model = OpenAILLM(model_name=self.llm_model_name)
            self._llm_model.load_model_parameters(llm_param_path)
        self._action_parser = ActionsParser(
            type=self.actions_type,
            separators=self._action_prompt._definition_separators,
        )
        self._condition_parser = ParametersParser(convert_units=False, amount=False)
        self._quantity_parser = ParametersParser(
            convert_units=False,
            time=False,
            temperature=False,
            pressure=False,
            atmosphere=False,
            size=False,
        )
        transfer_atributes: List[str] = ["recipient_name"]
        self._ph_parser = KeywordSearching(keywords_list=ph_keywords)
        self._complex_parser = ComplexParametersParser()
        self._evaporation_parser = KeywordSearching(keywords_list=EVAPORATION_REGISTRY)
        self._aqueous_parser = KeywordSearching(keywords_list=AQUEOUS_REGISTRY)
        self._organic_parser = KeywordSearching(keywords_list=ORGANIC_REGISTRY)
        self._centri_parser = KeywordSearching(keywords_list=CENTRIFUGATION_REGISTRY)
        self._filter_parser = KeywordSearching(keywords_list=FILTER_REGISTRY)
        self._transfer_schema_parser = SchemaParser(atributes_list=transfer_atributes)
        self._schema_parser = SchemaParser(atributes_list=atributes)
        self._transfer_banned_parser = KeywordSearching(
            keywords_list=BANNED_TRANSFER_REGISTRY
        )
        self._filtrate_parser = KeywordSearching(keywords_list=FILTRATE_REGISTRY)
        if self.banned_chemicals:
            self._banned_parser = KeywordSearching(
                keywords_list=BANNED_CHEMICALS_REGISTRY
            )
        else:
            self._banned_parser = KeywordSearching(keywords_list=["ai&/(=)"])
        self._precipitate_parser = KeywordSearching(keywords_list=PRECIPITATE_REGISTRY)
        self._microwave_parser = KeywordSearching(keywords_list=MICROWAVE_REGISTRY)
        self._molar_ratio_parser = MolarRatioFinder(chemicals_list=MOLAR_RATIO_REGISTRY)

    @staticmethod
    def delete_dict_keys(
        action: Dict[str, Any], keys_list: List[str]
    ) -> Dict[str, Any]:
        """Delete content keys from an action dictionary

        Args:
            action (Dict[str, Any]): action dictionary to transform
            keys_list (List[str]): keys to remove

        Returns:
            Dict[str, Any]: dictionary without the removed keys
        """
        for key in keys_list:
            try:
                del action["content"][key]
            except KeyError:
                pass
        return action

    @staticmethod
    def delete_material_dict_keys(
        dict_info: Dict[str, Any], keys_list: List[str]
    ) -> Dict[str, Any]:
        """Delete keys from a material dictionary

        Args:
            dict_info (Dict[str, Any]): material dictionary to transform
            keys_list (List[str]): keys to remove

        Returns:
            Dict[str, Any]: dictionary without the removed keys
        """
        for key in keys_list:
            try:
                del dict_info[key]
            except KeyError:
                pass
        return dict_info

    @staticmethod
    def empty_action(action: Dict[str, Any]) -> bool:
        """Verify the the action is empty. Being empty means that no content data was found for the action

        Args:
            action (Dict[str, Any]): action to analyse

        Returns:
            bool: True if the action is empty, false otherwise
        """
        content: Dict[str, Any] = action["content"]
        list_of_keys: List[str] = list(content.keys())
        is_empty: bool = True
        ignore_set: Dict[str, Any] = {"dropwise", "repetitions"}
        for key in list_of_keys:
            if key == "meterials":
                if content[key] != []:
                    is_empty = False
                    break
            elif key in ignore_set:
                pass
            else:
                if content[key] is not None:
                    is_empty = False
                    break
        return is_empty

    @staticmethod
    def eliminate_empty_sequence(
        action_list: List[Dict[str, Any]], threshold: int
    ) -> List[Dict[str, Any]]:
        """Remove seauences of empty actions from an action list. Being empty means that no content data was found for the action

        Args:
            action_list (List[Dict[str, Any]]): list of action in the full sequence
            threshold (int): minimum amount of empty action to remove the action sequence

        Returns:
            List[Dict[str, Any]]: The update action list without the empty action sequence
        """
        ignore_set: set[str] = {
            "CollectLayer",
            "Concentrate",
            "Filter",
            "PhaseSeparation",
            "Purify",
        }
        empty_sequence: int = 0
        i: int = 0
        for action in action_list:
            if action["action"] in ignore_set:
                empty_sequence = 0
            elif ActionExtractorFromText.empty_action(action):
                empty_sequence = empty_sequence + 1
                if empty_sequence == threshold:
                    action_list = (
                        action_list[: i - threshold + 1] + action_list[i + 1 :]
                    )
                    i = i - threshold
                elif empty_sequence > threshold:
                    del action_list[i]
                    i = i - 1
            else:
                empty_sequence = 0
            i = i + 1
        return action_list

    @staticmethod
    def correct_action_list(
        action_dict_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Correct an action list based on a set of action rules designed for zeolite synthesis

        Args:
            action_dict_list (List[Dict[str, Any]]): pre processed action sequence

        Returns:
            List[Dict[str, Any]]: The post processed action sequenced
        """
        new_action_list: List[Dict[str, Any]] = []
        initial_temp: Optional[str] = None
        current_atmosphere: Optional[str] = None
        add_new_solution: bool = True
        i_new_solution: int = 0
        i: int = 0
        for action in action_dict_list:
            action_name: str = action["action"]
            content: Dict[str, Any] = action["content"]
            try:
                new_temp: str = content["temperature"]
                if action_name in ["ThermalTreatment", "Dry", "Crystallization"]:
                    pass
                elif new_temp in set(["heat", "cool"]):
                    initial_temp = new_temp
                    new_action_list.append(
                        {
                            "action": "SetTemperature",
                            "content": {
                                "temperature": new_temp,
                                "microwave": False,
                                "heat_ramp": None,
                            },
                        }
                    )
                    del content["temperature"]
                elif new_temp != initial_temp and new_temp is not None:
                    initial_temp = new_temp
                    new_action_list.append(
                        {
                            "action": "SetTemperature",
                            "content": {
                                "temperature": new_temp,
                                "microwave": False,
                                "heat_ramp": None,
                            },
                        }
                    )
                    del content["temperature"]
                else:
                    del content["temperature"]
            except KeyError:
                pass
            if action_name == "Add":
                if add_new_solution is True:
                    add_new_solution = False
                    new_action_list.insert(
                        i_new_solution,
                        NewSolution(action_name="NewSolution").generate_dict(),
                    )
                if (
                    content["atmosphere"] != []
                    and content["atmosphere"] != current_atmosphere
                ):
                    current_atmosphere = content["atmosphere"]
                    new_action_list.append(
                        {
                            "action": "SetAtmosphere",
                            "content": {
                                "atmosphere": content["atmosphere"],
                                "pressure": None,
                                "flow_rate": None,
                            },
                        }
                    )
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(action, ["atmosphere"])
                )
            elif action["action"] == "NewSolution":
                add_new_solution = False
                initial_temp = None
                if i == len(action_dict_list) - 1:
                    pass
                elif action_dict_list[i + 1]["action"] not in set(
                    ["Add", "SetTemperature", "SetAtmosphere", "Repetition"]
                ):
                    pass
                else:
                    new_action_list.append(action)
            elif action_name == "Wash":
                if content["repetitions"] > 1:
                    repeat_action: Dict[str, Any] = {
                        "action": "Repeat",
                        "content": {"amount": content["repetitions"]},
                    }
                    new_action_list.append(
                        ActionExtractorFromText.delete_dict_keys(
                            action, ["duration", "repetitions"]
                        )
                    )
                    new_action_list.append(repeat_action)
                else:
                    new_action_list.append(
                        ActionExtractorFromText.delete_dict_keys(
                            action, ["duration", "repetitions"]
                        )
                    )
            elif action_name == "Repeat" and len(new_action_list) > 0:
                pre_action: Dict[str, Any] = new_action_list[-1]
                amount: float = float(action["content"]["amount"])
                if pre_action["action"] == "Repeat":
                    new_amount = float(pre_action["content"]["amount"])
                    if amount < new_amount:
                        new_action_list[-1] = action
                else:
                    new_action_list.append(action)
            elif action_name == "SetTemperature":
                if content["duration"] is not None:
                    if new_temp is not None:
                        new_action_list[-1] = {
                            "action": "Crystallization",
                            "content": {
                                "temperature": new_temp,
                                "duration": content["duration"],
                                "pressure": content["pressure"],
                                "stirring_speed": content["stirring_speed"],
                                "microwave": content["microwave"],
                            },
                        }
                    else:
                        new_action_list.append(
                            {
                                "action": "Crystallization",
                                "content": {
                                    "temperature": new_temp,
                                    "duration": content["duration"],
                                    "pressure": content["pressure"],
                                    "stirring_speed": content["stirring_speed"],
                                    "microwave": content["microwave"],
                                },
                            }
                        )
            elif action_name in set(["Wash", "Separate"]):
                add_new_solution = True
                new_action_list.append(action)
                i_new_solution = len(new_action_list)
            elif action_name in set(["Crystallization", "Dry", "ThermalTreatment"]):
                new_action_list.append(action)
                i_new_solution = len(new_action_list)
            elif action_name == "Sonicate":
                new_action_list.append(
                    {
                        "action": "Stir",
                        "content": {
                            "duration": content["duration"],
                            "stirring_speed": "sonicate",
                        },
                    }
                )
                i_new_solution = len(new_action_list)
            elif action_name == "Stir":
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["atmosphere", "pressure"]
                    )
                )
            else:
                new_action_list.append(action)
            i += 1
        if len(new_action_list) > 1:
            last_action: Dict[str, Any] = new_action_list[-1]
            second_last_action: Dict[str, Any] = new_action_list[-2]
            if last_action["action"] == "Wait" and second_last_action["action"] in set(
                ["Dry", "Wait", "ThermalTreatment", "Wash", "Separate"]
            ):
                del new_action_list[-1]
        return new_action_list

    @staticmethod
    def correct_organic_action_list(
        action_dict_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Correct an action list based on a set of action rules designed for organic synthesis

        Args:
            action_dict_list (List[Dict[str, Any]]): pre processed action sequence

        Returns:
            List[Dict[str, Any]]: The post processed action sequence
        """
        new_action_list: List[Dict[str, Any]] = []
        initial_temp: Optional[str] = None
        for action in action_dict_list:
            action_name: str = action["action"]
            content: Dict[str, Any] = action["content"]
            try:
                new_temp: str = content["temperature"]
                if new_temp is None:
                    pass
                elif new_temp.lower() in ["ice-bath", "ice bath"]:
                    new_temp = "0 °C"
                elif new_temp.lower() == "cool":
                    new_temp = "room temperature"
                if new_temp != initial_temp and new_temp is not None:
                    initial_temp = new_temp
                    new_action_list.append(
                        {
                            "action": "SetTemperature",
                            "content": {"temperature": new_temp},
                        }
                    )
                del content["temperature"]
            except KeyError:
                pass
            try:
                atmosphere: List[str] = content["atmosphere"]
                if len(atmosphere) > 0 and type(atmosphere) is list:
                    action["content"][atmosphere] = atmosphere[0]
                elif type(atmosphere) is str:
                    pass
                else:
                    action["content"][atmosphere] = None
            except KeyError:
                pass
            if action_name == "Partition":
                if content["material_1"] is None and content["material_2"] is None:
                    pass
                elif content["material_1"] is None:
                    material_1: Dict[str, Any] = content["material_2"]
                    content["material_1"] = material_1
                    content["material_2"] = None
                    action["content"]["material1"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material1"], ["concentration"]
                        )
                    )
                elif content["material_2"] is None:
                    pass
                else:
                    action["content"]["material1"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material1"], ["concentration"]
                        )
                    )
                    action["content"]["material2"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material2"], ["concentration"]
                        )
                    )
                    materials_list: List[Dict[str, Any]] = [
                        content["material_1"],
                        content["material_2"],
                    ]
                    sorted_material_list: List[Dict[str, Any]] = sorted(
                        materials_list, key=lambda d: d["name"]
                    )
                    content["material_1"] = sorted_material_list[0]
                    content["material_2"] = sorted_material_list[1]
                new_action_list.append(action)
            elif action_name == "Add":
                action["content"]["material"] = (
                    ActionExtractorFromText.delete_material_dict_keys(
                        content["material"], ["concentration"]
                    )
                )
                if content["material"]["name"] == "SLN":
                    pass
                elif content["ph"] is not None:
                    new_action_list.append(
                        {
                            "action": "PH",
                            "content": {
                                "ph": content["ph"],
                                "material": content["material"],
                                "dropwise": content["dropwise"],
                            },
                        }
                    )
                else:
                    new_action_list.append(
                        ActionExtractorFromText.delete_dict_keys(action, ["ph"])
                    )
            elif action_name == "Wash":
                action["content"]["material"] = (
                    ActionExtractorFromText.delete_material_dict_keys(
                        content["material"], ["concentration"]
                    )
                )
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["duration", "method"]
                    )
                )
            elif action_name in ["CollectLayer", "Yield"]:
                pass
            elif action_name == "SetTemperature":
                pass
            elif action_name == "Stir":
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["stirring_speed", "pressure"]
                    )
                )
            else:
                new_action_list.append(action)
        return new_action_list

    @staticmethod
    def correct_pistachio_action_list(
        action_dict_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Correct an action list based on a set of action rules designed for organic synthesis and the pistachio dataset structure

        Args:
            action_dict_list (List[Dict[str, Any]]): pre processed action sequence

        Returns:
            List[Dict[str, Any]]: The post processed action sequence
        """
        new_action_list: List[Dict[str, Any]] = []
        initial_temp: Optional[str] = None
        for action in action_dict_list:
            action_name: str = action["action"]
            content: Dict[str, Any] = action["content"]
            try:
                new_temp: str = content["temperature"]
                if new_temp is None:
                    pass
                elif new_temp.lower() in ["ice-bath", "ice bath"]:
                    new_temp = "0 °C"
                elif new_temp.lower() == "cool":
                    new_temp = "room temperature"
                if new_temp is None:
                    pass
                elif new_temp.lower() == "reflux":
                    pass
                elif new_temp != initial_temp:
                    initial_temp = new_temp
                    new_action_list.append(
                        {
                            "action": "SetTemperature",
                            "content": {"temperature": new_temp},
                        }
                    )
                del content["temperature"]
            except KeyError:
                pass
            try:
                atmosphere: List[str] = content["atmosphere"]
                if len(atmosphere) > 0 and type(atmosphere) is list:
                    action["content"]["atmosphere"] = atmosphere[0]
                    content["atmosphere"] = atmosphere[0]
                elif type(atmosphere) is str:
                    pass
                else:
                    action["content"]["atmosphere"] = None
                    content["atmosphere"] = None
            except KeyError:
                pass
            if action_name == "MakeSolution":
                chemical_list = content["materials"]
                for chemical in chemical_list:
                    new_add = {
                        "action": "Add",
                        "content": {
                            "material": chemical,
                            "dropwise": False,
                            "atmosphere": None,
                            "duration": None,
                        },
                    }
                    new_action_list.append(new_add)
            elif action_name == "Partition":
                if content["material_1"] is None and content["material_2"] is None:
                    pass
                elif content["material_1"] is None:
                    material_1: Dict[str, Any] = content["material_2"]
                    content["material_1"] = material_1
                    content["material_2"] = None
                    action["content"]["material_1"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material_1"], ["concentration"]
                        )
                    )
                elif content["material_2"] is None:
                    pass
                else:
                    action["content"]["material_1"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material_1"], ["concentration"]
                        )
                    )
                    action["content"]["material_2"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material_2"], ["concentration"]
                        )
                    )
                    materials_list: List[Dict[str, Any]] = [
                        content["material_1"],
                        content["material_2"],
                    ]
                    sorted_material_list: List[Dict[str, Any]] = sorted(
                        materials_list, key=lambda d: d["name"]
                    )
                    content["material_1"] = sorted_material_list[0]
                    content["material_2"] = sorted_material_list[1]
                new_action_list.append(action)
            elif action_name == "Add":
                if action["content"]["material"] is not None:
                    action["content"]["material"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material"], ["concentration"]
                        )
                    )
                    if content["material"]["name"] == "SLN":
                        pass
                    else:
                        new_action_list.append(
                            ActionExtractorFromText.delete_dict_keys(action, ["ph"])
                        )
            elif action_name == "Extract":
                if action["content"]["solvent"] is not None:
                    action["content"]["solvent"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["solvent"], ["concentration"]
                        )
                    )
                new_action_list.append(action)
            elif action_name == "Triturate":
                if action["content"]["solvent"] is not None:
                    action["content"]["solvent"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["solvent"], ["concentration"]
                        )
                    )
                new_action_list.append(action)
            elif action_name == "Recrystallize":
                if action["content"]["solvent"] is not None:
                    action["content"]["solvent"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["solvent"], ["concentration"]
                        )
                    )
                new_action_list.append(action)
            elif action_name == "Quench":
                if action["content"]["material"] is not None:
                    action["content"]["material"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material"], ["concentration"]
                        )
                    )
                new_action_list.append(action)
            elif action_name == "PH":
                if action["content"]["material"] is not None:
                    action["content"]["material"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material"], ["concentration"]
                        )
                    )
                    new_action = {
                        "action": "Add",
                        "content": {
                            "material": content["material"],
                            "dropwise": content["dropwise"],
                            "atmosphere": None,
                            "duration": None,
                        },
                    }
                    new_action_list.append(new_action)
            elif action_name == "Wash":
                if action["content"]["material"] is not None:
                    action["content"]["material"] = (
                        ActionExtractorFromText.delete_material_dict_keys(
                            content["material"], ["concentration"]
                        )
                    )
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["duration", "method"]
                    )
                )
            elif action_name in ["CollectLayer", "Yield"]:
                pass
            elif action_name == "Centrifuge":
                new_action_list.append({"action": "PhaseSeparation", "content": {}})
            elif action_name == "Filter":
                if content["phase_to_keep"] is None:
                    action["content"]["phase_to_keep"] = "filtrate"
            elif action_name == "Stir":
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["stirring_speed", "pressure"]
                    )
                )
            elif action_name == "SetTemperature":
                if new_temp is None:
                    pass
                elif new_temp.lower() == "reflux":
                    if content["atmosphere"] is None:
                        atmosphere = content["atmosphere"]
                    else:
                        atmosphere = None
                    new_action_list.append(
                        {
                            "action": "Reflux",
                            "content": {
                                "duration": content["duration"],
                                "dean_stark": False,
                                "atmosphere": atmosphere,
                            },
                        }
                    )
                elif content["duration"] is not None:
                    new_action_list.append(
                        {"action": "Wait", "content": {"duration": content["duration"]}}
                    )
            else:
                new_action_list.append(action)
        return new_action_list

    @staticmethod
    def correct_sac_action_list(
        action_dict_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Correct an action list based on a set of action rules designed for sac synthesis

        Args:
            action_dict_list (List[Dict[str, Any]]): pre processed action sequence

        Returns:
            List[Dict[str, Any]]: The post processed action sequence
        """
        new_action_list: List[Dict[str, Any]] = []
        for action in action_dict_list:
            action_name: str = action["action"]
            content: Dict[str, Any] = action["content"]
            try:
                new_temp: str = content["temperature"]
                if new_temp is None:
                    pass
                elif new_temp.lower() in ["ice-bath", "ice bath"]:
                    new_temp = "0 °C"
                elif new_temp.lower() == "cool":
                    new_temp = "room temperature"
            except KeyError:
                pass
            if action_name == "Add":
                if content["material"]["name"] == "SLN":
                    pass
                else:
                    new_action_list.append(
                        ActionExtractorFromText.delete_dict_keys(action, ["atmosphere"])
                    )
            elif action_name in ["CollectLayer", "Yield"]:
                pass
            elif action_name == "Stir":
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(
                        action, ["atmosphere", "pressure"]
                    )
                )
            elif action_name == "Centrifuge":
                new_action_list.append(
                    ActionExtractorFromText.delete_dict_keys(action, ["phase_to_keep"])
                )
            elif action_name == "Filter":
                if content["phase_to_keep"] is None:
                    action["content"]["phase_to_keep"] = "precipitate"
                new_action_list.append(action)
            elif action_name == "Transfer":
                action["content"]["recipient"] = action["content"]["recipient"].replace(
                    "N/A", ""
                )
                if action["content"]["recipient"] != "":
                    new_action_list.append(action)
            elif action_name in set(["PhaseSeparation", "Degas"]):
                pass
            elif action_name == "Dry":
                action["action"] = "DrySolid"
                new_action_list.append(action)
            elif action_name == "SetTemperature":
                if len(content["atmosphere"]) > 0:
                    new_action_list.append(
                        {
                            "action": "ThermalTreatment",
                            "content": {
                                "temperature": new_temp,
                                "duration": content["duration"],
                                "heat_ramp": content["heat_ramp"],
                                "atmosphere": content["atmosphere"],
                                "flow_rate": None,
                            },
                        }
                    )
                elif new_temp is not None:
                    new_action_list.append(
                        {
                            "action": "SetTemperature",
                            "content": {"temperature": new_temp},
                        }
                    )
                    if content["stirring_speed"] is not None:
                        new_action_list.append(
                            {
                                "action": "Stir",
                                "content": {
                                    "duration": content["duration"],
                                    "stirring_speed": content["stirring_speed"],
                                },
                            }
                        )
                    elif content["duration"] is not None:
                        new_action_list.append(
                            {
                                "action": "Wait",
                                "content": {"duration": content["duration"]},
                            }
                        )
            else:
                new_action_list.append(action)
        return new_action_list

    @staticmethod
    def transform_elementary(action_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correct an action list based on a set of action rules designedto trqnform complex action into elementary ones

        Args:
            action_dict_list (List[Dict[str, Any]]): pre processed action sequence

        Returns:
            List[Dict[str, Any]]: The post processed action sequence
        """
        i: int = 0
        for action in action_dict:
            if action["action"] == "Crystallization":
                content = action["content"]
                new_actions = []
                b = 2
                if content["temperature"] is not None:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": content["temperature"],
                            "microwave": content["microwave"],
                            "heat_ramp": None,
                        },
                    }
                else:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": "Heat",
                            "microwave": content["microwave"],
                            "heat_ramp": None,
                        },
                    }
                new_actions.append(temp)
                if content["pressure"] is not None:
                    atm = {
                        "action": "SetAtmosphere",
                        "content": {
                            "atmosphere": [],
                            "pressure": content["pressure"],
                            "flow_rate": None,
                        },
                    }
                else:
                    atm = {
                        "action": "SetAtmosphere",
                        "content": {
                            "atmosphere": [],
                            "pressure": "autogeneous",
                            "flow_rate": None,
                        },
                    }
                new_actions.append(atm)
                if content["duration"] is not None:
                    if content["stirring_speed"] is not None:
                        stir = {
                            "action": "Stir",
                            "content": {
                                "duration": content["duration"],
                                "stirring_speed": content["stirring_speed"],
                            },
                        }
                    else:
                        stir = {
                            "action": "Wait",
                            "content": {"duration": content["duration"]},
                        }
                    new_actions.append(stir)
                    b += 1
                action_dict = action_dict[:i] + new_actions + action_dict[i + 1 :]
                i += b
            elif action["action"] == "Dry":
                content = action["content"]
                new_actions = []
                b = 1
                if content["temperature"] is not None:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": content["temperature"],
                            "microwave": None,
                            "heat_ramp": None,
                        },
                    }
                else:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": "Heat",
                            "microwave": None,
                            "heat_ramp": None,
                        },
                    }
                new_actions.append(temp)
                if content["atmosphere"] != []:
                    atm = {
                        "action": "SetAtmosphere",
                        "content": {
                            "atmosphere": content["atmosphere"],
                            "pressure": None,
                            "flow_rate": None,
                        },
                    }
                    new_actions.append(atm)
                    b += 1
                if content["duration"] is not None:
                    stir = {
                        "action": "Wait",
                        "content": {"duration": content["duration"]},
                    }
                    new_actions.append(stir)
                    b += 1
                action_dict = action_dict[:i] + new_actions + action_dict[i + 1 :]
                i += b
            elif action["action"] == "ThermalTreatment":
                content = action["content"]
                new_actions = []
                b = 1
                if content["temperature"] is not None:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": content["temperature"],
                            "microwave": False,
                            "heat_ramp": content["heat_ramp"],
                        },
                    }
                else:
                    temp = {
                        "action": "SetTemperature",
                        "content": {
                            "temperature": "Heat",
                            "microwave": False,
                            "heat_ramp": content["heat_ramp"],
                        },
                    }
                new_actions.append(temp)
                if content["atmosphere"] != [] or content["flow_rate"] is not None:
                    atm = {
                        "action": "SetAtmosphere",
                        "content": {
                            "atmosphere": content["atmosphere"],
                            "pressure": None,
                            "flow_rate": content["flow_rate"],
                        },
                    }
                    new_actions.append(atm)
                    b += 1
                if content["duration"] is not None:
                    stir = {
                        "action": "Wait",
                        "content": {"duration": content["duration"]},
                    }
                    new_actions.append(stir)
                    b += 1
                action_dict = action_dict[:i] + new_actions + action_dict[i + 1 :]
                i += b
            else:
                i += 1
        return action_dict

    def retrieve_actions_from_text(
        self, paragraph: str, stop_words: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract the action sequence from a procedure

        Args:
            paragraph (str): procedure to be processed
            stop_words (List[str]): keywords used to stop the action detection

        Raises:
            AttributeError: if the object was not well initialized

        Returns:
            List[Dict[str, Any]]: Action sequence
        """
        if (
            self._llm_model is None
            or self._chemical_prompt is None
            or self._action_prompt is None
            or self._action_parser is None
            or self._schema_parser is None
            or self._quantity_parser is None
            or self._condition_parser is None
            or self._ph_parser is None
            or self._microwave_parser is None
        ):
            raise AttributeError("You need to post initilize the class")
        paragraph = self._molar_ratio_parser.substitute(paragraph)
        #print(paragraph)
        action_prompt: str = self._action_prompt.format_prompt(f"'{paragraph}'")
        #print(action_prompt)
        action_prompt = action_prompt.replace("\x03C", "°C")
        action_prompt = action_prompt.replace("oC", "°C")
        action_prompt = action_prompt.replace("8C", "°C")
        action_prompt = action_prompt.replace("1C", "°C")
        action_prompt = action_prompt.replace("0C", "°C")
        action_prompt = action_prompt.replace("∘C", "°C")
        action_prompt = action_prompt.replace("◦C", "°C")
        action_prompt = action_prompt.replace("ºC", "°C")
        action_prompt = action_prompt.replace("C", "°C")
        action_prompt = action_prompt.replace("C", "°C")
        action_prompt = action_prompt.replace("℃", "°C")
        action_prompt = action_prompt.replace("\x03C", "°C")
        actions_response: str = self._llm_model.run_single_prompt(action_prompt).strip()
        #print(actions_response)
        actions_info: Dict[str, List[str]] = self._action_parser.parse(actions_response)
        i: int = 0
        action_list: List[Dict[str, Any]] = []
        for action_name in actions_info["actions"]:
            context: str = action_name + actions_info["content"][i]
            try:
                action: Any = self._action_dict[action_name.lower()]
            except KeyError:
                action = None
            if action is None:
                #print(action_name)
                if action_name.lower() in stop_words:
                    break
            elif action in set([SetTemperature, Crystallization, ReduceTemperature]):
                new_action: List[Dict[str, Any]] = action.generate_action(
                    context,
                    self._condition_parser,
                    self._complex_parser,
                    self._microwave_parser,
                )
                action_list.extend(new_action)
            elif action in set([ThermalTreatment, Stir]):
                new_action = action.generate_action(
                    context, self._condition_parser, self._complex_parser
                )
                action_list.extend(new_action)
            elif action in set([Quench]):
                chemical_prompt: str = self._chemical_prompt.format_prompt(
                    f"'{context}'"
                )
                chemical_response: str = self._llm_model.run_single_prompt(
                    chemical_prompt
                ).strip()
                #print(chemical_response)
                schemas: List[str] = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._ph_parser,
                    self._banned_parser,
                )
                action_list.extend(new_action)
            elif action in set([MakeSolution, Add, NewSolution]):
                if action is Add:
                    chemical_prompt = self._add_chemical_prompt.format_prompt(
                        f"'{context}'"
                    )
                elif action is NewSolution or action is MakeSolution:
                    chemical_prompt = self._solution_chemical_prompt.format_prompt(
                        f"'{context}'"
                    )
                else:
                    chemical_prompt = self._chemical_prompt.format_prompt(
                        f"'{context}'"
                    )
                chemical_response = self._llm_model.run_single_prompt(
                    chemical_prompt
                ).strip()
                #print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._ph_parser,
                    self._banned_parser,
                    complex_parser=self._complex_parser,
                )
                action_list.extend(new_action)
            elif action is Wash:
                chemical_prompt = self._wash_chemical_prompt.format_prompt(
                    f"'{context}'"
                )
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                #print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._centri_parser,
                    self._filter_parser,
                    self._banned_parser,
                )
                action_list.extend(new_action)
            elif action is Transfer:
                transfer_prompt = self._transfer_prompt.format_prompt(f"'{context}'")
                transfer_response = self._llm_model.run_single_prompt(transfer_prompt)
                #print(transfer_response)
                schemas = self._transfer_schema_parser.parse_schema(transfer_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._transfer_schema_parser,
                    self._transfer_banned_parser,
                )
                action_list.extend(new_action)
            elif action is Separate:
                new_action = action.generate_action(
                    context,
                    self._condition_parser,
                    self._filtrate_parser,
                    self._precipitate_parser,
                    self._centri_parser,
                    self._filter_parser,
                    self._evaporation_parser,
                )
                action_list.extend(new_action)
            elif action is PhaseSeparation:
                new_action = action.generate_action(
                    context,
                    self._condition_parser,
                    self._filtrate_parser,
                    self._precipitate_parser,
                    self._centri_parser,
                    self._filter_parser,
                    self._evaporation_parser,
                )
                action_list.extend(new_action)
            elif action.type == "onlyconditions":
                new_action = action.generate_action(context, self._condition_parser)
                action_list.extend(new_action)
            elif action.type == "onlychemicals":
                chemical_prompt = self._chemical_prompt.format_prompt(f"'{context}'")
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                #print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._banned_parser,
                )
                action_list.extend(new_action)
            elif action.type == "chemicalsandconditions":
                chemical_prompt = self._chemical_prompt.format_prompt(f"'{context}'")
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                #print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._banned_parser,
                )
                action_list.extend(new_action)
            elif action is Filter:
                new_action = action.generate_action(
                    context, self._filtrate_parser, self._precipitate_parser
                )
                action_list.extend(new_action)
            elif action is CollectLayer:
                new_action = action.generate_action(
                    context, self._aqueous_parser, self._organic_parser
                )
                action_list.extend(new_action)
            elif action.type is None:
                new_action = action.generate_action(context)
                action_list.extend(new_action)
            i = i + 1
        #print(action_list)
        if self.post_processing is False:
            final_actions_list: List[Dict[str, Any]] = action_list
        elif self.actions_type == "pistachio":
            final_actions_list: List[Any] = (
                ActionExtractorFromText.correct_pistachio_action_list(action_list)
            )
        elif self.actions_type == "organic":
            final_actions_list = ActionExtractorFromText.correct_organic_action_list(
                action_list
            )
        elif self.actions_type == "materials":
            final_actions_list = ActionExtractorFromText.correct_action_list(
                action_list
            )
        elif self.actions_type == "sac":
            final_actions_list = ActionExtractorFromText.correct_sac_action_list(
                action_list
            )
        else:
            final_actions_list = action_list
        #print(final_actions_list)
        if self.elementar_actions is True:
            final_actions_list = ActionExtractorFromText.transform_elementary(
                final_actions_list
            )
        return final_actions_list


class ParagraphClassifier(BaseModel):
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    prompt_template_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        with open(self.prompt_schema_path, "r") as f:
            prompt_dict = json.load(f)
        self._prompt = PromptFormatter(**prompt_dict)
        self._prompt.model_post_init(self.prompt_template_path)
        if self.llm_model_parameters_path is None:
            llm_param_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "vllm_default_params.json"
            )
        else:
            llm_param_path = self.llm_model_parameters_path
        self._llm_model = ModelLLM(model_name=self.llm_model_name)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()

    def classify_paragraph(self, text: str) -> bool:
        """Classify a paragraph

        Args:
            text (str): Text to be analyzed

        Returns:
            bool: True if the paragraphs matches the prompt question, False otherwise
        """
        prompt: str = self._prompt.format_prompt(text)
        print(prompt)
        response: str = self._llm_model.run_single_prompt(prompt).strip()
        print(response)
        answer_amount: List[str] = re.findall(r"\b(yes|Yes|no|No)\b", response)
        if len(answer_amount) == 0:
            result: bool = True
        elif answer_amount[0].lower() == "yes":
            result = True
        else:
            result = False
        return result


class SamplesExtractorFromText(BaseModel):
    _list_parser: Optional[ListParametersParser] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        atributes = ["name", "preparation", "yield"]
        self._schema_parser = SchemaParser(atributes_list=atributes)
        self._list_parser = ListParametersParser()

    def retrieve_samples_from_text(self, paragraph: str) -> List[Any]:
        """generate the text for each sample presented in the procedure

        Args:
            paragraph (str): Text to be analyzed

        Returns:
            List[Any]: the text for each sample
        """
        paragraph = paragraph.replace("\x03C", "°C")
        paragraph = paragraph.replace("oC", "°C")
        paragraph = paragraph.replace("8C", "°C")
        paragraph = paragraph.replace("1C", "°C")
        paragraph = paragraph.replace("0C", "°C")
        paragraph = paragraph.replace("∘C", "°C")
        paragraph = paragraph.replace("◦C", "°C")
        paragraph = paragraph.replace("ºC", "°C")
        paragraph = paragraph.replace("C", "°C")
        paragraph = paragraph.replace("C", "°C")
        paragraph = paragraph.replace("℃", "°C")
        paragraph = paragraph.replace("\x03C", "°C")
        new_paragraphs_list: List[str] = []
        lists_in_text: List[str] = self._list_parser.find_lists(paragraph)
        print(lists_in_text)
        list_of_text: List[str] = []
        list_of_types: List[str] = []
        list_of_values: List[List[Dict[str, Any]]] = []
        for text_list in lists_in_text:
            parameters_dict = self._list_parser.find_parameters(text_list)
            if self._list_parser.verify_complementary_values(parameters_dict):
                pass
            elif self._list_parser.verify_equal_values(parameters_dict):
                pass
            elif self._list_parser.verify_value_range(parameters_dict):
                list_of_text.append(text_list)
                list_of_types.append(parameters_dict["units_type"])
                list_of_values.append(parameters_dict["values"])
        heteregeneous_indexes: List[List[int]] = (
            self._list_parser.indexes_heterogeneous_lists(
                list(list_of_types), list(list_of_text), paragraph
            )
        )
        text_to_combine: List[List[str]] = []
        indexes_to_delete: List[int] = []
        for index_list in heteregeneous_indexes:
            if len(index_list) == 1:
                index: int = index_list[0]
                indexes_to_delete.append(index)
            elif len(index_list) > 1:
                text: List[str] = []
                for index in index_list:
                    text.append(list_of_text[index])
                    indexes_to_delete.append(index)
                text_to_combine.append(text)
        for index in sorted(indexes_to_delete, reverse=True):
            del list_of_text[index]
            del list_of_types[index]
            del list_of_values[index]
        new_paragraphs_list += self._list_parser.generate_text_by_list(
            text_to_combine, paragraph
        )
        complementary_indexes: List[List[str]] = (
            self._list_parser.indexes_complementary_lists(
                list(list_of_types), list(list_of_values)
            )
        )
        new_list_of_values: List[List[Dict[str, Any]]] = []
        new_list_of_text: List[List[str]] = []
        for index_list in complementary_indexes:
            values: List[Dict[str, Any]] = []
            lists: List[str] = []
            for index in index_list:
                values.append(list_of_values[index])
                lists.append(list_of_text[index])
            new_list_of_values.append(values)
            new_list_of_text.append(lists)
        print(new_list_of_values)
        print(new_list_of_text)
        new_paragraphs_list += self._list_parser.generate_text_by_value(
            new_list_of_text, new_list_of_values, paragraph
        )
        samples_list: List[Dict[str, Any]] = []
        sample_index: int = 1
        for procedure in new_paragraphs_list:
            sample_dict: Dict[str, Any] = {}
            sample_dict["sample"] = f"sample {sample_index}"
            sample_dict["procedure"] = procedure
            samples_list.append(sample_dict)
            sample_index += 1
        return samples_list


class MolarRatioExtractorFromText(BaseModel):
    chemicals_path: Optional[str] = None
    _finder: Optional[MolarRatioFinder] = PrivateAttr(default=None)
    _chemical_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _number_parser: Optional[NumberFinder] = PrivateAttr(default=None)
    _variable_parser: VariableFinder = VariableFinder()
    _equation_parser: EquationFinder = EquationFinder()

    @validator("chemicals_path")
    def layer_options(cls, chemicals_path):
        if chemicals_path is None:
            pass
        elif os.path.splitext(chemicals_path)[-1] != ".txt":
            raise NameError("The file should be a .txt file")
        return chemicals_path

    def model_post_init(self, __context: Any) -> None:
        if self.chemicals_path is None:
            self._finder = MolarRatioFinder(chemicals_list=MOLAR_RATIO_REGISTRY)
            self._chemical_parser = KeywordSearching(keywords_list=MOLAR_RATIO_REGISTRY)
        else:
            with open(self.chemicals_path, "r") as f:
                chemicals_list = f.readlines()
            self._finder = MolarRatioFinder(chemicals_list=chemicals_list)
            self._chemical_parser = KeywordSearching(keywords_list=chemicals_list)
        self._chemical_parser.model_post_init(False)
        self._number_parser = NumberFinder()

    def correct_variables(
        self, ratio_dict: Dict[str, Optional[str]], text: str
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, str], str]:
        """Correct the molar ratio information by updating the variables with values

        Args:
            ratio_dict (Dict[str, Optional[str]]): Initial molar ratio information
            text (str): text from where the molar ratios was extracted

        Returns:
            Tuple[Dict[str, Optional[str]],  Dict[str, str], str]: The updated molar ratio information, the variable conversion information and text from where the molar ratios was extracted
        """
        keys: List[str] = list(ratio_dict.keys())
        conversion_dict: Dict[str, str] = {}
        for key in keys:
            ratio: Optional[str] = ratio_dict[key]
            if ratio is None:
                pass
            elif ratio.isnumeric():
                pass
            elif len(re.findall("[a-zA-Z]", ratio)) == 1:
                value_found: Optional[str] = self._variable_parser.find_value(
                    ratio, text
                )
                conversion_dict[ratio] = key
                if value_found is None:
                    pass
                elif value_found[-1] == ".":
                    ratio = value_found[:-1]
                else:
                    ratio = value_found
                text = text.replace(ratio, "")
                ratio = ratio.strip()
                ratio = ratio.replace(", ", ",")
                ratio = ratio.replace(" and ", ",")
                ratio = ratio.replace(" ", ",")
            ratio_dict[key] = ratio
        return ratio_dict, conversion_dict, text

    def correct_ratios(
        self, ratio_dict: Dict[str, Optional[str]], text: str
    ) -> Tuple[Dict[str, Optional[str]], str]:
        """Correct the molar ratio information with lists of numerical values the same size as the amount of chemical substances in the molar ratio

        Args:
            ratio_dict (Dict[str, Optional[str]]): Initial molar ratio information
            text (str): text from where the molar ratios was extracted

        Returns:
            Tuple[Dict[str, Optional[str]], str]: The updated molar ratio information and the text from where the molar ratios was extracted
        """
        keys: List[str] = list(ratio_dict.keys())
        numbers_list = self._number_parser.find_numbers_list(text, len(keys))
        if numbers_list is None:
            pass
        else:
            if numbers_list[-1] == ".":
                numbers_list = numbers_list[:-1]
            text = text.replace(numbers_list, "")
            numbers_list.replace("and", ",")
            values_list = re.split("[,:\\/]", numbers_list)
            i = 0
            for key in keys:
                ratio_dict[key] = values_list[i]
                i += 1
        return ratio_dict, text

    def find_equations(self, text: str) -> Tuple[List[str], str]:
        """find simple equations having numbers and/letter leeters (x,y,z,a,b,c) in text

        Args:
            text (str): Text to be analysed

        Returns:
            Tuple[List[str], str]: all equations found and the text without them
        """
        equation_list: List[str] = self._equation_parser.find_all(text)
        for equation in equation_list:
            text = text.replace(equation, "")
        return equation_list, text

    def find_ratios(self, text: str) -> Dict[str, str]:
        """Find the ratios of all chemicals in the molar composition

        Args:
            text (str): the molar composition to be analysed

        Returns:
            Dict[str, str]: the ratio of each chemical
        """
        molar_ratios: Dict[str, str] = {}
        all_ratios: Iterator[re.Match[str]] = self._finder.single_ratios(text)
        ratios_found: bool = False
        for ratio in all_ratios:
            print(ratio)
            ratios_found = True
            chemical1: str = ratio.group("chemical1")
            chemical2: str = ratio.group("chemical2")
            value: str = ratio.group("value")
            try:
                initial_value: str = molar_ratios[chemical2]
                list_initial_values: List[str] = re.split("[-–−]", initial_value)
            except KeyError:
                molar_ratios[chemical2] = "1"
                list_initial_values = ["1"]
            final_values: str = ""
            for initial_value in list_initial_values:
                new_value = float(value) * float(initial_value)
                final_values = final_values + f"{new_value}-"
            final_values = final_values[:-1]
            try:
                molar_ratios[chemical1] = molar_ratios[chemical1] + f",{final_values}"
            except KeyError:
                molar_ratios[chemical1] = final_values
        if ratios_found is False:
            chemical_values: Iterator[re.Match[str]] = self._finder.single_values(text)
            for chemical in chemical_values:
                chemical_name: str = chemical.group("chemical")
                ratio_value: str = chemical.group("value")
                molar_ratios[chemical_name] = ratio_value
        return molar_ratios

    def extract_molar_ratio(self, text: str) -> Dict[str, Any]:
        """Extract all molar ratios and respective information from a procedure

        Args:
            text (str): text to be analysed

        Returns:
            Dict[str, Any]: all molar ratios in text and the respective information such as equation  linking variables and letter-values matches
        """
        text = text.replace(" ­", "")
        molar_ratio_list: List[Any] = self._finder.find_molar_ratio(text)
        molar_ratios_result: List[Dict[str, Any]] = []
        equations: List[str] = []
        conversion_dict: Dict[str, str] = {}
        for molar_ratio in molar_ratio_list:
            string: str = molar_ratio[0]
            print(string)
            chemical_information: Dict[str, Any] = (
                self._finder.find_chemical_information(string)
            )
            ratio_dict: Dict[str, str] = chemical_information["result"]
            equations, text = self.find_equations(text)
            if chemical_information["values_found"] is False:
                ratio_dict, text = self.correct_ratios(ratio_dict, text)
            ratio_dict, conversion_dict, text = self.correct_variables(ratio_dict, text)
            if len(ratio_dict.keys()) > 2:
                molar_ratios_result.append(ratio_dict)
            else:
                conversion_dict = {}
        if molar_ratio_list == []:
            new_ratio_dict: Dict[str, str] = self.find_ratios(text)
            if len(new_ratio_dict.keys()) > 2:
                molar_ratios_result.append(new_ratio_dict)
        return {
            "molar_ratios": molar_ratios_result,
            "equations": equations,
            "letters": conversion_dict,
        }


class SteamingDataExtractor(BaseModel):
    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    prompt_template_path: Optional[str] = None
    data_prompt_template_path: Optional[str] = None
    pressure_prompt_template_path: Optional[str] = None
    flow_prompt_template_path: Optional[str] = None
    data_prompt_schema_path: Optional[str] = None
    pressure_prompt_schema_path: Optional[str] = None
    flow_prompt_schema_path: Optional[str] = None
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)
    _data_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _pressure_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _flow_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _condition_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _complex_parser: Optional[ParametersParser] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.data_prompt_schema_path is None:
            self.data_prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "get_steaming_data_schema.json"
            )
        elif self.pressure_prompt_schema_path is None:
            self.pressure_prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "get_steaming_pressure_schema.json"
            )
        elif self.flow_prompt_schema_path is None:
            self.flow_prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "get_steaming_flow_schema.json"
            )
        parser_params_path = self.flow_prompt_schema_path = str(
            importlib_resources.files("xlmexlab")
            / "resources/parsing_parameters"
            / "steaming_parsing_parameters.json"
        )
        with open(self.data_prompt_schema_path, "r") as f:
            data_prompt_dict = json.load(f)
        self._data_prompt = PromptFormatter(**data_prompt_dict)
        self._data_prompt.model_post_init(self.prompt_template_path)
        self._condition_parser = ParametersParser(
            parser_params_path=parser_params_path, convert_units=False
        )
        self._complex_parser = ComplexParametersParser(
            parser_params_path=parser_params_path
        )
        if self.llm_model_name is None:
            self._llm_model = ModelLLM(model_name="microsoft/Phi-3-medium-4k-instruct")
        else:
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
        self._llm_model.load_model_parameters(self.llm_model_parameters_path)
        self._llm_model.vllm_load_model()

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract the steaming conditions from a procedure

        Args:
            text (str): text to analyse

        Returns:
            Dict[str, Any]: all stealing conditions present in the procedure
        """
        data_prompt: str = self._data_prompt.format_prompt(f"'{text}'")
        data_response: str = self._llm_model.run_single_prompt(data_prompt).strip()
        print(data_response)
        result_dict: Dict[str, Any] = {}
        conditions: Conditions = self._condition_parser.get_parameters(data_response)
        complex_conditions: ComplexConditions = self._complex_parser.get_parameters(
            data_response
        )
        for field in Conditions.model_fields.keys():
            value_list = getattr(conditions, field)
            if field == "amount":
                if value_list == {}:
                    result_dict[field] = None
                else:
                    if len(value_list["value"]) == 0:
                        result_dict[field] = None
                    else:
                        result_dict[field] = value_list["value"][0]
            elif field == "pressure":
                if len(value_list) > 1:
                    unit = value_list[0].split(" ")[-1]
                    max_value = max(int(re.findall(r"[\d.]+", x)) for x in value_list)
                    min_value = min(int(re.findall(r"[\d.]+", x)) for x in value_list)
                    result_dict["pressure"] = f"{min_value} {unit}"
                    result_dict["total_pressure"] = f"{max_value} {unit}"
                elif len(value_list) == 1:
                    result_dict["pressure"] = value_list[0]
                    result_dict["total_pressure"] = "101.3 kpa"
                else:
                    result_dict["pressure"] = None
                    result_dict["total_pressure"] = "101.3 kpa"
            elif field == "temperature":
                if len(value_list) > 1:
                    unit = value_list[0].split(" ")[-1]
                    max_value = max(int(re.findall(r"[\d.]+", x)) for x in value_list)
                    min_value = min(int(re.findall(r"[\d.]+", x)) for x in value_list)
                    result_dict["temperature"] = f"{max_value} {unit}"
                    result_dict["saturation_temperature"] = f"{min_value} {unit}"
                elif len(value_list) == 1:
                    result_dict["temperature"] = value_list[0]
                    result_dict["saturation_temperature"] = None
                else:
                    result_dict["temperature"] = None
                    result_dict["saturation_temperature"] = None
            else:
                if len(value_list) == 0:
                    result_dict[field] = None
                else:
                    result_dict[field] = value_list[0]
        for complex_field in ComplexConditions.model_fields.keys():
            complex_value_list = getattr(complex_conditions, complex_field)
            if len(complex_value_list) == 0:
                result_dict[complex_field] = None
            else:
                result_dict[complex_field] = complex_value_list[0]
        return result_dict


class TableExtractor(BaseModel):
    table_type: str = "All"
    prompt_template_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    vlm_model_name: Optional[str] = None
    vlm_model_parameters_path: Optional[str] = None
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _vlm_model: Optional[ModelVLM2] = PrivateAttr(default=None)
    #_condition_parser: Optional[LaTeXTableParser] = PrivateAttr(default=None) # Latex

    def model_post_init(self, __context: Any) -> None:
        if self.vlm_model_parameters_path is None:
            vlm_param_path: str = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "vllm_default_params.json"
            )
        else:
            vlm_param_path = self.vlm_model_parameters_path
        if self.prompt_schema_path is None:
            self.prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "table_extraction_schema.json"
            )
        with open(self.prompt_schema_path, "r") as f:
            prompt_dict = json.load(f)
        self._prompt = PromptFormatter(**prompt_dict)
        self._prompt.model_post_init(self.prompt_template_path)
        if self.vlm_model_name is None:
            self._vlm_model = ModelVLM2(model_name="microsoft/Phi-3-medium-4k-instruct")
        else:
            self._vlm_model = ModelVLM2(model_name=self.vlm_model_name)
        self._vlm_model.load_model_parameters(vlm_param_path)
        self._vlm_model.vllm_load_model()
        #self._condition_parser = LaTeXTableParser() #latex

    def extract_table_info(self, image_path: str, scale: float = 1.0) -> None:
        image_name = os.path.basename(image_path) #adicionado

        prompt = self._prompt.format_prompt("<image>")
        print("[TableExtractor] Prompt:")
        print(prompt)
        output = self._vlm_model.run_image_single_prompt_rescale(
            prompt, image_path, scale=scale
        )
        #Latex
        #parsed_output = self._condition_parser.parse(output)
        #HTML
        print(f"output is: {output}")
        extractor = Extractor(output)
        extractor.parse()
        table = extractor.return_list()
        print(table)
        pattern = r'_\{\text\{([^}]+)\}\}'
        parsed_output = [[re.sub(pattern, r'\1', cell).replace(' \\)', '').replace('\\( ', '') for cell in row] for row in table]
        print(parsed_output)
        return image_path,  parsed_output

class Table2Blocks(BaseModel): #from pdf2data
    page: int
    name: str
    block: List[List[str]]
    type: str = "Table"
    collumn_headers: List[int] = []
    row_indexes: List[int] = []
    number: int = 0
    legend: str = ""
    box: List[float] = []
    letter_ratio: float = 3

    def find_collumn_headers(self) -> None:
        """find the collumn headers as rows that do not have numbers"""
        if len(self.block) == 0:
            pass
        elif len(self.block[0]) == 0:
            pass
        else:
            collumn_headers: List[int] = []
            find_number: bool = True
            for row_number in range(len(self.block)):
                if find_number is False:
                    collumn_headers.append(row_number - 1)
                find_number = False
                for entry in self.block[row_number]:
                    if entry == "":
                        digits: int = 0
                        letters: int = 0
                    else:
                        digits = len(re.findall("[1-9]", entry))
                        letters = len(re.findall("[a-zA-Z]", entry))
                    # Verify if the entry as any letter
                    if digits > self.letter_ratio * letters:
                        find_number = True
                        break
            self.collumn_headers = collumn_headers

    def find_row_indexes(self, max_rows: int = 2) -> None:
        """find the row indexes by finding collumns without entries with three times more digits then letters

        Parameters
        ----------
        max_rows : int, optional
            maximum rows to be considered, by default 2
        """
        row_indexes: List[int] = []
        find_number: bool = True
        if len(self.block) == 0:
            pass
        elif len(self.block) == 0:
            pass
        else:
            max_rows: int = min(len(self.block[0]), max_rows)
            for collumn_number in range(max_rows):
                find_number = False
                for row in self.block:
                    if row[collumn_number] == "":
                        digits: int = 0
                        letters: int = 0
                    else:
                        # test = re.search('[a-zA-Z]', row[collumn_number])
                        digits = len(re.findall("[1-9]", row[collumn_number]))
                        letters = len(re.findall("[a-zA-Z]", row[collumn_number]))
                        # print(f'{row[collumn_number]} presents {digits} digits and {letters} letters')
                    # Verify if the entry as any letter
                    # if test is None:
                    if digits > self.letter_ratio * letters:
                        find_number = True
                        break
                if find_number is False:
                    row_indexes.append(collumn_number)
            self.row_indexes = row_indexes

    def create_dict(
        self,
        page: Any,
        page_size: List[float],
        layout_boxes: List[List[float]],
        layout_types: List[str],
        index: int,
    ) -> Dict[str, Any]:
        """generates a dictionary describing the table object

        Parameters
        ----------
        page : Any
            pdf page to be considered
        page_size : List[float]
            size of the page
        boxes : List[List[float]]
            list of boxes of the page layout
        types : List[str]
            list of the types of the page layout
        index : int
            position of the table or figure in the layout list

        Returns
        -------
        Dict[str, Any]
            a dictionary with the page number, table entries, type, collumn headers, row indexes, table number, legend and table coordinates
        """
        i_vertical: int = 1
        # Go through all entries in the table
        self.find_collumn_headers()
        self.find_row_indexes()
        self.legend = find_legend(
            page, page_size, layout_boxes, layout_types, index, type=self.type
        )
        for i_horizontal in range(len(self.block)):
            j_horizontal: int = 0
            for j_vertical in range(len(self.block[i_horizontal])):
                if i_vertical < len(self.block):
                    # Verify if the entry is empty
                    if self.block[i_vertical][j_vertical] == "":
                        # New entry is the one above
                        new_entry_vert: str = self.block[i_vertical - 1][j_vertical]
                        if (
                            re.search("[a-zA-Z]", new_entry_vert) is not None
                            or j_vertical == 0
                        ):
                            self.block[i_vertical][j_vertical] = new_entry_vert
                if j_horizontal < len(self.block[i_horizontal]) and j_horizontal > 0:
                    if self.block[i_horizontal][j_horizontal] == "":
                        # New entry is the one on the left
                        new_entry_horiz = self.block[i_horizontal][j_horizontal - 1]
                        self.block[i_horizontal][j_horizontal] = new_entry_horiz
                j_horizontal = j_horizontal + 1
            i_vertical = i_vertical + 1
        image_rect: fitz.Rect = fitz.Rect(
            self.box[0], self.box[1], self.box[2], self.box[3]
        )
        mat: fitz.Matrix = fitz.Matrix(3, 3)
        # Get Image from the Rectangle
        image: Any = page.get_pixmap(matrix=mat, clip=image_rect)
        # Save as Tiff
        image.pil_save(self.name, format="TIFF")
        result = self.__dict__
        del result["letter_ratio"]
        return result


class List2Headers(BaseModel):
    table_type: str = "All"
    prompt_template_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    vlm_model_name: Optional[str] = None
    vlm_model_parameters_path: Optional[str] = None

    _vlm_model: Optional[ModelVLM2] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any = None) -> None:
        if self.vlm_model_name is None:
            self._vlm_model = ModelVLM2(model_name="microsoft/Phi-3-medium-4k-instruct")
        else:
            self._vlm_model = ModelVLM2(model_name=self.vlm_model_name)

        if self.vlm_model_parameters_path is None:
            vlm_param_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "vllm_default_params.json"
            )
        else:
            vlm_param_path = self.vlm_model_parameters_path

        self._vlm_model.load_model_parameters(vlm_param_path)
        self._vlm_model.vllm_load_model()

    @staticmethod
    def update_schema_with_extracted_data(base_json_path: str, extracted_data: list):
        with open(base_json_path, "r", encoding="utf-8") as f:
            base_json = json.load(f)

        schema_copy = copy.deepcopy(base_json)
        context_str = json.dumps(extracted_data, ensure_ascii=False)

        if "objective" in schema_copy and "{fille here for each image}" in schema_copy["objective"]:
            schema_copy["objective"] = schema_copy["objective"].replace(
                "{fille here for each image}", context_str
            )

        return schema_copy

    def extract_table_info(
        self,
        image_path: str,
        extracted_data: Optional[list] = None,
        scale: float = 1.0
    ):
        image_name = os.path.basename(image_path)

        if extracted_data is not None:
            image_schema = self.update_schema_with_extracted_data(
                self.prompt_schema_path, extracted_data
            )
        else:
            with open(self.prompt_schema_path, "r", encoding="utf-8") as f:
                image_schema = json.load(f)

        from xlmexlab.prompt import PromptFormatter
        prompt_formatter = PromptFormatter(**image_schema)
        prompt_formatter.model_post_init(self.prompt_template_path)
        prompt = prompt_formatter.format_prompt(image_schema)

        print(f"\n[HEADER PROMPT] {image_name}\n{prompt}\n")

        output = self._vlm_model.run_image_single_prompt_rescale(
            prompt, image_path, scale=scale
        )

        return image_path, output



class ImageExtractor(BaseModel):
    prompt_template_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    vlm_model_name: Optional[str] = None
    vlm_model_parameters_path: Optional[str] = None
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _vlm_model: Optional[ModelVLM2] = PrivateAttr(default=None)
    _image_parser: Optional[ImageParser] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.vlm_model_parameters_path is None:
            vlm_param_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "vllm_default_params.json"
            )
        else:
            vlm_param_path = self.vlm_model_parameters_path
        #SCHEMA LOADING
        if self.prompt_schema_path is None:
            self.prompt_schema_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/schemas"
                / "image_extraction_schema.json"
            )
        with open(self.prompt_schema_path, "r") as f:
            prompt_dict = json.load(f)
        self._prompt = PromptFormatter(**prompt_dict)
        #VLM loading
        self._prompt.model_post_init(self.prompt_template_path)
        if self.vlm_model_name is None:
            self._vlm_model = ModelVLM2(model_name="Llama2-70B-chat-hf")
        else:
            self._vlm_model = ModelVLM2(model_name=self.vlm_model_name)
        self._vlm_model.load_model_parameters(vlm_param_path)
        self._vlm_model.vllm_load_model()
        
        self._image_parser = ImageParser()

    def extract_image_info(self, image_path: str, scale: float = 1.0):
        image_name = os.path.basename(image_path)

        prompt = self._prompt.format_prompt("<image>")

        output = self._vlm_model.run_image_single_prompt_rescale(
            prompt, image_path, scale=scale
        )

        self._image_parser.parse(output)
        parsed_output = self._image_parser.get_data_dict()
        print(parsed_output)
        return {image_name: parsed_output}
