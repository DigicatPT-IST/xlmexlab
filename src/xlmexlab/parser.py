import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import importlib_resources
import pandas as pd
from pint import UnitRegistry
from pydantic import BaseModel, Field, PrivateAttr, validator
from quantulum3 import parser
from trieregex import TrieRegEx as TRE


class Amount(BaseModel):
    value: List[str] = []
    repetitions: List[int] = []


class ComplexConditions(BaseModel):
    stirring_speed: List[str] = []
    heat_ramp: List[str] = []
    concentration: List[str] = []
    flow_rate: List[str] = []


class Conditions(BaseModel):
    duration: List[str] = []
    temperature: List[str] = []
    pressure: List[str] = []
    atmosphere: List[str] = []
    size: List[str] = []
    amount: Dict[str, List[Any]] = {}
    other: List[str] = []


class Parameters(BaseModel):
    time_units: List[str] = []
    temperature_units: List[str] = []
    pressure_units: List[str] = []
    quantity_units: List[str] = []
    size_units: List[str] = []
    stirring_units: List[str] = []
    heat_ramp_units: List[str] = []
    concentration_units: List[str] = []
    flow_rate_units: List[str] = []
    time_words: List[str] = []
    temperature_words: List[str] = []
    pressure_words: List[str] = []
    atmosphere_words: List[str] = []
    amount_words: List[str] = []
    size_words: List[str] = []


class ParametersParser(BaseModel):
    parser_params_path: str = str(
        importlib_resources.files("xlmexlab")
        / "resources/parsing_parameters"
        / "synthesis_parsing_parameters.json"
    )
    time: bool = True
    temperature: bool = True
    pressure: bool = True
    atmosphere: bool = True
    amount: bool = True
    size: bool = True
    convert_units: bool = True
    base_time: str = "minute"
    base_temperature: str = "degree_Celsius"
    base_pressure: str = "bar"
    base_volume: str = "milliliter"
    base_mass: str = "milligram"
    base_quantity: str = "millimole"
    base_size: str = "meter"
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _ureg: Optional[UnitRegistry] = PrivateAttr(default=None)
    _Q: Any = PrivateAttr(default=None)
    UnitRegistry()

    @validator("parser_params_path")
    def verify_path(cls, v) -> str:
        if Path(v).exists() is False:
            raise ValueError("The path given does not exist")
        return v

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        self._ureg = UnitRegistry()
        self._Q = self._ureg.Quantity
        with open(self.parser_params_path, "r") as f:
            parser_params_dict = json.load(f)
        parser_params: Parameters = Parameters(**parser_params_dict)
        units_list: List[str] = []
        time_word_list: List[str] = [
            "$%##&#$@%"
        ]  # This avoids the problem of empty lists on regex
        temperature_word_list: List[str] = ["$%##&#$@%"]
        pressure_word_list: List[str] = ["$%##&#$@%"]
        atmosphere_word_list: List[str] = ["$%##&#$@%"]
        amount_word_list: List[str] = ["$%##&#$@%"]
        size_word_list: List[str] = ["$%##&#$@%"]
        if self.time is True:
            units_list += parser_params.time_units
            if parser_params.time_words != []:
                time_word_list = parser_params.time_words
        if self.temperature is True:
            units_list += parser_params.temperature_units
            if parser_params.temperature_words != []:
                temperature_word_list = parser_params.temperature_words
        if self.pressure is True:
            units_list += parser_params.pressure_units
            if parser_params.pressure_words != []:
                pressure_word_list = parser_params.pressure_words
        if self.atmosphere is True:
            if parser_params.atmosphere_words != []:
                atmosphere_word_list = parser_params.atmosphere_words
        if self.amount is True:
            units_list += parser_params.quantity_units
            if parser_params.amount_words != []:
                amount_word_list = parser_params.amount_words
        if self.size is True:
            units_list += parser_params.size_units
            if parser_params.size_words != []:
                size_word_list = parser_params.size_words
        units_tre: re.Pattern[str] = correct_tre(units_list)
        time_word_tre: re.Pattern[str] = correct_tre(time_word_list)
        temperature_word_tre: re.Pattern[str] = correct_tre(temperature_word_list)
        pressure_word_tre: re.Pattern[str] = correct_tre(pressure_word_list)
        atmosphere_word_tre: re.Pattern[str] = correct_tre(atmosphere_word_list)
        amount_word_tre: re.Pattern[str] = correct_tre(amount_word_list)
        size_word_tre: re.Pattern[str] = correct_tre(size_word_list)
        regex: str = (
            rf"([\"'\(\[\s,]((?P<repetitions1>\d+\.?,?\d*)[xX×]+)?(?P<number1>\+?-?-?\d+\.?,?\d*)(?P<unit1>.?)(-*|to)\s*(?P<number2>\d*\.?,?\d*)\s*(?P<unit2>.?\s*{units_tre})([xX×]+(?P<repetitions2>\d+\.?,?\d*))?(?=[\)\]\s,\"'\(\.])|\b(?P<word>(?P<time>{time_word_tre})|(?P<temperature>{temperature_word_tre})|(?P<pressure>{pressure_word_tre})|(?P<atmosphere>{atmosphere_word_tre})|(?P<amount>{amount_word_tre})|(?P<size>{size_word_tre}))\b)"
        )
        self._regex = re.compile(regex, re.IGNORECASE | re.MULTILINE)

    def transform_value(self, number: str, unit: str) -> tuple[str, str, str]:
        """transform value to standard units

        Args:
            number (str): numerical value
            unit (str): numerical unit

        Returns:
            tuple[str, str, str]: the value, unit and type of the update parameter
        """
        try:
            parameter: UnitRegistry.Quantity = self._Q(float(number), unit)
            if parameter.check("[time]") is True:
                unit_type: str = "duration"
                if self.base_time is not None and self.convert_units is True:
                    parameter.ito(self.base_time)
            elif parameter.check("[temperature]") is True:
                unit_type = "temperature"
                if self.base_temperature is not None and self.convert_units is True:
                    parameter.ito(self.base_temperature)
            elif parameter.check("[pressure]") is True:
                unit_type = "pressure"
                if self.base_pressure is not None and self.convert_units is True:
                    parameter.ito(self.base_pressure)
            elif parameter.check("[mass]") is True:
                unit_type = "quantity"
                if self.base_mass is not None and self.convert_units is True:
                    parameter.ito(self.base_mass)
            elif parameter.check("[length]") is True:
                unit_type = "size"
                if self.base_mass is not None and self.convert_units is True:
                    parameter.ito(self.base_mass)
            elif parameter.check("[substance]") is True:
                unit_type = "quantity"
                if self.base_quantity is not None and self.convert_units is True:
                    parameter.ito(self.base_quantity)
            elif parameter.check("[volume]") is True:
                unit_type = "quantity"
                if self.base_volume is not None and self.convert_units is True:
                    parameter.ito(self.base_volume)
            else:
                unit_type = "other"
            if self.convert_units is True:
                digit: str = str(parameter.magnitude)
                final_unit: str = str(parameter.units)
            else:
                digit = number
                final_unit = unit
        except Exception:
            unit_type = "other"
            digit = number
            final_unit = unit
        return digit, final_unit, unit_type

    def get_value(self, match: re.Match) -> Dict[str, Union[str, int]] | None:
        """get the parameter value information from a regex match

        Args:
            match (re.Match): regex match of the parameter

        Returns:
            Dict[str, Union[str, int]] | None: the value type, digit, unit type, unit and amount of repetions or None if the match do not contain a value
        """
        if match.group("number2") == "":
            value_type: str = "single"
            digit: str = match.group("number1")
            unit: str = match.group("unit1") + match.group("unit2")
            unit = unit.replace(" ", "")
            try:
                final_digit, final_unit, unit_type = self.transform_value(digit, unit)
            except Exception:
                return None
        else:
            value_type = "range"
            number1: str = match.group("number1")
            number2: str = match.group("number2")
            unit = match.group("unit2").replace(" ", "")
            try:
                digit1, final_unit, unit_type = self.transform_value(number1, unit)
                digit2, final_unit, unit_type = self.transform_value(number2, unit)
                final_digit = f"{digit1}-{digit2}"
            except Exception:
                return None
        if match.group("repetitions1") is not None:
            repetitions: int = round(float(match.group("repetitions1")))
        elif match.group("repetitions2") is not None:
            repetitions = round(float(match.group("repetitions2")))
        else:
            repetitions = 1
        if unit_type == "other":
            final_unit = unit
        return {
            "value_type": value_type,
            "number": final_digit,
            "condition_type": unit_type,
            "unit": final_unit,
            "repetitions": repetitions,
        }

    def get_string(self, match: re.Match) -> Dict[str, str] | None:
        """get the parameter word information from a regex match

        Args:
            match (re.Match): regex match of the paraemeter

        Returns:
            Dict[str, Union[str, int]] | None: the value type, condition type, and word or None if the match do not contain a word
        """
        if match.group("word") is None:
            return None
        string = match.group("word")
        value_type: str = "word"
        if match.group("time") is not None:
            word_type: str = "duration"
        elif match.group("temperature") is not None:
            word_type = "temperature"
        elif match.group("pressure") is not None:
            word_type = "pressure"
        elif match.group("atmosphere") is not None:
            word_type = "atmosphere"
        elif match.group("amount") is not None:
            word_type = "amount"
        else:
            word_type = "other"
        return {"value_type": value_type, "condition_type": word_type, "string": string}

    def get_parameters(self, text: str) -> Conditions:
        """get all the parameters wanted from a string

        Args:
            text (str): string to be processed

        Raises:
            ValueError: if the parser was not initilized

        Returns:
            Conditions: an object containing list of all the parameter found
        """
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        text = " " + text + " "  # needed to avoid errors at the regex parser
        results: Iterator[re.Match[str]] = self._regex.finditer(text)
        conditions: Conditions = Conditions()
        amount: Amount = Amount()
        condition: Optional[Dict[str, Any]]
        value: str
        for result in results:
            if result.group("number1") is not None:
                condition: Optional[Dict[str, str | int]] = self.get_value(result)
                if condition is not None:
                    condition_type: str = condition["condition_type"]
                    value: str = f"{condition['number']} {condition['unit']}"
                    repetitions: int = condition["repetitions"]
            else:
                condition = self.get_string(result)
                if condition is not None:
                    condition_type = condition["condition_type"]
                    value = condition["string"]
                    repetitions = 1
            if condition is None:
                pass
            elif condition_type == "duration":
                conditions.duration.append(value)  # type: ignore
            elif condition_type == "temperature":
                conditions.temperature.append(value)  # type: ignore
            elif condition_type == "pressure":
                conditions.pressure.append(value)  # type: ignore
            elif condition_type == "atmosphere":
                conditions.atmosphere.append(value)  # type: ignore
            elif condition_type == "quantity":
                amount.value.append(value)  # type: ignore
                amount.repetitions.append(repetitions)  # type: ignore
            elif condition_type == "size":
                conditions.size.append(value)  # type: ignore
            else:
                conditions.other.append(value)  # type: ignore
        conditions.amount = amount.__dict__
        return conditions


class ListParametersParser(BaseModel):
    parser_params_path: str = str(
        importlib_resources.files("xlmexlab")
        / "resources/parsing_parameters"
        / "synthesis_parsing_parameters.json"
    )
    quantity_range: int = 100
    _individual_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _list_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        with open(self.parser_params_path, "r") as f:
            parser_params_dict = json.load(f)
        parser_params = Parameters(**parser_params_dict)
        parser_params = Parameters(**parser_params_dict)
        time_units_list: List[str] = parser_params.time_units
        print(time_units_list)
        temperature_units_list: List[str] = parser_params.temperature_units
        pressure_units_list: List[str] = parser_params.pressure_units
        quantity_units_list: List[str] = parser_params.quantity_units
        stirring_units_list: List[str] = parser_params.stirring_units
        heating_ramp_units_list: List[str] = parser_params.heat_ramp_units
        concentration_units_list: List[str] = parser_params.concentration_units
        flow_rate_units_list: List[str] = parser_params.flow_rate_units
        time_units_tre: re.Pattern[str] = correct_tre(time_units_list)
        temperature_units_tre: re.Pattern[str] = correct_tre(temperature_units_list)
        pressure_units_tre: re.Pattern[str] = correct_tre(pressure_units_list)
        quantity_units_tre: re.Pattern[str] = correct_tre(quantity_units_list)
        stirring_units_tre: re.Pattern[str] = correct_tre(stirring_units_list)
        heating_ramp_units_tre: re.Pattern[str] = correct_tre(heating_ramp_units_list)
        concentration_units_tre: re.Pattern[str] = correct_tre(concentration_units_list)
        flow_rate_units_tre: re.Pattern[str] = correct_tre(flow_rate_units_list)
        temperature_word_tre: re.Pattern[str] = correct_tre(
            ["room temperature", "ambient temperature"]
        )
        banned_words: List[str] = ["sample", "solution"]
        lookbehind: str = r""
        for word in banned_words:
            lookbehind += rf"(?<!{word})"
        individual_regex = rf"(?P<value>[\d\.\-–−]+|{temperature_word_tre})\s*((?P<time>{time_units_tre})|(?P<temperature>\s*[^C]?\s*{temperature_units_tre})|(?P<pressure>.?{pressure_units_tre})|(?P<quantity>.?{quantity_units_tre})|(?P<stirring_speed>[^-\),\[\]\d\s]?\s*{stirring_units_tre})|(?P<heat_ramp>.?\s*{heating_ramp_units_tre})|(?P<concentration>.?\s*{concentration_units_tre})|(?P<flow_rate>[^-\),\[\]\d\s]?\s*{flow_rate_units_tre}))*"
        list_regex = (
            lookbehind
            + rf"[^\w\-](([\d\.\-–−]+|{temperature_word_tre})\s*(({time_units_tre})|(.?{quantity_units_tre})|(\s*[^C]?\s*{temperature_units_tre})|(.?{pressure_units_tre})|([^-\),\[\]\d\s]?\s*{stirring_units_tre})|(.?\s*{heating_ramp_units_tre})|(.?\s*{concentration_units_tre})|([^-\),\[\]\d\s]?\s*{flow_rate_units_tre}))*\s*(,|\/|\bor\b|\band\b|,\s*and|,\s*or)\s*)+([\d\.])+\s*(({time_units_tre})|(.?{quantity_units_tre})|(\s*[^C]?\s*{temperature_units_tre})|(.?{pressure_units_tre})|([^-\),\[\]\d\s]?{stirring_units_tre})|(.?\s*{heating_ramp_units_tre})|(.?{concentration_units_tre})|([^-\),\[\]\d\s]?\s*{flow_rate_units_tre}))[^\w\-]"
        )
        self._individual_regex = re.compile(
            individual_regex, re.IGNORECASE | re.MULTILINE
        )
        self._list_regex = re.compile(list_regex, re.IGNORECASE | re.MULTILINE)

    def find_lists(self, text: str) -> List[str]:
        """Find all parameters lists in the text

        Args:
            text (str): text to be processed

        Raises:
            ValueError: if the regex parser was not initialized

        Returns:
            List[str]: All the parameters lists founds
        """
        if self._list_regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        results: Iterator[re.Match[str]] = self._list_regex.finditer(text)
        results_list: List[str] = []
        for match in results:
            results_list.append(match.group(0))
        return results_list

    def get_units(self, parameter: re.Match[str]) -> Dict[str, str]:
        """get the units from a parameters list match

        Args:
            parameter (re.Match[str]): match of a parameters list

        Returns:
            Dict[str, str]: return the unit of the parameters and the unit type
        """
        result_dict: Dict[str, str] = {}
        result_dict["unit"] = ""
        result_dict["unit_type"] = ""
        groups_list: List[str] = list(self._individual_regex.groupindex.keys())
        for group in groups_list:
            if group == "value":
                pass
            elif parameter.group(group) is not None:
                result_dict["unit"] = parameter.group(group)
                result_dict["unit_type"] = UNITS_LETTER_REGISTRY[group]
        return result_dict

    def verify_complementary_values(self, values_dict: Dict[str, Any]) -> bool:
        """Verify if the list conatain parameters that complement each other

        Args:
            values_dict (Dict[str, Any]): all values inside the list

        Returns:
            bool: True if there are complementary parameters, False otherwise
        """
        units_type: str = values_dict["units_type"]
        units_type = units_type.replace(UNITS_LETTER_REGISTRY["quantity"], "")
        units_type = units_type.replace(UNITS_LETTER_REGISTRY["concentration"], "")
        test: bool = True
        if units_type != "":
            return False
        i = 0
        for value in values_dict["values"][:-1]:
            value_unit: str = value["unit"]
            for other_value in values_dict["values"][i + 1 :]:
                if value_unit == other_value["unit"]:
                    test = False
                    break
            i += 1
        return test

    def verify_equal_values(self, values_dict: Dict[str, Any]) -> bool:
        """Verify if there are parameters that are the same is in a list

        Args:
            values_dict (Dict[str, Any]): All parameters in the list

        Returns:
            bool: True if there are values that are the same in the list, False otherwise
        """
        test: bool = False
        i: int = 0
        for value in values_dict["values"][:-1]:
            value_string: str = value["value"] + value["unit"]
            for other_value in values_dict["values"][i + 1 :]:
                other_value_string: str = other_value["value"] + other_value["unit"]
                if value_string == other_value_string:
                    test = True
                    break
            i += 1
        return test

    def verify_value_range(self, values_dict: Dict[str, Any]) -> bool:
        """verify if the separation between the minimum and maximum value does not surpass a threshold

        Args:
            values_dict (Dict[str, Any]): All parameters in the list

        Returns:
            bool: True if the distance between the minimum and maximum value on the list are inside the threshold, False otherwise
        """
        test: bool = True
        units_type: str = values_dict["units_type"]
        units_type = units_type.replace(UNITS_LETTER_REGISTRY["quantity"], "")
        if units_type == "":
            sorted_values: List[Dict[str, Any]] = sorted(
                values_dict["values"], key=lambda d: float(d["value"])
            )
            min_value: float = float(sorted_values[0]["value"])
            max_value: float = float(sorted_values[-1]["value"])
            if max_value > self.quantity_range * min_value and min_value != 0:
                test = False
        return test

    def indexes_heterogeneous_lists(
        self, list_of_types: List[str], list_of_text: List[str], text: str
    ) -> List[List[int]]:
        """Find the index of all list containing parameters with different units

        Args:
            list_of_types (List[str]): all unit type seauence from all lists in the text
            list_of_text (List[str]): all lists strings found on the entire text
            text (str): text source for the parameters list

        Returns:
            List[List[int]]: the indexes of all the ehteregeneous lists
        """
        i: int = 0
        final_lists: List[List[int]] = []
        initial_index: int = 0
        while i < len(list_of_types):
            type_sequence: str = list_of_types[i]
            if len(set(type_sequence)) > 1:
                list_of_index: List[int] = [initial_index]
                position: int = text.find(list_of_text[i])
                j: int = i + 1
                new_index: int = initial_index + 1
                while j < len(list_of_types):
                    other_sequence: str = list_of_types[j]
                    other_position: int = text.find(list_of_text[j])
                    if position == other_position:
                        del list_of_types[j]
                    elif (
                        type_sequence == other_sequence
                        and position > other_position - 10
                    ):
                        list_of_index.append(new_index)
                        del list_of_types[j]
                    else:
                        j += 1
                    new_index += 1
                del list_of_types[i]
                final_lists.append(list_of_index)
            else:
                i += 1
            initial_index += 1
        return final_lists

    def values_are_equal(
        self, values_list1: List[Dict[str, Any]], values_list2: List[Dict[str, Any]]
    ) -> bool:
        """Verify if two lists have excatly the same parameter

        Args:
            values_list1 (List[Dict[str, Any]]): first list of values
            values_list2 (List[Dict[str, Any]]): second list of values

        Returns:
            bool: True if the lists containing the same parameters, False otherwise
        """
        test: bool = True
        i: int = 0
        for values in values_list1:
            if values["value"] != values_list2[i]["value"]:
                test = False
                break
            i += 1
        return test

    def indexes_complementary_lists(
        self, list_of_types: List[str], lists_of_values: List[List[Dict[str, Any]]]
    ) -> List[List[int]]:
        """Group the indexes of lists of parameters that are complementary to each other. Complementary lists are the same size and containin the same parameters or parameters with complementary units(e.g amount and concentration)

        Args:
            list_of_types (List[str]): All parameter type sequences
            lists_of_values (List[List[Dict[str, Any]]]): All the lists present in text

        Raises:
            AttributeError: if the amount of parameter type sequence does not match the amount of parameters lists

        Returns:
            List[List[int]]: groups of indexes of complementary lists
        """
        if len(list_of_types) != len(lists_of_values):
            raise AttributeError("Both lists must be of the same length")
        i: int = 0
        final_lists: List[List[int]] = []
        while i < len(list_of_types):
            list_of_index: List[int] = [i]
            sequence_type: str = list_of_types[i]
            values_list: Dict[str, Any] = lists_of_values[i]
            type_match: set = set(TYPE_COMPARISSON_REGISTRY[sequence_type[0]])
            j: int = i + 1
            while j < len(list_of_types) and len(set(sequence_type)) == 1:
                other_sequence_type: str = list_of_types[j]
                other_values_list: Dict[str, Any] = lists_of_values[j]
                if other_sequence_type[0] in type_match and len(sequence_type) == len(
                    other_sequence_type
                ):
                    if len(type_match) > 1:
                        list_of_index.append(j)
                        del list_of_types[j]
                        del lists_of_values[j]
                    elif self.values_are_equal(values_list, other_values_list):
                        list_of_index.append(j)
                        del list_of_types[j]
                        del lists_of_values[j]
                    else:
                        j += 1
                else:
                    j += 1
            i += 1
            final_lists.append(list_of_index)
        return final_lists

    def find_parameters(self, text: str) -> Dict[str, Any]:
        """Find all paremeters lists in text

        Args:
            text (str): text source

        Raises:
            ValueError: if the regex parser was not initialized

        Returns:
            Dict[str, Any]: all parameters list and their respective parameter type sequence
        """
        if self._individual_regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        if text[-1] == ".":
            text_to_analyse: str = text[:-1]
        else:
            text_to_analyse: str = text
        results: List[re.Match[str]] = list(
            self._individual_regex.finditer(text_to_analyse)
        )
        results_dict: Dict[str, Any] = {}
        results_dict["values"] = []
        results_dict["units_type"] = ""
        final_unit_info: Dict[str, Any] = self.get_units(results[-1])
        final_unit: str = final_unit_info["unit"]
        final_unit_type: str = final_unit_info["unit_type"]
        for result in results:
            result_info: Dict[str, Any] = {}
            result_info["value"] = result.group("value")
            unit_info: Dict[str, Any] = self.get_units(result)
            result_info["unit"] = unit_info["unit"]
            unit_type: str = unit_info["unit_type"]
            if result_info["unit"] == "":
                if bool(re.search(r"\d", result_info["value"])) is False:
                    result_info["unit"] = ""
                    unit_type: str = "t"
                else:
                    result_info["unit"] = final_unit
                    unit_type: str = final_unit_type
            results_dict["units_type"] += unit_type
            results_dict["values"].append(result_info)
        return results_dict

    def generate_text_by_value(
        self, list_of_sequences: List[str], list_of_lists: List[List[Any]], text: str
    ) -> List[str]:
        """Generate variations by substituting full parameters lists for values in the source text

        Args:
            list_of_sequences (List[str]): parameters type seauence of each list
            list_of_lists (List[List[Any]]): parameters list to consider
            text (str): source text

        Raises:
            AttributeError: if the amount of parameter type sequence does not match the amount of parameters lists

        Returns:
            List[str]: all the text variations created
        """
        if len(list_of_sequences) != len(list_of_lists):
            raise AttributeError("Both lists must be of the same length")
        text_list: List[str] = []
        if len(list_of_lists) == 0:
            return [text]
        list_of_values: List[Any] = list_of_lists[0]
        list_of_strings: List[Any] = list_of_sequences[0]
        for i in range(len(list_of_values[0])):
            for j in range(len(list_of_values)):
                string: str = list_of_strings[j]
                value: Dict[str, str] = list_of_values[j][i]
                if (
                    value["value"].isdigit()
                    or re.match(r"^-?\d+(?:[\.,]\d+)$", value["value"]) is not None
                ):
                    value_string: str = " " + value["value"] + value["unit"] + " "
                else:
                    value_string: str = " " + value["value"] + " "
                new_text = text.replace(string, value_string)
            text_list += self.generate_text_by_value(
                list_of_sequences[1:], list_of_lists[1:], new_text
            )
        return text_list

    def generate_text_by_list(
        self, list_of_lists: List[List[Any]], text: str
    ) -> List[str]:
        """Generate variations by removing parameters lists in the text source

        Args:
            list_of_lists (List[List[Any]]): groups of parameters lists to keep in text
            text (str): source text

        Returns:
            List[str]: all the text variations created
        """
        text_list_result: List[str] = []
        for text_list in list_of_lists:
            i = 0
            for text_keep in text_list:
                j = 0
                new_text = text
                for text_remove in text_list:
                    if i != j:
                        new_text = new_text.replace(text_remove, "")
                    j += 1
                text_list_result.append(new_text)
                i += 1
        return text_list_result


class ComplexParametersParser(BaseModel):
    parser_params_path: str = str(
        importlib_resources.files("xlmexlab")
        / "resources/parsing_parameters"
        / "synthesis_parsing_parameters.json"
    )
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        with open(self.parser_params_path, "r") as f:
            parser_params_dict = json.load(f)
        parser_params = Parameters(**parser_params_dict)
        stirring_units_list: List[str] = parser_params.stirring_units
        heating_ramp_units_list: List[str] = parser_params.heat_ramp_units
        concentration_units_list: List[str] = parser_params.concentration_units
        flow_rate_units_list: List[str] = parser_params.flow_rate_units
        stirring_units_tre: re.Pattern[str] = correct_tre(stirring_units_list)
        heating_ramp_units_tre: re.Pattern[str] = correct_tre(heating_ramp_units_list)
        concentration_units_tre: re.Pattern[str] = correct_tre(concentration_units_list)
        flow_rate_units_tre: re.Pattern[str] = correct_tre(flow_rate_units_list)
        regex: str = (
            rf"([\"'\(\[\s,](?P<number1>\+?-?-?\d+\.?,?\d*)-*(?P<number2>\d*\.?,?\d*)\s*((?P<stirring_speed>[^-\),\[\]\d\s]?\s*{stirring_units_tre})|(?P<heat_ramp>.?\s*{heating_ramp_units_tre})|(?P<concentration>.?\s*{concentration_units_tre})|(?P<flow_rate>[^-\),\[\]\d\s]?\s*{flow_rate_units_tre}))(?=[\)\]\s,\"'\(\.]))"
        )
        self._regex = re.compile(regex, re.IGNORECASE | re.MULTILINE)

    def generate_value(self, match: re.Match) -> Dict[str, str]:
        """Generate all parameter information from a regex match

        Args:
            match (re.Match): Regex match to consider

        Returns:
            Dict[str, str]: the parameter with value and unit and the type of parameter
        """
        if match.group("number2") != "":
            value: str = f"{match.group('number1')}-{match.group('number2')}"
        else:
            value = f"{match.group('number1')}"
        if match.group("stirring_speed") is not None:
            unit: str = f"{match.group('stirring_speed')}"
            condition_type: str = "stirring_speed"
        if match.group("heat_ramp") is not None:
            unit = f"{match.group('heat_ramp')}"
            condition_type = "heat_ramp"
        if match.group("concentration") is not None:
            unit = f"{match.group('concentration')}"
            condition_type = "concentration"
        if match.group("flow_rate") is not None:
            unit = f"{match.group('flow_rate')}"
            condition_type = "flow_rate"
        return {"value": f"{value} {unit}", "condition_type": condition_type}

    def get_parameters(self, text: str) -> ComplexConditions:
        """Get all parameters present in a text source and organize them by classes

        Args:
            text (str): source text

        Raises:
            ValueError: if the regex parser have not been initialized

        Returns:
            ComplexConditions: an object contaning all the parameter organized
        """
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        text = " " + text + " "  # needed to avoid errors at the regex parser
        results: Iterator[re.Match[str]] = self._regex.finditer(text)
        conditions: ComplexConditions = ComplexConditions()
        for result in results:
            result_dict: Dict[str, str] = self.generate_value(result)
            condition_type: str = result_dict["condition_type"]
            value: str = result_dict["value"]
            if condition_type == "stirring_speed":
                conditions.stirring_speed.append(value)  # type: ignore
            elif condition_type == "heat_ramp":
                conditions.heat_ramp.append(value)  # type: ignore
            elif condition_type == "concentration":
                conditions.concentration.append(value)
            elif condition_type == "flow_rate":
                conditions.flow_rate.append(value)  # type: ignore
        return conditions


class ActionsParser(BaseModel):
    separators: List[str] = []
    type: str = "materials"
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        if self.type == "materials":
            self.separators = self.separators + MATERIAL_SEPARATORS_REGISTRY
        elif self.type == "pistachio":
            self.separators = self.separators + PISTACHIO_SEPARATORS_REGISTRY
        tre_regex: re.Pattern[str] = correct_tre(self.separators)
        self._regex = re.compile(f"\\b{tre_regex}\\b", re.IGNORECASE | re.MULTILINE)

    def change_separators(
        self,
        extra_separators: List | None = None,
        separators_to_remove: List | None = None,
    ) -> None:
        """Update the separators from the object

        Args:
            extra_separators (List | None, optional): list of extra separator to consider. Defaults to None.
            separators_to_remove (List | None, optional): list of separator to not be consider. Defaults to None.
        """
        tre: TRE = TRE(*self.separators)
        if extra_separators is not None:
            tre.add(*extra_separators)
        if separators_to_remove is not None:
            tre.remove(*separators_to_remove)
        self._regex = re.compile(f"\\b{tre.regex()}\\b", re.IGNORECASE | re.MULTILINE)

    def parse(self, text: str) -> Dict[str, List[str]]:
        """Separate the text by separator and extract the text between separator

        Args:
            text (str): source text

        Returns:
            Dict[str, List[str]]: All the separator found and the text between them
        """
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.initialize()"
            )
        actions: List[str] = self._regex.findall(text)
        content: List[str] = self._regex.split(text)[1:]
        i = 0
        for action in actions[:-1]:
            if action == actions[i + 1] and len(content[i]) < 6:
                del actions[i]
                del content[i]
            elif content[i].lower().replace(":", "").strip() == "n/a":
                del actions[i]
                del content[i]
            else:
                i += 1
        return {"actions": actions, "content": content}


class KeywordSearching(BaseModel):
    keywords_list: List[str]
    limit_words: bool = True
    word_type: str = "normal"
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    @validator("keywords_list")
    def list_not_empty(cls, v) -> List[str]:
        if v == []:
            raise ValueError(
                "The keyword list is empty, please give a valid keyword list"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        tre_regex: re.Pattern[str] = correct_tre(self.keywords_list)
        if self.word_type == "units":
            self._regex = re.compile(
                f"(?<![a-zA-Z]){tre_regex}(?![a-zA-Z])", re.IGNORECASE | re.MULTILINE
            )
        elif self.limit_words is False:
            self._regex = re.compile(f"{tre_regex}", re.IGNORECASE | re.MULTILINE)
        else:
            self._regex = re.compile(f"\\b{tre_regex}\\b", re.IGNORECASE | re.MULTILINE)

    def find_keywords(self, text: str) -> List[str]:
        """find all the keywords inside a text

        Args:
            text (str): source text

        Returns:
            List[str]: all the keywords found
        """
        if self._regex is None:
            raise AttributeError("There is no valid regex loaded")
        return self._regex.findall(text)


class SchemaParser(BaseModel):
    atributes_list: List[str]
    limiters: Dict[str, str] = {"initial": "{", "final": "}"}
    _schema_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _atributes_regex: Dict[str, re.Pattern[str]] = PrivateAttr(default={})

    @validator("limiters")
    def limiters_list_not_empty(cls, v) -> Dict[str, str]:
        if v == {}:
            raise ValueError(
                "The limiters dictionary is empty, please give a valid limiter dicionary. The correct format is:\n {{'inital' : 'put_inital_limiter_here', 'final' : 'put_final_limiter_here'}}"
            )
        elif "initial" not in v.keys() and "final" not in v.keys():
            raise ValueError(
                "The limiters format is not valid. The correct format is:\n {{'initial' : 'put_inital_limiter_here', 'final' : 'put_final_limiter_here'}}"
            )
        return v

    @validator("atributes_list")
    def atributes_list_not_empty(cls, v) -> List[str]:
        if v == []:
            raise ValueError("The atributes_list should not be an empty list")
        return v

    def model_post_init(self, _context: Any) -> None:
        limiters_list: List[str] = [self.limiters["initial"], self.limiters["final"]]
        tre_regex: re.Pattern[str] = correct_tre(limiters_list)
        self._schema_regex = re.compile(f"{tre_regex}", re.IGNORECASE | re.MULTILINE)
        for atribute in self.atributes_list:
            self._atributes_regex[atribute] = re.compile(
                rf"[\"']*{atribute}[\"']*\s*[:=-]\s*[\"']*([^\"']*)[\"']*",
                re.IGNORECASE | re.MULTILINE,
            )

    def parse_schema(self, text: str) -> List[str]:
        """Obtain all schema results from a text

        Args:
            text (str): source text

        Raises:
            AttributeError: if the regex parser was not initialized

        Returns:
            List[str]: All schema found in the text
        """
        if self._schema_regex is None:
            raise AttributeError(
                "You need to Initialize the object first using <object>.model_post_init method"
            )
        limiters: List[str] = self._schema_regex.findall(text)
        context: List[str] = self._schema_regex.split(text)
        if len(context) > len(limiters):
            context = context[1:]
        i = 0
        close_limiter_value: int = 0
        result: List[str] = []
        summed_context: str = ""
        for limiter in limiters:
            if limiter == self.limiters["initial"]:
                close_limiter_value += 1
            else:
                close_limiter_value -= 1
            if close_limiter_value <= 0:
                summed_context = summed_context + limiter
                result.append(summed_context)
                close_limiter_value = 0
                summed_context = ""
            else:
                summed_context = summed_context + limiter + context[i]
            i += 1
        return result

    def get_atribute_value(self, text: str, atribute: str) -> List[Any]:
        """get all values for an atirbute of the schema

        Args:
            text (str): source text
            atribute (str): schema atribute

        Raises:
            ValueError: if the atribute does not exist on the schema

        Returns:
            List[Any]: all values linked with the specific atribute
        """
        if atribute not in self.atributes_list:
            raise ValueError(
                f"The give atribute is not valid, the valid atributes are {self.atributes_list}"
            )
        regex: re.Pattern[str] = self._atributes_regex[atribute]
        results: List[Any] = regex.findall(text)
        i = 0
        for result in results:
            if result == "":
                results[i] = result
            else:
                while result[-1] in set([",", " "]):
                    result = result[:-1]
                    if result == "":
                        break
                    print(result)
                results[i] = result
            i += 1
        return results


class DimensionlessParser:

    @classmethod
    def get_dimensionless_numbers(cls, context: str) -> List[str]:
        """Find all dimentionless number inside a text

        Args:
            context (str): source text

        Returns:
            List[str]: all the dimentionless numbers found
        """
        quants: List[Any] = parser.parse(context)
        dimensionless_list: List[str] = []
        for quant in quants:
            if quant.unit.entity.name == "dimensionless":
                dimensionless_list.append(str(quant.value))
        return dimensionless_list


class MolarRatioFinder(BaseModel):
    chemicals_list: List[str]
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _single_ratio_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _single_value_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _entries_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any):
        tre_regex: re.Pattern[str] = correct_tre(self.chemicals_list)
        self._regex = re.compile(
            rf"(([   \t\(]*([\d\.\s\-–−]|[xyznkabc\+])*[   \t\)]*({tre_regex})[   \t\()]*([\d\.\s\-–−]|[xyznkabc\+])*\)?"
            + r"[   \t\)]*[:\/\-]?){3,})",
            re.IGNORECASE | re.MULTILINE,
        )
        self._entries_regex = re.compile(
            rf"[   \t\(]*(?P<number1>\+?-?-?\d+\.?,?\d*[-–−]*?\d*\.?,?\d*|[xXyYzZnkaAbBcC\+]\d?)*[   \t\-)]*(?P<chemical>({tre_regex}))[   \t\(-]*(?P<number2>\+?-?-?\d+\.?,?\d*[-–−]*?\d*\.?,?\d*|[xXyYzZnkaAbBcC\+]\d?)*[   \t\)]*[:\/\-]?",
            re.MULTILINE,
        )
        self._single_ratio_regex = re.compile(
            rf"(?P<chemical1>({tre_regex}))[ \t]*[/]+[ \t]*(?P<chemical2>({tre_regex}))[ \t]*(=|is|was)[ \t]*(?P<value>[\d\.-–−]+)",
            re.IGNORECASE | re.MULTILINE,
        )
        self._single_value_regex = re.compile(
            rf"(?P<chemical>({tre_regex}))[ \t]*(=|is|was)[ \t]*(?P<value>[\d\.-–−]+)",
            re.IGNORECASE | re.MULTILINE,
        )

    def find_molar_ratio(self, text: str) -> List[Any]:
        """Fin all molar ratios in text

        Args:
            text (str): source text

        Raises:
            AttributeError: if the regex parser was not initialized

        Returns:
            List[Any]: all molar ratios found
        """
        if self._regex is None:
            raise AttributeError("There is no valid regex loaded")
        return self._regex.findall(text)

    def single_ratios(self, text: str) -> Iterator[re.Match[str]]:
        """find singular ratios between two chemical substances

        Args:
            text (str): source text

        Returns:
            Iterator[re.Match[str]]: all the indidual ratios
        """
        all_ratios: Iterator[re.Match[str]] = self._single_ratio_regex.finditer(text)
        return all_ratios

    def single_values(self, text: str) -> Iterator[re.Match[str]]:
        """find individual ratio values

        Args:
            text (str): text containing a ratio

        Returns:
             Iterator[re.Match[str]]: all individual ratio values found
        """
        all_ratios: Iterator[re.Match[str]] = self._single_value_regex.finditer(text)
        return all_ratios

    def find_chemical_information(self, text: str) -> Dict[str, Any]:
        """Extract all chemical substances and associated rarios form a molar ratio string

        Args:
            text (str): string from the molar ratio

        Raises:
            AttributeError: if the regex parser was not initialized

        Returns:
            Dict[str, Any]: the chemical co,position dictionary and the indication if it found any ratio value
        """
        if self._entries_regex is None:
            raise AttributeError("There is no valid regex loaded")
        found_values: bool = False
        chemicals_list: List[re.Match[str]] = self._entries_regex.finditer(text)
        final_dict: Dict[str, str] = {}
        for chemical in chemicals_list:
            print(chemical)
            chemical_name: str = chemical.group("chemical").replace(" ", "")
            number: Optional[str] = None
            if chemical.group("number1") is not None:
                number = chemical.group("number1")
                print(number)
                found_values = True
            elif chemical.group("number2") is not None:
                number = chemical.group("number2")
                found_values = True
            final_dict[chemical_name] = number
        return {"result": final_dict, "values_found": found_values}

    def substitute(self, text: str):
        """substitute the molar ratios in text by "unknown"

        Args:
            text (str): source text

        Returns:
            _type_: the text the the molar ratios substituted
        """
        molar_ratio_list = self.find_molar_ratio(text)
        print(molar_ratio_list)
        if len(molar_ratio_list) == 0:
            return text
        for molar_ratio in molar_ratio_list:
            molar_ratio_value = molar_ratio[0]
            if molar_ratio_value[0] != " ":
                new_string: str = molar_ratio_value[0] + " unknown "
            else:
                new_string = " unknown "
            text = text.replace(molar_ratio_value, new_string)
        return text


class NumberFinder(BaseModel):
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _list_regex: str = rf"[\d\.xyz\-–−]+(?:[ \t]*(,|and|:|\/)[ \t]*[\d\.xyz\-–−]+)"

    def model_post_init(self, __context: Any):
        regex = rf"[-–−]*\s*((\+?-?-?\d+\.?,?\d*)[-–−]*(\d*\.?,?\d*))"
        self._regex = re.compile(regex, re.IGNORECASE | re.MULTILINE)

    def find_numbers(self, text: str) -> List[str]:
        """Find number with our without units

        Args:
            text (str): source text

        Returns:
            _List[str]: all number found in text
        """
        return self._regex.findall(text)

    def find_numbers_list(self, text: str, size: int) -> Optional[str]:
        """Find a list of number with our without units with a specific size

        Args:
            text (str): source text
            size (int): size of the list

        Returns:
            Optional[str]: the first list found with the specific size
        """
        regex_string: str = self._list_regex + "{" + rf"{size - 1}" + "}"
        lists_found: Optional[re.Match] = re.search(regex_string, text, re.MULTILINE)
        if lists_found is None:
            result: Optional[str] = None
        else:
            result = lists_found.group(0)
        return result


class VariableFinder(BaseModel):
    _value_regex: str = (
        rf"[  \t ]*(=|at|is|was|were|are)[  \t]*(?P<value>[\d\.]+(?:[  \t]*(,|and|-|–)[  \t]*[\d\.]+)*)"
    )

    def find_value(self, variable: str, text: str) -> Optional[str]:
        """Find the value of a varable mentioned in the text

        Args:
            variable (str): variable name
            text (str): source text

        Returns:
            Optional[str]: the value associated with the variables
        """
        regex_string: str = r"[^+\/][  \t ]+" + variable + self._value_regex
        pattern: Optional[re.Match] = re.search(regex_string, text, re.MULTILINE)
        if pattern is None:
            value: Optional[str] = None
        else:
            value = pattern.group("value")
        return value


class EquationFinder(BaseModel):
    _equation_regex: str = (
        r"(?P<value>[\d\.xyzabc]+(?:[  \t]*(,|and|\+|\\)[  \t]*[\d\.xyzabc]+)+)[  \t]*=[  \t]*[\d\.xyzabc]+"
    )

    def find_all(self, text: str) -> List[str]:
        """Find simple equations containing number and/or x,y,z,a,b,c

        Args:
            text (str): source text

        Returns:
            List[str]: all the equations found
        """
        matches: Iterator[re.Match[str]] = re.finditer(
            self._equation_regex, text, re.MULTILINE
        )
        equation_list: List[str] = []
        for match in matches:
            equation = match.group(0)
            if equation[-1] == ".":
                equation = equation[:-1]
            equation_list.append(equation)
        return equation_list


def correct_tre(word_list: List[str]) -> re.Pattern[str]:
    """Correct the regex TRE by repeating the process until the TRE is not  empty

    Args:
        word_list (List[str]): list of word for the TRE creation

    Raises:
        TimeoutError: if the TRE keeps being e,pty after 100 iteractions

    Returns:
        re.Pattern[str]: TRE object of the word list
    """
    tre: TRE = TRE(*word_list)
    regex: re.Pattern[str] = tre.regex()
    i: int = 0
    while len(regex) < 1:
        tre = TRE(*[""])
        tre = TRE(*word_list)
        regex = tre.regex()
        if i > 100:
            raise TimeoutError(
                "It was not possible to achieve the correct regex in the maximum amount of interations"
            )
        i += 1
    return regex


class TableParser(BaseModel):
    table_type: str = "materials_characterization"
    _words_registry: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _word_searcher: Optional[Dict[str, KeywordSearching]] = PrivateAttr(default={})
    _unit_searcher: Optional[KeywordSearching] = PrivateAttr(default={})
    _unique_set: set = set()

    def model_post_init(self, __context: Any):
        if self.table_type == "materials_characterization":
            self._words_registry = MATERIALS_CHARACTERIZATION_REGISTRY
        if self._words_registry is None:
            raise AttributeError(f"{self.table_type} is not a valid table_type")
        unit_list: List[str] = []
        regex_pattern: str = ""
        unique_variables = []
        for key in self._words_registry.keys():
            self._word_searcher[key] = KeywordSearching(
                keywords_list=self._words_registry[key]["words"],
                limit_words=self._words_registry[key]["limit_words"],
            )
            if self._words_registry[key]["unique"]:
                unique_variables.append(key)
            for unit in self._words_registry[key]["units"]:
                if unit not in unit_list:
                    unit_list.append(unit)
        print(self._word_searcher)
        self._unique_set = set(unique_variables)
        self._unit_searcher = KeywordSearching(
            keywords_list=unit_list, word_type="units"
        )

    def update_result(
        self,
        results: List[Dict[str, Any]],
        table_entries: List[List[str]],
        indexes_to_ignore: List[int],
        units: List[str],
        key: str,
        index: int,
    ) -> List[Dict[str, Any]]:
        """update all lines results with the information related with the specified collumn from the table

        Args:
            results (List[Dict[str, Any]]): results containing all entries to be updated
            table_entries (List[List[str]]): full table
            indexes_to_ignore (List[int]): table indexes to ignore
            units (List[str]): units found for the column to be extracted
            key (str): key associated with the table column to be extracted
            index (int): column to be extracted index

        Returns:
            List[Dict[str, Any]]: results containing the updated entries
        """
        if len(units) > 0:
            unit: str = f" {units[0]}"
        else:
            unit = " empty"
        j: int = 0
        line_index: int = 0
        for line in table_entries:
            if line_index not in indexes_to_ignore:
                entry: str = line[index]
                try:
                    test: Dict[str, Any] = results[j]
                    new_entry: str = entry.strip() + unit
                except IndexError:
                    results.append({})
                    new_entry = entry.strip() + unit
                new_entry = new_entry.replace(" empty", "")
                if entry.strip() in EMPTY_VALUES_REGISTRY:
                    pass
                elif unit[1:] not in self._words_registry[key]["units"]:
                    pass
                elif key in results[j].keys():
                    if key in self._unique_set and len(results[j][key]) > 0:
                        pass
                    else:
                        results[j][key].append(new_entry)
                else:
                    results[j][key] = [new_entry]
                j += 1
            line_index += 1
        return results

    def extract_columns(
        self, table_entries: List[List[str]], collumn_headers: List[int]
    ) -> List[Dict[str, Any]]:
        """extract relevant data from each column and organize it by line

        Args:
            table_entries (List[List[str]]): full table
            collumn_headers (List[int]): collumn headers indexes

        Returns:
            List[Dict[str, Any]]: All relevant data per line
        """
        results: List[Dict[str, Any]] = []
        headers: List[str] = []
        for i in range(len(table_entries[0])):
            header_string: str = ""
            previous_index: int = -1
            for index in collumn_headers:
                if index > previous_index + 1:
                    break
                header_string += " " + table_entries[index][i].strip()
                previous_index = index
            headers.append(header_string)
        keys_to_use: List[str] = list(self._words_registry.keys())
        i = 0
        for header in headers:
            units: List[str] = self._unit_searcher.find_keywords(header.lower())
            corrected_header: str = header.replace(",", "")
            corrected_header = corrected_header.replace("(", "")
            corrected_header = corrected_header.replace(")", "")
            corrected_header = corrected_header.replace("[", "")
            corrected_header = corrected_header.replace("]", "")
            corrected_header = corrected_header.replace("  ", " ")
            if len(units) > 0:
                corrected_header = corrected_header.replace(units[0], "")
                corrected_header = corrected_header.replace("  ", " ")
            print(corrected_header.lower())
            print(units)
            if corrected_header == " ":
                key: str = "sample"
                print(key)
                results = self.update_result(
                    results, table_entries, collumn_headers, units, key, i
                )
            for key in keys_to_use:
                words_found: List[str] = self._word_searcher[key].find_keywords(
                    corrected_header.lower()
                )
                if len(words_found) > 0:
                    print(key)
                    results = self.update_result(
                        results, table_entries, collumn_headers, units, key, i
                    )
                    break
            i += 1
        return results

    def find_infos(self, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all relevant data from the table an organize it

        Args:
            table_info (Dict[str, Any]): table information

        Returns:
            List[Dict[str, Any]]: all relevant data organized
        """
        table_entries: List[List[str]] = table_info["block"]
        collumn_headers: List[int] = table_info["collumn_headers"]
        row_indexes: List[int] = table_info["row_indexes"]
        results: List[Dict[str, Any]] = self.extract_columns(
            table_entries, collumn_headers
        )
        if len(results) == 0:
            pass
        elif len(results[0].keys()) < 2:
            pass
        else:
            return results
        new_table_entries: List[List[str]] = list(map(list, zip(*table_entries)))
        results = self.extract_columns(new_table_entries, row_indexes)
        return results

class LaTeXTableParser(BaseModel):
    """
    A versatile parser for LaTeX tables that converts them to lists of lists.
    Supports: tabular, tabularx, longtable, array environments.
    Handles: multicolumn, multirow, hline, cline, and various formatting.
    """

    table_environments: List[str] = Field(default_factory=lambda: ['tabular', 'tabularx', 'longtable', 'array'])

    def parse(self, latex_content: str) -> List[List[str]]:
        """
        Parse the first table from LaTeX content.
        
        Returns:
            A list of rows, where each row is a list of cells
        """
        print(f'The LatEX content is:{latex_content}')
        for env in self.table_environments:
            pattern = rf'\\begin{{{env}}}.*?\\end{{{env}}}'
            matches = re.finditer(pattern, latex_content, re.DOTALL)
            
            for match in matches:
                table_content = match.group(0)
                parsed_table = self._parse_single_table(table_content, env)
                if parsed_table:
                    print(f'The result after parsing is: {parsed_table}')
                    return parsed_table  # Return first valid table
        
        return []  # Return empty list if no tables found

    def _parse_single_table(self, table_content: str, env: str) -> Optional[List[List[str]]]:
        pattern = rf'\\begin{{{env}}}(?:\[[^\]]*\])?(?:\{{[^}}]*\}})*\s*(.*)\\end{{{env}}}'
        match = re.search(pattern, table_content, re.DOTALL)

        if not match:
            return None

        content = match.group(1)
        content = content.replace('@', '')
        content = re.sub(r'^\s*[lcrp|{}\d\s]+(?=&|\\)', '', content)
        # Protect escaped special characters
        content = content.replace(r'\%', '<<<PERCENT_ESC>>>')
        content = content.replace(r'\$', '<<<DOLLAR_ESC>>>')
        content = content.replace(r'\&', '<<<AMPERSAND_ESC>>>')
        content = content.replace(r'\_', '<<<UNDERSCORE_ESC>>>')
        content = content.replace(r'\#', '<<<HASH_ESC>>>')

        # Remove comments
        content = re.sub(r'%.*?$', '', content, flags=re.MULTILINE)

        # Remove rules
        # Remove \hline, \cline, \toprule, \midrule, \bottomrule, \cmidrule
        content = re.sub(r'\\(?:hline|toprule|midrule|bottomrule|cline\{[^}]+\}|cmidrule(?:\([^)]*\))?\{[^}]+\})', '', content)

        rows = re.split(r'\\\\', content)

        parsed_rows: List[List[str]] = []
        multirow_tracker: Dict[int, Tuple[int, str]] = {}

        for row in rows:
            row = row.strip()
            if not row:
                continue

            parsed_row = self._parse_row(row, multirow_tracker)
            if parsed_row:
                parsed_rows.append(parsed_row)

                keys_to_delete = []
                for col_idx, (remaining, content_val) in multirow_tracker.items():
                    remaining -= 1
                    if remaining <= 0:
                        keys_to_delete.append(col_idx)
                    else:
                        multirow_tracker[col_idx] = (remaining, content_val)

                for key in keys_to_delete:
                    del multirow_tracker[key]

        return parsed_rows if parsed_rows else None

    def _parse_row(
        self,
        row: str,
        multirow_tracker: Dict[int, Tuple[int, str]]
    ) -> Optional[List[str]]:

        raw_cells = row.split('&')

        expanded_cells: List[str] = []
        for raw_cell in raw_cells:
            raw_cell = raw_cell.strip()

            mc_pattern = r'\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}'
            mc_match = re.match(mc_pattern, raw_cell)

            if mc_match:
                num_cols = int(mc_match.group(1))
                content = self._clean_cell_content(mc_match.group(2))
                expanded_cells.extend([content] * num_cols)
            else:
                expanded_cells.append(raw_cell)

        final_cells: List[str] = []
        for cell_idx, cell in enumerate(expanded_cells):
            mr_pattern = r'\\multirow\{(\d+)\}\{[^}]*\}\{(.*)\}'
            mr_match = re.search(mr_pattern, cell)

            if mr_match:
                num_rows = int(mr_match.group(1))
                content = self._clean_cell_content(mr_match.group(2))
                multirow_tracker[cell_idx] = (num_rows, content)
                final_cells.append(content)
            elif cell.strip() == '':
                if cell_idx in multirow_tracker:
                    final_cells.append(multirow_tracker[cell_idx][1])
                else:
                    final_cells.append('')
            else:
                cleaned = self._clean_cell_content(cell)
                if cleaned == '' and cell_idx in multirow_tracker:
                    final_cells.append(multirow_tracker[cell_idx][1])
                else:
                    final_cells.append(cleaned)

        return final_cells or None

    def _clean_cell_content(self, cell: str) -> str:
        cell = cell.replace(r'\times', '×')
        cell = cell.replace(r'\mu', 'µ')
        cell = cell.replace(r'\textmu', 'µ')
        cell = cell.replace(r'\circ', '°')
        cell = cell.replace(r'\(', '')
        cell = cell.replace(r'\)', '')
        max_iterations = 10
        for _ in range(max_iterations):
            old_cell = cell

            cell = re.sub(r'\\textbf\{([^}]*)\}', r'\1', cell)
            cell = re.sub(r'\\textit\{([^}]*)\}', r'\1', cell)
            cell = re.sub(r'\\emph\{([^}]*)\}', r'\1', cell)
            cell = re.sub(r'\\text\{([^}]*)\}', r'\1', cell)
            cell = re.sub(r'\\\w+\{([^}]*)\}', r'\1', cell)

            if cell == old_cell:
                break

        cell = re.sub(r'\\\w+', '', cell)
        cell = re.sub(r'\\textit\{([^}]*)\}', r'\1', cell)
        cell = re.sub(r'\\\\', '', cell)
        cell = re.sub(r'\\', '', cell)
        cell = re.sub(r'\n', '', cell)
        cell = re.sub(r'[{}]', '', cell)

        cell = cell.replace('<<<DOLLAR_ESC>>>', '$')
        cell = cell.replace('<<<PERCENT_ESC>>>', '%')
        cell = cell.replace('<<<AMPERSAND_ESC>>>', '&')
        cell = cell.replace('<<<UNDERSCORE_ESC>>>', '_')
        cell = cell.replace('<<<HASH_ESC>>>', '#')
        cell = re.sub(r'[_$^]', '', cell)

        return cell.strip()

    def parse_to_dict(self, latex_content: str, has_header: bool = True) -> List[Dict[str, str]]:
        tables = self.parse(latex_content)
        result: List[Dict[str, str]] = []

        for table in tables:
            if not table:
                continue

            if has_header and len(table) > 1:
                headers = table[0]
                for row in table[1:]:
                    row_dict = {
                        headers[i]: row[i] if i < len(row) else ''
                        for i in range(len(headers))
                    }
                    result.append(row_dict)
            else:
                for row in table:
                    result.append({f'col_{i}': val for i, val in enumerate(row)})

        return result


PISTACHIO_SEPARATORS_REGISTRY: List[str] = [
    "Initialization",
    "Note",
    "Notes",
    "FollowOtherProcedure",
    "NMR",
    "ESIMS",
    "Pour",
    "Distill",
    "Collect",
    "Dissolve",
    "Final Product:",
    "Remove",
    "Warm",
    "Dilute",
    "Solidify",
    "Provide",
    "Afford",
    "Obtain",
]

MATERIAL_SEPARATORS_REGISTRY: List[str] = [
    "Initialization",
    "Note",
    "Notes",
    "NMR",
    "ESIMS",
]

MOLAR_RATIO_REGISTRY: List[str] = [
    "TBP OH",
    "NaAlO2",
    "P2O5",
    "Mor",
    "TEAOH",
    "TEOS",
    "TBPOH",
    "[Cu(NH2CH2CH2NH2)2]2+",
    "Al(NO3)3",
    "CTAB",
    "TPOA",
    "TMAda",
    "Si/28",
    "OSDA",
    "GeO2",
    "C22–6–6(OH)2",
    "NH3",
    "Ni",
    "Fe(NO3)3",
    "HCL",
    "NaCl",
    "n-butylamine",
    "EtOH",
    "MnO",
    "SiO2",
    "TiO2",
    "TPAOH",
    "H2O",
    "Mn(NO3)2•4H2O",
    "Fe(NO3)3•9H2O",
    "TEPA",
    "Al2O3",
    "OPA",
    "H2SO4",
    "NaOH",
    "TPABr",
    "Ga2O3",
    "Na2O",
    "Na2 O",
    "CDM",
    "TPOAC",
    "TPA2O",
    "ODAC",
    "template",
    "K2O",
    "Al (OH)3",
    "NH4F",
    "Al",
    "TPA Br",
    "Fe2O3",
    "Mn",
    "OH",
    "TPA20",
    "F127",
    "TBAOH",
    "SnCl4",
    "RN-OH",
    "DEA",
    "Au",
    "C16IMZ",
    "Cs2O",
    "TMAdaOH",
    "CTAB(DTAB)",
    "IPA",
    "T-40",
    "SDA",
    "B2O3",
]

UNITS_LETTER_REGISTRY: Dict[str, str] = {
    "time": "d",
    "temperature": "t",
    "pressure": "p",
    "quantity": "q",
    "stirring_speed": "s",
    "heat_ramp": "h",
    "concentration": "c",
    "flow_rate": "f",
}

TYPE_COMPARISSON_REGISTRY: Dict[str, str] = {
    "d": ["d"],
    "t": ["t"],
    "p": ["p"],
    "q": ["q", "c"],
    "s": ["s"],
    "h": ["h"],
    "c": ["q", "c"],
    "f": ["f"],
}

MATERIALS_CHARACTERIZATION_REGISTRY: Dict[str, Any] = {
    "sample": {
        "words": [
            "sample",
            "catalyst",
            "zeolites",
            "samples",
            "zeolite",
            "material",
            "support",
        ],
        "units": ["empty"],
        "limit_words": False,
        "unique": True,
    },
    "yield": {
        "words": ["yield"],
        "units": ["%", "empty"],
        "limit_words": False,
        "unique": True,
    },
    "external_area": {
        "words": [
            "sext",
            "smes",
            "s mes",
            "Surface area Meso",
            "External",
            "area external",
        ],
        "units": ["m2/g", "m2/g", "m2g-1", "m2 g-1", "m2.g-1"],
        "limit_words": False,
        "unique": True,
    },
    "micropore_area": {
        "words": ["smic", "s mic", "bet area microporous", "area micropore"],
        "units": ["m2/g", "m2/g", "m2g-1", "m2 g-1", "m2.g-1"],
        "limit_words": False,
        "unique": True,
    },
    "surface_area": {
        "words": [
            "sbet",
            "Surface area BET",
            "Surface area",
            "bet area",
            "area total",
            "stotal",
            "s total",
        ],
        "units": ["m2/g", "m2/g", "m2g-1", "m2.g-1", "m2 g-1"],
        "limit_words": False,
        "unique": True,
    },
    "micropore_volume": {
        "words": [
            "vmic",
            "v mic",
            "Pore volume Micro",
            "Micropore volume",
            "Microporous volume",
            "volume micro",
        ],
        "units": [
            "cm3/g",
            "cm3g-1",
            "cm3.g-1",
            "cm3 g-1",
            "mm3/g",
            "mm3g-1",
            "mm3.g-1",
            "mm3 g-1",
            "ml/g",
        ],
        "limit_words": False,
        "unique": True,
    },
    "mesopore_volume": {
        "words": [
            "vmes",
            "v mes",
            "Pore volume Meso",
            "Mesopore volume",
            "Vext",
            "volume meso",
        ],
        "units": [
            "cm3/g",
            "cm3g-1",
            "cm3.g-1",
            "cm3 g-1",
            "mm3/g",
            "mm3g-1",
            "mm3.g-1",
            "mm3 g-1",
        ],
        "limit_words": False,
        "unique": True,
    },
    "total_volume": {
        "words": ["vp", "pore volume", "vtotal", "v total", "volume total", "vt"],
        "units": ["cm3/g", "cm3g-1", "cm3 g-1"],
        "limit_words": False,
        "unique": True,
    },
    "sio2_al2o_ratio_gel": {
        "words": ["sio2/al2o3 gel", "gel siO2/al2o3"],
        "units": ["empty"],
        "limit_words": False,
        "unique": True,
    },
    "sio2_al2o3_ratio": {
        "words": ["sio2/al2o3"],
        "units": ["empty"],
        "limit_words": False,
        "unique": True,
    },
    "si_al_ratio_filtrate": {
        "words": ["si/al filtrate", "filtrate si/al", "Si/Al ﬁltrate"],
        "units": ["empty"],
        "limit_words": False,
        "unique": True,
    },
    "si_al_ratio": {
        "words": ["si/al", "Molar ratio", "si/albulk", "si/ albulk", "Si/ Al"],
        "units": ["empty"],
        "limit_words": False,
        "unique": True,
    },
    "b_l_ratio": {
        "words": ["b/l"],
        "units": ["empty"],
        "limit_words": False,
        "unique": False,
    },
    "l_b_ratio": {
        "words": ["l/b"],
        "units": ["empty"],
        "limit_words": False,
        "unique": False,
    },
    "time": {
        "words": ["time", "period", "t (min)"],
        "units": ["min", "h"],
        "limit_words": False,
        "unique": True,
    },
    "temperature": {
        "words": ["t (k)", "temperature"],
        "units": ["k", "ºc", "°c"],
        "limit_words": False,
        "unique": True,
    },
    "crystallinity": {
        "words": ["crystallinity", "cristallinity"],
        "units": ["%", "empty"],
        "limit_words": False,
        "unique": True,
    },
    "Si": {
        "words": ["si", "nsi"],
        "units": ["wt%", "empty", "umol/g", "mmol/g"],
        "limit_words": True,
        "unique": True,
    },
    "Al": {
        "words": ["al", "nal"],
        "units": ["wt%", "empty", "umol/g", "mmol/g", "umol.g-1"],
        "limit_words": True,
        "unique": True,
    },
    "lewis_sites": {
        "words": ["l", "nlewis", "lewis", "lpy", "pyl", "clewis", "cl"],
        "units": ["μmol/g", "mmol g-1", "umol/g", "umol.g-1", "mmolg-1", "lmol g-1"],
        "limit_words": True,
        "unique": False,
    },
    "bronsted_sites": {
        "words": [
            "b",
            "nbronstead",
            "bronstead",
            "bronsted",
            "bpy",
            "pyh",
            "cbronsted",
            "cb",
        ],
        "units": ["μmol/g", "mmol g-1", "umol/g", "umol.g-1", "mmolg-1", "lmol g-1"],
        "limit_words": True,
        "unique": False,
    },
    "naoh_c": {
        "words": ["naoh", "concentration", "c (m)"],
        "units": ["m", "empty"],
        "limit_words": False,
        "unique": True,
    },
}

EMPTY_VALUES_REGISTRY = set(["-", "-"])


subscript_map = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "ₐ": "A",
    "ₑ": "E",
    "ₕ": "H",
    "ᵢ": "I",
    "ⱼ": "J",
    "ₖ": "K",
    "ₗ": "L",
    "ₘ": "M",
    "ₙ": "N",
    "ₒ": "O",
    "ₚ": "P",
    "ᵣ": "R",
    "ₛ": "S",
    "ₜ": "T",
    "ᵤ": "U",
    "ᵥ": "V",
    "ₓ": "X",
}

superscript_map = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁿ": "N",
    "ᵃ": "A",
    "ᵇ": "B",
    "ᶜ": "C",
    "ᵈ": "D",
    "ᵉ": "E",
    "ᶠ": "F",
    "ᵍ": "G",
    "ʰ": "H",
    "ᶦ": "I",
    "ʲ": "J",
    "ᵏ": "K",
    "ˡ": "L",
    "ᵐ": "M",
    "ⁱ": "I",
    "ᵒ": "O",
    "ᵖ": "P",
    "ʳ": "R",
    "ˢ": "S",
    "ᵗ": "T",
    "ᵘ": "U",
    "ᵛ": "V",
    "ʷ": "W",
    "ˣ": "X",
    "ʸ": "Y",
    "ᶻ": "Z",
}


class ImageParser(BaseModel):
    data_dict: Dict[str, Dict[str, list]] = Field(default_factory=dict)
    data_string: str = ""

    def __init__(self, data_string: Union[str, dict] = "", **data):
        super().__init__(data_string=data_string, **data)
        self._parse_input(data_string)

    def _safe_json_loads(self, s: str):
        s_stripped = s.strip()

        # 1) Normalize keys that have extra/mixed quotes before the colon.
        #    Examples handled: "'10 000':   or  "'key'":   or  "\"'key'\":  etc.
        s = re.sub(r'([{\s,])\s*["\']+\s*([^"\':]+?)\s*["\']+\s*:', r'\1"\2":', s)

        # 2) If it still looks like a Python-dict (single-quoted), convert to JSON double-quotes
        if re.search(r"{\s*'", s) or re.search(r"'\w", s):
            # convert 'some'  ->  "some"
            s = re.sub(r"'\s*([^']*?)\s*'", r'"\1"', s)
            # fallback: convert remaining single quotes to doubles (safe-guard)
            s = re.sub(r"(?<!\\)\'", '"', s)

        # note: I'm not doing a global control-character replacement here because
        # replacing newlines/tabs outside JSON strings can break parsing.
        return json.loads(s)

    def _parse_input(self, input_data: Union[str, dict]):
        print("RAW INPUT DATA:", input_data)
        if isinstance(input_data, dict):
            input_data = self._convert_na_to_null(input_data)
            input_data = self._clean_keys(input_data)
            self.data_dict = self._filter_na_points(input_data)
        else:
            try:
                input_data = input_data.strip()

                # Extract content inside triple backticks
                matches = re.findall(r"```(?:json)?(.*?)```", input_data, re.DOTALL)
                if matches:
                    input_data = matches[0].strip()

                input_data = re.sub(r'"\{\}"', '""', input_data)
                input_data = re.sub(r"\\u208", "", input_data)
                input_data = re.sub(r"\\u00b", "", input_data)

                input_data = re.sub(
                    r'(?<=[:\[,])\s*(?:"N/A"|N/A|NA|NaN|nan|na)\s*(?=[,\]\}])',
                    " null",
                    input_data,
                    flags=re.IGNORECASE,
                )

                parsed_data = self._safe_json_loads(input_data)

                outer_key = None
                if isinstance(parsed_data, dict) and len(parsed_data) == 1:
                    only_key = next(iter(parsed_data))
                    if isinstance(parsed_data[only_key], dict):
                        outer_key = only_key
                        parsed_data = parsed_data[only_key]

                parsed_data = self._restructure_if_needed(
                    parsed_data, outer_key=outer_key
                )
                parsed_data = self._convert_na_to_null(parsed_data)
                parsed_data = self._clean_keys(parsed_data)
                if outer_key is not None and len(parsed_data) == 2:
                    parsed_data = {outer_key: parsed_data}
                self.data_dict = self._filter_na_points(parsed_data)

            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}\nInput was:\n{input_data}")
                self.data_dict = {}

    def _restructure_if_needed(self, data: dict, outer_key: str = None) -> dict:
        if not isinstance(data, dict):
            return data

        # If outer_key is present and the inner dict has more than 3 keys → apply restructuring
        if outer_key and isinstance(data, dict) and len(data) > 3:
            # First key in the dict is assumed to be the x-axis
            inner_keys = list(data.keys())
            x_key = inner_keys[0]
            x_values = data[x_key]

            restructured = {}
            for k in inner_keys[1:]:  # Skip x_key
                v = data[k]
                if isinstance(v, list) and len(v) == len(x_values):
                    restructured[k] = {
                        x_key: x_values,
                        outer_key: v,  # Use outer_key as y-axis label
                    }

            return restructured if restructured else data

        # Otherwise: normal behavior
        x_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        if x_key is None:
            return data

        x_values = data[x_key]
        restructured = {}
        for k, v in data.items():
            if k == x_key:
                continue
            if isinstance(v, list) and len(v) == len(x_values):
                series_name = outer_key if outer_key else k
                restructured[series_name] = {x_key: x_values, k: v}

        return restructured if restructured else data

    def _convert_na_to_null(self, data: Dict) -> Dict:
        def convert_list(vals):
            cleaned = []
            for v in vals:
                if isinstance(v, (int, float)):
                    cleaned.append(v)
                elif isinstance(v, str):
                    stripped = v.strip().lower()
                    if stripped in {"", "na", "n/a", "nan"}:
                        cleaned.append(None)
                    else:
                        try:
                            # Convert numeric strings to float
                            num = float(v)
                            cleaned.append(num)
                        except ValueError:
                            # Remove any non-numeric strings
                            cleaned.append(None)
                elif v is None:
                    cleaned.append(None)
                else:
                    cleaned.append(None)  # Catch anything unexpected
            return cleaned

        converted = {}
        for key, subdict in data.items():
            if isinstance(subdict, dict):
                converted_sub = {
                    k: convert_list(vals) if isinstance(vals, list) else vals
                    for k, vals in subdict.items()
                }
                converted[key] = converted_sub
            else:
                converted[key] = subdict
        return converted

    def _normalize_sub_super_scripts(self, text: str) -> str:
        return "".join(
            subscript_map.get(char, superscript_map.get(char, char)) for char in text
        )

    def _clean_keys(self, data: Dict) -> Dict:
        def clean_key(k: str) -> str:

            k = re.sub(r"\{_?([^{}]+)\}", r"\1", k)

            k = self._normalize_sub_super_scripts(k)

            k = k.replace("_", "").strip()

            return k

        cleaned = {}
        for key, value in data.items():
            new_key = clean_key(key)
            if isinstance(value, dict):
                cleaned[new_key] = self._clean_keys(value)
            else:
                cleaned[new_key] = value
        return cleaned

    """def _filter_na_points(
        self, data: Dict[str, Dict[str, list]]
    ) -> Dict[str, Dict[str, list]]:
        filtered = {}
        for series_name, axes in data.items():
            keys = list(axes.keys())
            if len(keys) < 2:
                continue

            x_key, y_key = keys[0], keys[1]
            x_vals = axes[x_key]
            y_vals = axes[y_key]

            if len(x_vals) != len(y_vals):
                continue

            x_filtered, y_filtered = [], []
            for x, y in zip(x_vals, y_vals):
                if x is not None and y is not None:
                    x_filtered.append(x)
                    y_filtered.append(y)

            filtered[series_name] = {x_key: x_filtered, y_key: y_filtered}

        return filtered"""

    def _filter_na_points(
        self, data: Dict[str, Dict[str, list]]
    ) -> Dict[str, Dict[str, list]]:
        filtered = {}
        for series_name, axes in data.items():

            keys = list(axes.keys())
            if len(keys) < 2:
                continue

            x_key, y_key = keys[0], keys[1]
            x_vals, y_vals = axes[x_key], axes[y_key]
            min_len = min(len(x_vals), len(y_vals))
            x_vals = x_vals[:min_len]
            y_vals = y_vals[:min_len]

            # Filtra pontos N/A
            x_filtered, y_filtered = [], []
            for x, y in zip(x_vals, y_vals):
                if x is not None and y is not None:
                    x_filtered.append(x)
                    y_filtered.append(y)

            # Guarda a série filtrada
            filtered[series_name] = {x_key: x_filtered, y_key: y_filtered}

        return filtered

    def parse(self, data_string: Union[str, dict]):
        self._parse_input(data_string)
        return self.get_data_dict()

    def get_data_dict(self):
        return self.data_dict

    def to_json(self):
        return json.dumps(self.data_dict, indent=4)

    def get_axis_labels(self):
        axis_labels = {}
        for series, axes in self.data_dict.items():
            keys = list(axes.keys())
            if len(keys) >= 2:
                axis_labels[series] = {"x_axis": keys[0], "y_axis": keys[1]}
        return axis_labels
