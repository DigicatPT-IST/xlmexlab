import pytest
from pydantic import ValidationError

from xlmexlab.parser import ActionsParser, Conditions, ParametersParser, SchemaParser


def test_action_parser():
    string = """test
MakeSolution 1
Add 2
DrySolution(3)
Notes:4"""
    parser = ActionsParser()
    parser.model_post_init(None)
    results = parser.parse(string)
    action_list = results["actions"]
    content_list = results["content"]
    assert action_list[0] == "MakeSolution"
    assert action_list[1] == "Add"
    assert action_list[2] == "DrySolution"
    assert action_list[3] == "Notes"
    assert content_list[0] == " 1\n"
    assert content_list[1] == " 2\n"
    assert content_list[2] == "(3)\n"
    assert content_list[3] == ":4"


def test_parameter_parser():
    string = "19 mL, one day at room temperature with -20-25° C, 23 bar and 7 min methyl-1-yl"
    parser = ParametersParser()
    parser.model_post_init(None)
    results: Conditions = parser.get_parameters(string)
    assert results.amount["value"] == ["19.0 milliliter"]
    assert results.temperature[0] == "room temperature"
    assert results.temperature[1] == "-20.0-25.0 degree_Celsius"
    assert results.pressure[0] == "23.0 bar"
    assert results.duration[0] == "7.0 minute"
    assert results.atmosphere == []
    with pytest.raises(ValueError):
        ParametersParser(parser_params_path="wrong_path")


def test_schema_parser():
    string = '{"chemical": "sodium borohydride","amount": [0.25 g, 6.6 mmol],"dropwise": True}'
    atributes = ["chemical", "dropwise"]
    schema_parser = SchemaParser(atributes_list=atributes)
    schema_parser.model_post_init(None)
    result = schema_parser.parse_schema(string)
    chemical = schema_parser.get_atribute_value(result[0], "chemical")
    dropwise = schema_parser.get_atribute_value(result[0], "dropwise")
    assert len(result) == 1
    assert chemical[0] == "sodium borohydride"
    assert dropwise[0] == "True"
    with pytest.raises(ValueError):
        schema_parser.get_atribute_value(result[0], "not valid atribute")
    with pytest.raises(ValidationError):
        SchemaParser(atributes_list=[])
    with pytest.raises(ValidationError):
        SchemaParser(atributes_list=atributes, limiters={})
