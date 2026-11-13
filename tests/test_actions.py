import pytest
from pydantic import ValidationError

from xlmexlab.actions import (
    ActionsWithchemicals,
    Chemical,
    CollectLayer,
    Filter,
    MakeSolution,
    Stir,
)
from xlmexlab.parser import ParametersParser, SchemaParser


def test_filter_raises_for_invalid_phase():
    # Those inputs should not raise anything
    Filter()
    Filter(phase_to_keep=None)
    Filter(phase_to_keep="precipitate")
    Filter(phase_to_keep="filtrate")

    with pytest.raises(ValidationError):
        Filter(phase_to_keep="solvent")
    with pytest.raises(ValidationError):
        Filter(phase_to_keep="liquid")
    with pytest.raises(ValidationError):
        Filter(phase_to_keep="solid")


def test_collect_raises_for_invalid_layer():
    # Those inputs should not raise anything
    CollectLayer(layer="aqueous")
    CollectLayer(layer="organic")

    with pytest.raises(ValidationError):
        CollectLayer(layer="DCM")
    with pytest.raises(ValidationError):
        CollectLayer(layer="")
    with pytest.raises(ValidationError):
        CollectLayer(layer="aq.")


def test_makesolution_raises_for_too_few_materials():
    dummy_chemical = Chemical(name="dummy", quantity=["0.1 ml"])

    # Those inputs should not raise anything
    MakeSolution(materials=[dummy_chemical, dummy_chemical])
    MakeSolution(materials=[dummy_chemical, dummy_chemical, dummy_chemical])

    with pytest.raises(ValidationError):
        MakeSolution(materials=[])
    with pytest.raises(ValidationError):
        MakeSolution(materials=[dummy_chemical])


def test_chemical_class():
    string = """Sure! Here's the information extracted from the given paragraph:

    {
    "chemical": "sodium borohydride",
    "amount": [0.25 g, 6.6 mmol],
    "dropwise": True
    }

    Note:

    The chemical name is "sodium borohydride" (NaBH4).
    The amount added is 0.25 g, which is equivalent to 6.6 mmol.
    The addition is done dropwise over a period of 20 minutes."""
    atributes = ["chemical", "dropwise"]
    schema_parser = SchemaParser(atributes_list=atributes)
    schema_parser.model_post_init(None)
    quantity_parser = ParametersParser(time=False, temperature=False, pressure=False)
    quantity_parser.model_post_init(None)
    new_chemical = Chemical()
    schemas = schema_parser.parse_schema(string)
    print(schemas)
    dropwise = new_chemical.get_chemical(schemas[0], schema_parser)
    repetitions = new_chemical.get_quantity(schemas[0], quantity_parser)
    assert len(schemas) == 1
    assert dropwise is True
    assert repetitions == 1
    assert new_chemical.name == "sodium borohydride"
    assert new_chemical.quantity == ["250.0 milligram", "6.6 millimole"]


def test_validate_conditions():
    conditions_parser = ParametersParser()
    conditions_parser.model_post_init(None)
    string = "(3 hours, -20°C)"
    action = Stir(action_name="stir", action_context=string)
    action.validate_conditions(conditions_parser)
    assert action.duration == "180.0 minute"
    assert action.temperature == "-20.0 degree_Celsius"
    assert action.atmosphere is None


def test_validate_chemicals():
    atributes = ["chemical", "dropwise"]
    schema_parser = SchemaParser(atributes_list=atributes)
    schema_parser.model_post_init(None)
    quantity_parser = ParametersParser(time=False, temperature=False, pressure=False)
    quantity_parser.model_post_init(None)
    string = "(water, 2×10 mL)"
    schemas = ["{'chemical': 'water', 'amount': [2×10 mL], 'dropwise': False}"]
    chemicals_info = ActionsWithchemicals.validate_chemicals(
        schemas, schema_parser, quantity_parser, string
    )
    assert chemicals_info.chemical_list[0].name == "water"
    assert chemicals_info.chemical_list[0].quantity == ["10.0 milliliter"]
    assert chemicals_info.dropwise[0] is False
    assert chemicals_info.repetitions == 2
