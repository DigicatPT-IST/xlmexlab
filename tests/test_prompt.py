from xlmexlab.prompt import PromptFormatter


def test_format_action():
    action_dict1 = {"Initialization": "1", "Add": "2", "Stir": "3"}
    action_dict2 = {"Add": "2", "Stir": "3", "Initialization": "1"}
    action_dict3 = {"Add": "2", "Stir": "3"}
    template1 = PromptFormatter(actions=action_dict1)
    template2 = PromptFormatter(actions=action_dict2)
    template3 = PromptFormatter(actions=action_dict3)
    result12 = "1\n-'Add' : 2\n-'Stir' : 3\n"
    result3 = "-'Add' : 2\n-'Stir' : 3\n"
    assert template1.actions_to_string() == result12
    assert template2.actions_to_string() == result12
    assert template3.actions_to_string() == result3


def test_format_prompt():
    action_dict = {"Initialization": "5"}
    template1 = PromptFormatter()
    template2 = PromptFormatter(
        expertise="1",
        initialization="2",
        objective="3",
        actions=action_dict,
        answer_schema={},
        conclusion="6",
    )
    template1.model_post_init(None)
    template2.model_post_init(None)
    result1 = "No specific objective, just chatting...\n''\n"
    result2 = """1\n2\n3\n'4'\n5\n6"""
    assert template1.format_prompt() == result1
    assert template2.format_prompt(context="4") == result2
