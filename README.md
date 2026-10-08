# **xlmexlab**
[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![GitHub star chart](https://img.shields.io/github/stars/Pocoyo7798/xlmexlab?style=flat-square)](https://star-history.com/#Pocoyo7798/xlmexlab)
[![Open Issues](https://img.shields.io/github/issues-raw/Pocoyo7798/xlmexlab?style=flat-square)](https://github.com/Pocoyo7798/xlmexlab/issues)

xlmexlab is tool built to create and run LLM and VLM based extraction pipelines. It allows you to create loops of prompt - model response - parsing for a Q&A aproach for data extraction, accelariting multiple model comparison and model optimization.
## **Getting Started**

```bash
conda create --name xlmexlab python=3.10
conda activate xlmexlab
git clone git@github.com:Pocoyo7798/xlmexlab.git
cd xlmexlab
pip install -e .
```
## **LLM-based Extraction Pipelines**

In this repository you have multiple extraction pipelines available. Every pipeline comes with pre-defined models, model parameters, model template and prompts. Hence you just pass the requirement to run each extraction command. In the case that you want to test different combinations for an existing extraction pipeline we can play with this 4 different arguments:
```bash
--llm_model_name
--llm_model_parameters_path
--prompt_template_path
--prompt_schema_path
```
Right now we support any hugginface model supported by the [vllm](https://docs.vllm.ai/en/v0.7.0/models/supported_models.html) python library. You can define the model you want to run by passing the model name on huggingface or the model folder on your computer on the ```--llm_model_name``` argument. The improve the performance of any pre-trained model you can rely on two things. First you can set the model parameters as temperature and top p passing a file containing the parameters such a [this one](src/xlmexlab/resources/model_parameters/vllm_default_params.json) on the ```--llm_model_parameters_path``` argument. The performance of a model is also improved by passing the correct prompt template file. Multiple models already have the correct prompt template associated with their name. You can find the list [here](src/xlmexlab/prompt.py) under the ```TEMPLATE_REGISTRY```. If you want to use custom template, download [this file](src/xlmexlab/resources/template/ibm_granite3_default_template.json), change the content of the "template" key, based on what your need and then pass the new prompt template path on the ```--prompt_template_path``` argument.  Finnaly the prompt is defined on a schema file that can be passed on the ```--prompt_schema_path```. The prompt is divided in 6 parts:

* **Expertise**: Define the closest field related to your problem.
* **Initialization**: Give general tips for the LLM answer.
* **Objective**: Problem definition and indentification of what the LLM should do.
* **Definitions**: Define concepts to be used by the LLM.
  * **Initialization**:Indicate for what to use the definitions.
  * **Concepts**:Indicate the concept name and describe it.
* **Answer Schema**: Define the answer format of the LLM.
  * **Initialization**:Indicate the part of the response that should follow the format and describe it.
  * **Format**:Give an example of the format.
* **Conclusion**: End your prompt giving the last details and take the opportunity to be kind.

You can find multiple explamples of prompt schemas [here](src/xlmexlab/resources/schemas). Now lets look at all the pipelines you have available now!
### Paragraph Classification
Paragraph classification consists in identifying as True or False if the paragraph is in a certain class. For example you can use it to paragraphas cotnaining experimental procedures from other paragraphs. You can run it in the following way:
```bash
xlmexlab-paragraph_classifier --type "n2_physisorption" pathe_to_paragraphs.txt path_to_results.txt
```
The input should be a .txt file containing a different paragraph in each line, while the output is also a .txt file with True or False in each line. Right now there are 3 options for the ```--type``` argument implemented: 'n2_physisorption' to identify nitrogen physorption experimental procedures paragraphs, 'ftir_pyridine' to identity FTIR spectroscopy with adsorbed pyridine experimental procedures pragraphs and 'desilication_dealumination' to identify desilication and dealumination experiemetanl procedures paragraphs. You can run it for other by passing a new prompt on the ```--prompt_schema_path``` argument. You can find a example of a clssifying prompt [here](src/xlmexlab/resources/schemas/classify_n2_physisorption_schema.json)
### Action Extraction
Action Extraction consists in extracting a sequence of experimental actions that describe the procedure present in the paragraph. To apply run the following line:
```bash
xlmexlab-text2actions --actions_type materials  path_to_paragraphs.txt path_to_results.txt
```
The input should be a .txt file containing a different paragraph in each line, while the output is also a .txt with a list of actions as python dictionaries. Right now you have 4 options for the ```--actions_type```: 'organic' that uses the the action set defined [here](https://www.nature.com/articles/s41467-020-17266-6), 'pistachio' that uses the same action set from organic with small changes to adapt to the [pistachio dataset](https://www.nextmovesoftware.com/pistachio.html), sac that uses the action set defined [here](https://www.nature.com/articles/s41467-023-43836-5) and "materials" that uses the action set defined [here](https://research.ibm.com/publications/catalysts-synthesis-procedures-extraction-from-synthesis-paragraphs-using-large-language-models). An example of a action extraction prompt is available [here](src/xlmexlab/resources/schemas/material_synthesis_actions_schema.json). Note that if you want to not use any post processing for your model response you need to set the ```--actions_type``` argument to ```Custom```, otherwise the post processing of that field will be aplied on the model response. With the ```--actions_type``` argument set to ```Custom``` the action set sued is controled on the prompt level, allowing you to control the action just by removing and adding actions from the prompt. To augment an existing action set you need to add the new action at the respective action registry [here](src/xlmexlab/actions.py). To create a complete new action set you can contact us for further colaborations. All the actions that can be used right now are present in the following table:

| Action    | Description | Available Words for Prompt |
| -------- | ------- | ------- |
| Add | Add a substance to the mixture | add, disperse |
| AcidTreatment | Treatment were a mterial is mixed with an acid solution | acidtreatment |
| AlkalineTreatment | Treatment were a mterial is mixed with an base solution | alkalinetreatment |
| CollectLayer | Select aqueous or organic fraction(s) | collectlayer, collect |
| Concentrate | Evaporate the solvent | concentrate, evaporate |
| Crystallization | Thermal treatment that generates crystals | crystallization |
| Degas | Purge the  mixture with a gas | degas |
| Dry | Drying of a mixture | dry |
| DrySolid | Dry a solid | drysolid |
| DrySolution | Dry an organic solution with a desiccant | drysolution |
| Extract | Transfer compound into a different solvent | extract |
| Grind | Break the material into smaller particles | grind |
| IonExchange | Treatment were a mterial is mixed with an neutral salt solution | ionexchange, ion-exchange |
| MakeSolution | Mix several substances to generate a mixture or solution
 | makesolution, makemixture |
| NewSolution | Indication that a new soltion started | newsolution, newmixture |
| Partition | Add two immiscible solvents for subsequent phase separation | partition |
| PhaseSeparation | Separate the mixture creating an Filter or Centrifugate action if its about solid-liquid separation | phaseseparation, filter, centrifugate |
| PH | Change the pH of the reaction mixture
 | adjustph |
| Purify | Purification (chromatography) | purify |
| Quench | Stop the reaction by adding a substance or changing temperature | quench |
| Recrystallize | Recrystallize a solid from a solvent or mixture of solvents | recystallize |
| ReduceTemperature | Creates a SetTemperature, if the temperature is not mentions, will set it as "cool" | reducetemperature |
| reflux | Reflux the reaction mixture | reflux |
| Repeat | Indicate repetion of the rpevious actions | repeat |
| Separate | Separate the mixture creating adding the separation type in the action parameters | separate |
| SetTemperature | Change the temperature of the reaction mixture | settemperature, changetemperature |
| Sieve | Screen the particles by size | sieve |
| Sonicate | Agitate the solution with sound waves
 | sonicate |
| ThermalTreatment | Aplication of high temperatures to a material | thermaltreatment, posttreatment, calcination, calcine, carbonize, reduce |
| Transfer | Move the mixture to a new recipient | transfer |
| Triturate | Triturate the residue | triturate |
| Wait | Leave the mixture to stand for a specified duration | wait |
| Wash | Wash the solid after separation | wash |

###Sample Finder
The extraction pipeline is Work in Progress

###Table Extraction
Table extraction consists in extracting data from samples or experiments present in tables. For example, for a table containing characterization data all the characterized properties will be extract for each sample, with the names standardized. You can run it using the following line.
```bash
xlmexlab-table2data --type catalyst_characterization  image_folder_path path_to_results.txt
```
The input should be a folder containing the table images, while the output is a .txt files containing a list of dictionaries containing the info associated to each samples/experiment. Right now, only one option for ```--type``` argument, that is used to extract data from tables containing characterization data from heteregeneous catalysts. To apply it to other kind of tables, you just need to pass a .json file containing the a new table schema on ```--table_schema_path``` argument. The schema should have a structure like [this](src/xlmexlab/resources/schemas/table_extraction_schema.json), where for each type of data you want to extract you need to identify the possible keywords and units associated with it.

###Graphic Extraction

## **Non LLM-based Extraction Pipelines**

In this section we present some extraction pipelines that use only a parser applied directly on a data source. Generally, this pipelines were built to support LLM based pipeline, but since they can also be usefull indivdually we added command line support for them.

###Molar Ratio Extraction
Molar Ratio Extraction consists in finding molar ratios in a paragraph, together with the equation associated if it is the case. You can run it like this:

```bash
xlmexlab-text2molar_ratios --valid_chemicals_path path_to_chemical_file.txt  path_to_paragraphs.txt path_to_results.txt
```
The input is a .txt file containing all the paragraphs that you want to process line by line, while the uput file is also a .txt file containing a list od dcitionaries containing the molar ratio information. The ```--valid_chemicals_path``` argmuent takes a .txt file containing chemicals substances line by line (as they would apperar in a molar ratio). If you do not pass this argument, a pre defined one setted up for zeolites will be used.

## **Model and Settings Evaluation**

There could be the case that you have a new model or a setting combination that could improve an existing. For this, we have an evaluator for each pipleine to speed up LLM screning and parameter optimization. You can already find multiple model paremeter files [here](src/xlmexlab/resources/model_parameters). To do the evaluation you should run the extraction pipeline with the new settings on a reference dataset available [here](src/xlmexlab/resources/datasets). Then the following evaluator are availble:
```bash
xlmexlab-eval_classifier path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
xlmexlab-eval_actions path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
xlmexlab-eval_graphs path_to_reference_file path_to_results_file path_to_evaluation_file.xlxs
```
You can also find the reference files in [here](src/xlmexlab/resources/datasets). Note that each evaluator as a set of paremeters pre defined. If you want to change them run the ```--help``` argument to verify what are the paraemeters available for you to tune.

## **Creating New Pipelines**

The xlmexlab tool comes with a bunch of parsers based on regex for you to create your own extraction tool. In this part we will give you an example on how create one. Note all the files used in the example are [here](src/xlmexlab/resources/example) So, imagine that you want to create a tool to identify the alkaline treatment temperature used in the following procedure:

```diff
"The alkaline treatment process was carried out using a 0.2 mol at 338 K. L-1 NaOH solution. In all experiments, 1 g of ZSM-5 zeolite and 100 mL of solution were used. The duration of the process was limited to 30 and 10 min for conventional electric and microwave (500 W) heating’s, respectively."
```
As you can see even if you look at text you can see that two temeperatures are given. So how can you extract only the alkaline treatment temperature? Lets use LLMs to helps. First we are going to ask to the LLMs to give us the alkaline treatment temperature:

```python
with open("prompt_schema.json", "r") as f:
            prompt_dict = json.load(f)
text = "The alkaline treatment process was carried out using a 0.2 mol at 338 K. L-1 NaOH solution. In all experiments, 1 g of ZSM-5 zeolite and 100 mL of solution were used. The duration of the process was limited to 30 and 10 min for conventional electric and microwave (500 W) heating’s, respectively."
prompt = PromptFormatter(**prompt_dict)
prompt.model_post_init("phi_mini_4k_template.json")
llm_model = ModelLLM(model_name="microsoft/Phi-3-mini-4k-instruct")
llm_model.load_model_parameters("model_parameters.json")
llm_model.vllm_load_model()
final_prompt = prompt.format_prompt(text)
response = llm_model.run_single_prompt(final_prompt).strip()
print(response)
```
With this is loaded the prompt structure and the model parameters. With this setup you can obtain the model response for multiple paragaphs as well if you need to apply it in large scale. After this the model response should be something like this:

```diff
"The temperature of the alkaline treatment is 338 K."
```

Now the alkaline tratment tempereture value is isolated from other temperatures, however other conditions are also present. To solve it we will use a parameters parser to identify only the temperature and convert it to degrees celsius:

```python
from xlmexlab.parser import ParametersParser
temperature_parser = ParametersParser(
            parser_params_path = "parser_parameters.json",
            convert_units=True,
            time=False,
            temperature=True,
            pressure=False,
            amount=False,
            atmosphere=False,
            size=False
        )
temperature = temperature_parser.get_parameters(response).temperature
print(temperature)
```

With this we obtain the temperature in degree celsius:

```diff
['64.85000000000002 degree_Celsius']
```

Obviously different extraction pipelines will require different parser or a combiantion of them. Hence here it is a table containing all the parser available right and a description on how to use them:

| Parser    | Description |
| -------- | ------- |
| ParametersParser  | Used to find values associated with basic units as temperetura, time, mass or volume. For using it you need to set a file like [this](src/xlmexlab/resources/parsing_parameteres/synthesis_parsing_parameters.json) containing the units that you want to consider |
| ListParametersParser | Used to find seuqence of number in text like "The treatment was performe at **10, 30 and 60** min"|
| ComplexParametersParser  | Used to find values associated with complex units (that can be formed from basic one) such as concentration, heating speed, flow rates and stirring speed. For using it you need to set a file like [this](src/xlmexlab/resources/parsing_parameteres/synthesis_parsing_parameters.json) containing the units that you want to consider |
| ActionsParser  | Used to separate action and respective context from text, by passing the respective list of action on your action set |
| KeywordSearching  | Used to find keywords defined in a python list |
| SchemaParser  | Used to find and extract specfific schema patterns that you dfined on your prompt. For example "The chemical substance information is: **{'name': NaOH, 'quantity': 2 g}**" by specifying "{" and "}" as limiter and "name" and "quantity" as atributes you can separate the dicitoonary from the text a get the value in each key.|
| DimensionlessParser  | Used to ding numbers without a unit. |
| MolarRatioFinder  | Used to find molar ratios and other information associeated as equation and relations in text. |
| NumberFinder  | Used to find number independently of having units or not    |
| EquationFinder  |Used to find equations using a,b,c,x,y,z letters and +, /, = signals |
| TableParser  | Parses a table built as a list of python lists, organizing the information based on a regsitry of words |
