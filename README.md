# Recipes for LLM's
## Recipe data extraction using text mining for LLM applications
This is an experimental repository to use Text Mining to turn recipies, especially list of ingredients into a more searchable format.
The idea is to create a more searchable format for recipes, so that you can more efficiently incorporate them with LLM Agents.

Originally a school project, might keep working with it if it works out well.

## Idea
What I want to accomplish, is being able to pass a list of ingredients to a LLM agent, and get a bunch of recipes back that match those ingredients, to be able to query the agent for recipe reccomendations, new recipe creation, etc.
An example would be "I have these ingredients [...], can you give me an recipe that takes less than 30 minutes that uses no more than the ingredients I have.", or "I have these things [...] in my pantry, I'm not sure what to do with them, can you give me some recommentations? I don't mind buying more ingredients"

To make the idea more interesting, something I really want to incorporate is to somehow be able to express prepared ingredients differently from their base ingredients.
For example, if I have chopped tomatoes, I don't want to be reccomended a recipe that requires whole tomatoes (I can't reconstruct the tomatoes), but if I have normal tomatoes, I can chop them to create chopped tomatoes.
Basically creating one directional relations between ingredients, their prepared counter parts and recipes.

**TODO**:
- [x] Set up [doccano](https://github.com/doccano/doccano) for tagging
- [X] Set up [label-studio](https://github.com/HumanSignal/label-studio) because doccano doesn't work properly
- [X] Train a NER tagging model using [spaCy](https://github.com/explosion/spaCy)
- [ ] Set up MLops pipeline for tagging/training/serving NER model
    - [X] Make MLflow training work to continiuously improve NER model
    - [ ] Set up Prefect orchestration
        - [ ] Periodically check for new labeling data
        - [ ] Download and preprocess data
        - [ ] Train NER model using spaCy/MLflow
        - [ ] Serve new model to API
    - [X] Set up automatic tagging API for doccano using served NER model
- [ ] Create baselines for evaluations:
    - [X] A basic Bag-of-Words search/database
    - [ ] Pre-trained embeddings using some model, e.g. BERT
- [ ] Create evaluation tasks
    - [ ] Find a subset of recipes where a specific query would find the recipe to allow evaluation using metrics
    - [ ] Create some search queries that can then be benchmarked using time measurement
    - [ ] Think of something more?
- [ ] Define data structure for:
    - [ ] DuckDB + Parquet
    - [ ] GraphDB
- [ ] Use trained NER and POS tagging to extract data from ingredient lists into:
    - [ ] DuckDB + Parquet
    - [ ] GraphDB
- [ ] Evaluate
    - [ ] Run the evaluation tasks on the different evaluation tasks, collect metrics on performance and evaluate methods
- [ ] Take the best model, create a simple LLM Agent PoC and subjectively assess usefulness
    - [ ] Cook some recipes based on LLM queries

### NER Tagging
To extract different features from the text, that can be used to create the proper databases, the following tags were created, using the domain knowledge I posses:
- **Quantity** - Self explanatory
- **Unit** - Self explanatory
- **Food** - The actual food item. Here some judgement needs to be considered, is brown sugar different enough from sugar to be classified as a different food item? Are tomatoes and cherry tomatoes? My own guidelines are as follows: If it can be substituted without much flavour difference, they should be considered the same food, so my judgement for the above two examples, are brown and normal sugar are not both "sugar", but tomatoes and cherry tomatoes are both "tomatoes".
- **Variety** - Continuing from the last point, this is what you would label "cherry" as, in cherry tomatoes. This is basically to show that it's different enough that you might want to separate it, but in most cases it does not matter.
- **Preparation** - These are irreversible cooking steps. For example, "chopped", "ground". We want to be able to match tomatoes to chopped tomatoes, but not vice-versa.
- **Alteration** - These are reversible cooking steps. For example, heated, frozen, etc.
- **Brand** - Brands on things. Might be useful for some specific searches.
- **Optional** - This just tags any words that implies the ingredient is optional, so that it does not need to be included in a search
- **State** - This one is probably the hardest to define properly. I've mostly used this for states that changes naturally and are not very important for the recipe as a whole. This are things like "Fresh" tomatoes. I've also put descriptors of size in this category, which might be a mistake. Like "Large" egg. This also goes against the "changes naturally" description, as the size will not change.

Out of these, the most important ones for creating a database and searching for recipes using ingredients, will be mainly **Food**, where **Variety** and **Preparation** can be specified to limit your search space if you only have for example chopped tomatoes.
But of course, the rest of the tags would be present in the database and would also be possible to do filtering on, but it probably doesn't provide much. The tags are there more for the NER tagger to be able to diffirentiate critical vs non-critical attributes (like **Preparation** vs **Alteration**).

#### Tagging pipeline
The tagging is done using [doccano](https://github.com/doccano/doccano), the first dataset is a modified version of [TASTEset](https://github.com/taisti/TASTEset), where some of the above tags are present, and some are changed.
Therefore the labels have had to be relabeled.
Doccano supports API-labeling, so after the first NER model has been trained, an API will be created and then used to automatically tag ingredient sentences, and then you can readjust if neccessary.

