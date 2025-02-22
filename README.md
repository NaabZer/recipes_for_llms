# Recipe data extraction using text mining for LLM applications
This is an experimental repository to use Text Mining to turn recipies, especially list of ingredients into a more searchable format.
The idea is to create a more searchable format for recipes, so that you can more efficiently incorporate them with LLM Agents.

Originally a school project, might keep working with it if it works out well.

## Idea
What I want to accomplish, is being able to pass a list of ingredients to a LLM agent, and get a bunch of recipes back that match those ingredients, to be able to query the agent for recipe reccomendations, new recipe creation, etc.
An example would be "I have these ingredients [...], can you give me an recipe that takes less than 30 minutes that uses no more than the ingredients I have.", or "I have these things [...] in my pantry, I'm not sure what to do with them, can you give me some recommentations? I don't mind buying more ingredients"

To make the idea more interesting, something I really want to incorporate is to somehow be able to express prepared ingredients differently from their base ingredients.
For example, if I have chopped tomatoes, I don't want to be reccomended a recipe that requires whole tomatoes (I can't reconstruct the tomatoes), but if I have normal tomatoes, I can chop them to create chopped tomatoes.
Basically creating one directional relations between ingredients, their prepared counter parts and recipes.

