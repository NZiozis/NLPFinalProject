import json

if __name__ == "__main__":
	# Get list of all ingredient words
	with open('Tasty_Videos_Dataset/all_recipes_processed.txt', 'r') as recipeFile:
		recipe_data = json.load(recipeFile)

	ingredient_words = set()
	for name, recipe_dict in recipe_data.items():
		ing = recipe_dict["ingredients"]
		ingredient_words.update(ing)
	print("Number of ingredients: ", len(ingredient_words))

	# Check if ingredients in id2word_tasty dict
	only_words = []
	with open('Tasty_Videos_Dataset/id2word_tasty.txt', 'r') as idFile:
		words = idFile.readlines()
	words = words[0]
	split_entries = words.split('\'')
	for entry in split_entries:
		if entry[0].isalnum():
			only_words.append(entry)
	# TODO: Clean elements of punctuation
	with open('all_words.txt', 'w') as txtFile:
		for elt in only_words:
			txtFile.write(elt+'\n')
	txtFile.close()