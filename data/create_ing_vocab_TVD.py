import json
import codecs

if __name__ == "__main__":
	# Get list of all ingredient words
	with open('Tasty_Videos_Dataset/all_recipes_processed.txt', 'r') as recipeFile:
		recipe_data = json.load(recipeFile)

	ingredient_words = set()
	for name, recipe_dict in recipe_data.items():
		ing = recipe_dict["ingredients"]
		ingredient_words.update(ing)
	print("Number of ingredients: ", len(ingredient_words))

	# Create dictionary with decoded values
	with open('Tasty_Videos_Dataset/id2word_tasty.txt', 'rb') as idFile:
		data = idFile.read()
	id_dict = eval(data)
	id_dict_decoded = dict()
	for key, val in id_dict.items():
		id_dict_decoded[key] = codecs.decode(val)

	# Create text file with decoded values on each line
	with open('all_words_codecs.txt', 'w') as txtFile:
		for val in id_dict_decoded.values():
			txtFile.write(val+'\n')
	txtFile.close()