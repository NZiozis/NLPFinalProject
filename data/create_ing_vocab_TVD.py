import json
import codecs


if __name__ == "__main__":

    ''' # Uncomment this block to get dictionary containing ingredients
    # Get list of all ingredient words
    with open('Tasty_Videos_Dataset/all_recipes_processed.txt', 'r') as recipeFile:
        recipe_data = json.load(recipeFile)
    ingredient_words = set()
    for name, recipe_dict in recipe_data.items():
        ing = recipe_dict["ingredients"]
        ingredient_words.update(ing)
    print("Number of ingredients: ", len(ingredient_words))
    # Create file with dictionary mapping ingredient to index from 0 -> number ingredients
    ingredient_dict = dict()
    idx = 0
    for i in ingredient_words:
        ingredient_dict[i] = idx
        idx+=1
    json = json.dumps(ingredient_dict)
    f = open("ingredient_dict.json","w")
    f.write(json)
    f.close()
    '''

    ''' # Uncomment this block to get dictionary containing steps
    # Get list of all step words
    with open('Tasty_Videos_Dataset/all_recipes_processed.txt', 'r') as recipeFile:
        recipe_data = json.load(recipeFile)
    steps_words = set()
    for name, recipe_dict in recipe_data.items():
        steps = recipe_dict["steps"]
        for step in steps:
            split_step = step.split(' ')
            steps_words.update(split_step)
    print("Number of words in steps: ", len(steps_words))
    # Create file with dictionary mapping words in recipe steps to index from 0 -> number words
    steps_dict = dict()
    idx = 0
    for i in steps_words:
        steps_dict[i] = idx
        idx+=1
    json = json.dumps(steps_dict)
    f = open("steps_vocab_dict.json","w")
    f.write(json)
    f.close()
    '''

    '''# Uncomment this block to get dictionary mapping word index to decoded string
    with open('steps_vocab_dict.json', 'rb') as stepsDictFile:
        steps_vocab_dict = json.load(stepsDictFile)
    idx2word = dict()
    for k,v in steps_vocab_dict.items():
        #print("v ", v, " k ", k)
        idx2word[int(v)] = k #codecs.decode(k)
    f = open("index_to_vocab.json", "w")
    jsonDict = json.dumps(idx2word)
    f.write(jsonDict)
    f.close()
    '''
    
    '''
    # Create dictionary with decoded values
    with open('Tasty_Videos_Dataset/id2word_tasty.txt', 'rb') as idFile:
        data = idFile.read()
    id_dict = eval(data)
    id_dict_decoded = dict()
    for key, val in id_dict.items():
        id_dict_decoded[key] = codecs.decode(val)
    '''

    ''' # Uncomment to create text file with decoded values on each line
    with open('all_words_codecs.txt', 'w') as txtFile:
       for val in id_dict_decoded.values():
           txtFile.write(val+'\n')
    txtFile.close()
    '''

    '''
    with open('fasttext_embeds_codecs.txt', 'r') as embedFile:
        embeds = embedFile.readlines()
    # Create dictionary mapping word to embedding vector (Python list of strings of floats)
    embed_dict = dict()
    for embed in embeds:
        split = embed.split(' ')
        embed_dict[split[0]] = split[1:-1]
    '''

