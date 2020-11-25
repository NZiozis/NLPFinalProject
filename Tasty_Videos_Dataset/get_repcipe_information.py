import os
import pdb
from xml.etree.ElementTree import parse


def get_ingredients():
    '''
    Gets all the ingredients for a recipe and stores in an
    ingredients.txt file in that directory
    '''
    for r, dirs, _ in os.walk("ALL_RECIPES_without_videos"):
        for d in dirs:
            for root, _, files in os.walk(os.path.join(r, d)):
                for f in files:
                    if f.startswith('recipe.xml'):
                        file_path = os.path.join(root, f)
                        with open(file_path) as recipe:
                            doc = parse(recipe)
                            for child in doc.getroot():
                                if child.tag == "ingredients":
                                    for ingredient in child.getchildren():
                                        ingredient_text = ''
                                        try:
                                            ingredient_text = ingredient.text.split(';')[1].strip()
                                        except:
                                            ingredient_text = ingredient.text

                                        if os.path.exists(os.path.join(root, 'ingredients')):
                                            os.remove(os.path.join(root, "ingredients"))
                                        with open(os.path.join(root, "ingredients.txt"), 'a') as out:
                                            out.write(ingredient_text + "\n")


