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
                        
                        if os.path.exists(os.path.join(root, 'ingredients.txt')):
                            os.remove(os.path.join(root, "ingredients.txt"))

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

                                        with open(os.path.join(root, "ingredients.txt"), 'a') as out:
                                            out.write(ingredient_text + "\n")

def get_steps():
    for r, dirs, _ in os.walk("ALL_RECIPES_without_videos"):
        for d in dirs:
            for root, _, files in os.walk(os.path.join(r, d)):
                for f in files:
                    if f.startswith('recipe.xml'):
                        file_path = os.path.join(root, f)

                        if os.path.exists(os.path.join(root, 'original_steps.txt')):
                            os.remove(os.path.join(root, "original_steps.txt"))
                        if os.path.exists(os.path.join(root, 'updated_steps.txt')):
                            os.remove(os.path.join(root, "updated_steps.txt"))
                        with open(file_path) as recipe:
                            doc = parse(recipe)
                            for child in doc.getroot():
                                if child.tag == 'steps':
                                    for step in child.getchildren():
                                        step_text = step.text.strip()
                                        with open(os.path.join(root, "original_steps.txt"), 'a') as out:
                                            out.write(step_text + "\n")
                                elif child.tag == 'steps_updated':
                                    for step in child.getchildren():
                                        step_text = step.text.strip()
                                        with open(os.path.join(root, "updated_steps.txt"), 'a') as out:
                                            out.write(step_text + "\n")

get_steps()
