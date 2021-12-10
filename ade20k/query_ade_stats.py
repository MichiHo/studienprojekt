"""Search for classes by name interactively and show statistics (parents, counts etc.).

Just enter a class-name and press Enter. If the class-name does not exist exactly like it, 
or if you append ~ to the beginning, it will instead present all class-names containing 
your input and give you a choice."""
import json

import numpy as np
import pandas as pd

import ade_utils as utils

print(__doc__)
print()

ade_index = utils.AdeIndex.load()
try:
    ade_stats = utils.AdeStats.load()
except FileNotFoundError:
    print("ade_stats.pkl file could not be found. You can create it with the script create_ade_stats.py")
    exit()


while True:
    query = input("Class to show: ")
    if not query or len(query) == 0: break
    name = query
    class_id = -1
    if not query.startswith("~"):
        try:
            class_id = ade_index['objectnames'].index(name)
        except ValueError:
            class_id = -1
            
    if class_id == -1:
        if query.startswith("~"): query = query[1:]
        
        guesses = list(filter(lambda name: query in name,ade_index['objectnames']))
        if (len(guesses) == 0):
            print("Nothing found")
            continue
        print (f"Found {len(guesses)} guesses:")
        for i in range(len(guesses)):
            print(f"-  [{i:2}] {guesses[i]}")
        index = input("Choice: ").strip()
        try:
            index = int(index)
            name = guesses[index]
        except (ValueError, IndexError):
            print("No valid choice.")
            continue
        class_id = utils.AdeIndex.class_index(ade_index,name)
    
    d = ade_stats['classes'][class_id]
    print()
    print("##",name)
    print(f"total instances: {d['object_count']}")
    print(f"images containing it: {d['image_count']}")
    print("scenes:",' '.join([f"{c}x {s}" for s,c in d['scenes'].items()]))
    print("parents:")
    # Parents of class_id, sorted by count
    for p_id, p_count in sorted(d['parents'].items(),key= lambda x:x[1],reverse=True):
        print(f"- {'NONE' if p_id == -1 else ade_index['objectnames'][p_id]} : {p_count}")
    print()

