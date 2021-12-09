"""Search for classes by name and show parents of them"""
import json

import numpy as np
import pandas as pd

import ade_utils as utils

## CSV Columns:
#0 Wordnet name
#1 Name index 
#2 is object counts 
#3 is part counts 
#4 ADE names
#5 Attributes
#6 Has parts
#7 Is part of

#objects = utils.objects.load()
ade_index = utils.adeindex.load()
objects2 = utils.ade_stats.load()


#print(objects.loc['wall'])

while True:
    query = input("🡆  Class to show: ")
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
            print("🡆  Nothing found")
            continue
        print (f"🡆  Found {len(guesses)} guesses:")
        for i in range(len(guesses)):
            print(f"·  [{i:2}] {guesses[i]}")
        index = input("🡆  Which one to choose? ").strip()
        if len(index) == 0: continue
        index = int(index)
        name = guesses[index]
        class_id = utils.adeindex.class_index(ade_index,name)
    
    d = objects2['classes'][class_id]
    print()
    # Parents of class_id, sorted by count
    for p_id, p_count in sorted(d['parents'].items(),key= lambda x:x[1]):
        print(f"- {'NONE' if p_id == -1 else ade_index['objectnames'][p_id]} : {p_count}")
    print("             🡇")
    scenes_str = ' '.join([f"{c}x {s}" for s,c in d['scenes'].items()])
    print(f"{name} : objectcount {d['object_count']} \n({scenes_str})")
    # print("🡇")
    # childstring = line['has_parts'].strip()
    # print(childstring if childstring!="" else "NONE")
    print()

#print(objects.loc[[0]])

# print(not objects.loc[1][7])
# f = open("root_objects.csv","w")
# f.write("name; part of none; no parts; object count; part count\n")

# for i,row in objects.iterrows():
#     part_of_none = not row[7]
#     no_parts = not row[6]
#     if part_of_none or no_parts:
#         f.write(f"{row[0]}; {part_of_none}; {no_parts}; {row[2]}; {row[3]}\n")

# f.close()
