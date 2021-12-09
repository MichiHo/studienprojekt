import ade_utils as utils

ade_index = utils.AdeIndex.load()


while True:
    imgid = int(input("Enter img id: "))
    
    for prop in ["folder","filename","scene"]:
        print(f" - {prop} = {ade_index[prop][imgid]}")
    print(" - contains:")
    
    for i in range(utils.num_classes):
        c = ade_index['objectPresence'][i,imgid]
        if c > 0:
            print(f"   {c:3}x {utils.AdeIndex.classname(ade_index,i)}")