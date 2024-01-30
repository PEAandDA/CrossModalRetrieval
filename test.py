import os
from utils import dataloaders



opt=dataloaders.load_dataset_yaml('./data/test.yaml')
# dataloaders.preprocess_dataset(opt)
# dataset=dataloaders.ImageTextDataset(dataset_json=os.path.join(opt["output"],opt["train"]),vocab_json= os.path.join(opt["output"],opt["vocab"]),mode= "train",captions_num=opt["captions_num"],max_len=opt["max_len"])
# len(dataset)
# print(dataset[0])

train,val,test=dataloaders.create_dataloader(opt,batch_size=4,workers=0)

print(next(iter(train)))
print(opt)