from utils import dataloaders



opt=dataloaders.load_dataset_yaml('./data/test.yaml')

train,val,test=dataloaders.create_dataloader(opt,batch_size=4,workers=0)

print(next(iter(train)))
print(opt)