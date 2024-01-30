from utils import dataloaders

opt=dataloaders.load_dataset_yaml('./data/test.yaml')
print(dataloaders.preprocess_dataset(opt))


# print(opt)