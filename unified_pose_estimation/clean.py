from dataset import UnifiedPoseDataset

dataset = UnifiedPoseDataset(mode='train', loadit=False, name='train', normalized=False)
print(len(dataset))
dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test', normalized=False)
print(len(dataset))

