import json
import random
random.seed(2023)


with open('data_identifiers.json','r') as f:
    valid = json.load(f)

print(len(valid))

idxes = list(valid.keys())

# 20%
test_idx=random.sample(idxes,int(len(idxes)*0.2))

train_val_idx = [idx for idx in idxes if idx not in test_idx]

# 10%
val_idx = random.sample(train_val_idx,int(len(train_val_idx)*0.125))

# 70%
train_idx = [idx for idx in train_val_idx if idx not in val_idx]
train_idx = {idx:ids for idx, ids in valid.items() if idx in train_idx}

val_idx = {idx:ids for idx, ids in valid.items() if idx in val_idx}
test_idx = {idx:ids for idx, ids in valid.items() if idx in test_idx}

print(len(train_idx),len(val_idx),len(test_idx))
print(len(train_idx)/len(valid)*100,len(val_idx)/len(valid)*100,len(test_idx)/len(valid)*100)

with open('train_idx.json','w') as f:
    json.dump(train_idx,f)

with open('val_idx.json','w') as f:
    json.dump(val_idx,f)

with open('test_idx.json','w') as f:
    json.dump(test_idx,f)
