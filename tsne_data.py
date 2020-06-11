import numpy as np
# you need to change this to your path
tsne_data = np.load('collected_data/SAE_15.npz')


# the dictionary consists of the following four items:
# 'behaviors': the label used for baseline binary classification.
# 'features': numpy embedding vectors
# 'behaviors_names': list consists of stage type of each frame: 'anomaly', 'reaction', 'misbehavior', 'healing', 'gap', 'normal'
# 'misbehavior_names': list consists of misbehavior types or ''

print(tsne_data['behaviors'].shape, tsne_data['features'].shape, tsne_data['behaviors_names'].shape, tsne_data['misbehavior_names'].shape)
