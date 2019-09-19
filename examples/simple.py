from char_rbm.simple import CharRBM


dataset = "data/black_metal_bands.txt"
rbm = CharRBM()
train = True
if train:
    rbm.train(dataset, preserve_case=True)
    rbm.save()
else:
    model_path = "models/heavy_metal_bands_.pickle"
    rbm.load(model_path)

samples = rbm.sample()
print(samples)
