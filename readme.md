# char_rbm

Train and sample character-level RBMs for short text

Read [Dreaming of names with RBMs](https://colinmorris.github.io/blog/dreaming-rbms) for more information


## Install

    pip install char_rbm
    
## Usage

See the [examples](./examples) folder for more usage scripts

```python
from char_rbm.simple import CharRBM

dataset = "data/heavy_metal_bands.txt"
model_path = "models/heavy_metal_bands.pickle"
rbm = CharRBM()
train = True
if train:
    rbm.train(dataset, preserve_case=True)
    rbm.save(model_path)
else:
    rbm.load(model_path)

samples = rbm.sample()
print(samples)
```

### Example Scripts

To train a small model on first names:

    wget http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/other/names.txt
    python train.py --maxlen 10 --extra-chars '' --hid 100 names.txt
    python sample.py names__nh100.pickle

This should give you some output like...

    wietzer     
    sarnimono   
    buttheo     
    ressinosoo  
    bernington

### Interpreting training output

During training, you'll see debug output like...

    [CharBernoulliRBMSoftmax] Iteration 3/5 t = 14.46s
    Pseudo-log-likelihood sum: -115047.96   Average per instance: -2.13
    E(vali):        -14.00  E(train):       -14.07  difference: 0.07
    Fantasy samples: moll$$$$$$|anderd$$$$|gronbel$$$

Without going into too much detail, the pseudo-log-likelihood (-2.13 above), is a pretty decent estimation of how well the model is currently fitting the training data. The lower the better.

The next line compares the energy assigned to the training data vs. the validation set. The difference (0.07 in this case) gives an idea of how much the model is overfitting. The higher the difference, the worse. A difference of 0 implies no overfitting.

The final line has string representions of a few of the "fantasy particles" used for the persistent contrastive divergence training.


## Credits

All credits to ColinMorris, original repo [here](https://github.com/colinmorris/char-rbm)

I simply packaged this for ease of use and made it python3 compatible
