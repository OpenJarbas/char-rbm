from sklearn.model_selection import train_test_split
import joblib
from os.path import basename
from char_rbm import utils
from char_rbm import sampling
from char_rbm.codec import ShortTextCodec, BinomialShortTextCodec
from char_rbm import CharBernoulliRBM, CharBernoulliRBMSoftmax


class CharRBM:
    def __init__(self):
        self.rbm = None
        self.model_kwargs = {}

    def train(self, text_file, test_ratio=0.05, softmax=True,
              n_hidden=180, learning_rate=0.1, batch_size=10,
              weight_cost=0.0001, learning_rate_backoff=False,
              epochs=5, binomial=False, maxlen=20, minlen=1,
              extra_chars=" ", left=False, preserve_case=False):
        """
        Train a character-level RBM on short texts

        :param text_file: A text file to train on, with one instance per line
        :param test_ratio: The ratio of data to hold out to monitor for overfitting
        :param softmax: use softmax visible units
        :param n_hidden: Number of hidden units
        :param learning_rate: learning_rate
        :param batch_size: Size of a (mini)batch. This also controls # of fantasy particles.
        :param weight_cost: Multiplied by derivative of L2 norm on weights. Practical Guide recommends 0.0001 to start
        :param learning_rate_backoff:
        :param epochs: Number of times to cycle through the training data
        :param binomial: Use the binomial text codec (for comma-separated two-part names)
        :param maxlen: Maximum length of strings (i.e. # of softmax units). Longer lines in the input file will be ignored
        :param minlen: Minimum length of strings. Shorter lines in input file will be ignored.
        :param extra_chars: Characters to consider in addition to [a-zA-Z]
        :param left: Pad strings shorter than maxlen from the left rather than the right.
        :param preserve_case: Preserve case, rather than lowercasing all input strings. Increases size of visible layer substantially.

        """
        codec_kls = BinomialShortTextCodec if binomial else ShortTextCodec
        codec = codec_kls(extra_chars, maxlen, minlen, preserve_case, left)
        model_kwargs = {'codec': codec,
                        'n_components': n_hidden,
                        'learning_rate': learning_rate,
                        'lr_backoff': learning_rate_backoff,
                        'n_iter': epochs,
                        'verbose': 1,
                        'batch_size': batch_size,
                        'weight_cost': weight_cost
                        }
        kls = CharBernoulliRBMSoftmax if softmax else CharBernoulliRBM
        self.rbm = kls(**model_kwargs)

        model_kwargs["text_file"] = text_file
        model_kwargs["n_hidden"] = n_hidden
        model_kwargs["softmax"] = softmax
        model_kwargs["left"] = left
        model_kwargs["preserve_case"] = preserve_case

        self.model_kwargs = model_kwargs
        vecs = utils.vectors_from_txtfile(text_file, codec)
        train, validation = train_test_split(vecs, test_size=test_ratio)
        print("Training data shape : " + str(train.shape))
        self.rbm.fit(train, validation)
        return self.rbm

    @staticmethod
    def _stringify_param(name, value):
        prefix = ''
        if "_" in name:
            # e.g. preserve_case -> pc
            name = "".join([a[0] for a in name.split("_")])
        if isinstance(value, bool):
            value = name if value else ''
        elif isinstance(value, float):
            # e.g. learning_rate -> lr_1E-03
            value = name + '_{:.0E}'.format(value)
        else:
            value = name + str(value)
        return prefix + str(value)

    def _pickle_name(self):
        fname = self.model_kwargs["text_file"].split('.')[0].split('/')[-1]
        fname += '_'
        for arg in ['tag', 'batch_size', 'n_hidden', 'softmax', 'lr_backoff', 'preserve_case', 'n_iter',
                    'learning_rate', 'weight_cost', 'left']:
            value = self.model_kwargs.get(arg)
            if value:
                fname += '_' + self._stringify_param(arg, value)
        return fname + '.pickle'

    def save(self, model_path=None):
        """
        Persist trained model to disk

        :param model_path:
        :return:
        """
        model_path = model_path or self._pickle_name()
        if self.rbm is None:
            print("SAVE ERROR: Model not trained/loaded")
        else:
            joblib.dump(self.rbm, model_path)
            print("Saved model to " + model_path)

    def load(self, model_path):
        """
        Load trained model from disk

        :param model_path:
        :return:
        """
        print("Loading model defined at {}".format(model_path))
        self.rbm = joblib.load(model_path)
        self.model_kwargs = {"text_file": basename(model_path).replace(".pickle", ".txt")}

    def sample(self, n_samples=5, every=1, first=-1, iters=10 ** 3,
               energy=False, start_temp=1.0, end_temp=1.0, sil=None):
        """
        Sample short texts from a model

        :param n_samples (int) : How many samples to draw
        :param every: How often to sample.If -1 (default) only sample after the last iteration.
        :param first: Which iteration to draw the first sample at (if --every is provided and this is not, defaults to --every)
        :param iters: How many rounds of Gibbs sampling to perform
        :param energy: Along with each sample generated, print its free energy
        :param start_temp: Temperature for first iteration
        :param end_temp: Temperature at last iteration
        :param sil: data file for silhouettes

        :return: list [samples], or tuple (samples, free_energy) if energy==True
        """
        if self.rbm is None:
            print("SAMPLE ERROR: Model not trained/loaded")
            return []
        if every == -1:
            sample_indices = [iters - 1]
        else:
            first = every if first == -1 else first
            sample_indices = list(range(first, iters, every))
            if sample_indices[-1] != iters - 1:
                sample_indices.append(iters - 1)

        cb = sampling.print_sample_callback

        kwargs = dict(start_temp=start_temp, final_temp=end_temp, sample_energy=energy,
                      callback=cb)
        if sil:
            kwargs['init_method'] = sampling.VisInit.silhouettes
            kwargs['training_examples'] = sil

        vis = sampling.sample_model(self.rbm, n_samples, iters, sample_indices, **kwargs)

        if energy:
            fe = self.rbm._free_energy(vis)
            print('Final energy: {:.2f} (stdev={:.2f})\n'.format(fe.mean(), fe.std()))
            return (vis, fe)
        return vis[1]
