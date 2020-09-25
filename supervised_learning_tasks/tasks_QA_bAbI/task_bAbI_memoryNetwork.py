from collections import namedtuple
from typing import List
from functools import reduce
import tarfile
import re

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from tensorflow import convert_to_tensor
import numpy as np

from supervised_learning_tasks.task_supervised_keras import TaskKeras


class TaskBabiMemoryNetwork(TaskKeras):

    def __init__(self, dataset: str = 'single_supporting_fact', verbose_init: bool = False):
        if dataset not in ['single_supporting_fact','two_supporting_facts']:
            raise ValueError
        self.dataset = dataset
        TaskKeras.__init__(self, verbose_init=verbose_init)

    def get_x_train(self, sample_IDs: List[int] = "all") -> List[np.ndarray]:
        if sample_IDs == "all":
            return [self.inputs_train, self.queries_train]
        else:
            inputs_train = np.concatenate([self.inputs_train[i, None] for i in sample_IDs])
            queries_train = np.concatenate([self.queries_train[i, None] for i in sample_IDs])
            return [inputs_train, queries_train]

    def get_y_train(self, sampleIDs: List[int] = "all") -> np.ndarray:
        if sampleIDs == "all":
            return self.answers_train
        else:
            return np.concatenate([self.answers_train[i, None] for i in sampleIDs])

    def get_x_test(self) -> List[np.ndarray]:
        return [self.inputs_test, self.queries_test]

    def get_y_test(self) -> np.ndarray:
        return self.answers_test

    def get_loss_function(self):
        return losses.SparseCategoricalCrossentropy

    def get_samples_repr_1d(self, sample_IDs: List[int] = 'all') -> dict:
        if not hasattr(self, "samples_repr_1d"):
            inputs_train, queries_train = self.get_x_train()
            model_repr_1d = Model(inputs=[self.model.layers[0].input, self.model.layers[1].input],
                                  outputs=self.model.get_layer("repr_1d").output)
            samples_repr_1d = model_repr_1d.predict([inputs_train, queries_train])
            self.samples_repr_1d = np.reshape(samples_repr_1d, newshape=(samples_repr_1d.shape[0], -1))
        if not isinstance(sample_IDs, str) or not sample_IDs == 'all':
            samples_repr_1d = np.concatenate([self.samples_repr_1d[i, None] for i in sample_IDs])
        else:
            samples_repr_1d = self.samples_repr_1d
        return samples_repr_1d

    def define_model(self, params: dict = None):

        if params == None:
            params = dict()
        regularization_factor_l1 = params.get("regularization_factor_l1", 5.36e-5)
        regularization_factor_l2 = params.get("regularization_factor_l2", 1.38e-4)
        dropout_rate = params.get("dropout_rate", 0.265)
        learning_rate = params.get("learning_rate", 0.0038)
        optimizer = params.get("optimizer", RMSprop)
        lstm_neurons = int(params.get("lstm_neurons", 32))
        kernel_regularizer = l1_l2(regularization_factor_l1, regularization_factor_l2)

        # placeholders
        input_sequence = Input((self.data_params.story_maxlen,))
        question = Input((self.data_params.query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=self.data_params.vocab_size,
                                      output_dim=64))
        input_encoder_m.add(Dropout(dropout_rate))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=self.data_params.vocab_size,
                                      output_dim=self.data_params.query_maxlen))
        input_encoder_c.add(Dropout(dropout_rate))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=self.data_params.vocab_size,
                                       output_dim=64,
                                       input_length=self.data_params.query_maxlen))
        question_encoder.add(Dropout(dropout_rate))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded], name="repr_1d")

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = Dropout(learning_rate)(answer)
        answer = LSTM(lstm_neurons, kernel_regularizer=kernel_regularizer)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(learning_rate)(answer)
        answer = Dense(self.data_params.vocab_size, kernel_regularizer=kernel_regularizer)(
            answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation('softmax')(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        optimizer = optimizer(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def model_fit(self, x_train, y_train, batch_size=2, epochs=32, verbose=False, withAugmentation=False):

        return self.model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              workers=1,
                              use_multiprocessing=0)

    def train_with_hyperopt(self, no_random_samples: int = 1000, no_iterations: int = 100):
        from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials, space_eval

        param_space = dict()
        # optimizers = [SGD, Adam,Adagrad,RMSprop]
        # optimizers = optimizers[1:2]
        names = ["regularization_factor_l1", "regularization_factor_l2", "dropout_rate", "learning_rate",
                 "optimizer", "lstm_neurons", "no_epochs", "batch_size"]
        param_space[names[0]] = hp.loguniform(names[0], np.log(1e-5), np.log(9e-4))
        param_space[names[1]] = hp.loguniform(names[1], np.log(1e-5), np.log(9e-4))
        param_space[names[2]] = hp.uniform(names[2], 0.1, 0.3)
        param_space[names[3]] = hp.loguniform(names[3], np.log(1e-4), np.log(90e-4))
        # param_space[names[4]] = hp.choice(names[4], optimizers)
        param_space[names[5]] = hp.loguniform(names[5], np.log(12), np.log(32 + 1), q=1)
        param_space[names[6]] = hp.loguniform(names[6], np.log(20), np.log(50 + 1), q=1)
        # param_space[names[7]] = hp.loguniform(names[7], np.log(1), np.log(8+1), q=1)

        # train
        x_test = self.get_x_test()
        y_test = self.get_y_test()
        no_training_samples = len(self.get_x_train()[0])

        def objective_function(hyperparam_dict: dict, verbose=False) -> float:
            self.model = self.define_model(hyperparam_dict)
            # randomly sample subset
            sample_IDs = np.random.choice(range(no_training_samples), size=no_random_samples, replace=False)
            no_epochs = int(hyperparam_dict.get('no_epochs', self.get_no_epochs()))
            batch_size = int(hyperparam_dict.get('batch_size', 32))
            loss, acc = self.train_on_batch(sample_IDs=sample_IDs, epochs=no_epochs, resetWeights=False,
                                            batch_size=batch_size)
            print("loss:" + str(loss) + " acc: " + str(acc) + " hyperparams: " + str(hyperparam_dict))
            if np.isnan(loss):
                print("ERROR: validation loss is nan with following hyperparams:")
                print(hyperparam_dict)
                raise ValueError
            return -1 * acc

        # perform optimization
        # minimize the objective over the space
        from hyperopt import fmin, atpe, tpe
        best = fmin(objective_function, param_space, algo=atpe.suggest, max_evals=no_iterations, verbose=True)
        print("best params: " + str(best))
        bestAcc = objective_function(best, verbose=True)
        print("best validation acc:" + str(bestAcc))
        # print("weights:" + str(list(self.model.get_weights()[0])))

    def get_dataset(self, verbose_init):
        try:
            path = get_file('babi-tasks-v1-2.tar.gz',
                            origin='https://s3.amazonaws.com/text-datasets/'
                                   'babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
                  '.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact': 'tasks_1-20_v1-2/en-10k/qa1_'
                                          'single-supporting-fact_{}.txt',
            # QA2 with 10,000 samples
            'two_supporting_facts': 'tasks_1-20_v1-2/en-10k/qa2_'
                                        'two-supporting-facts_{}.txt',
        }
        challenge = challenges[self.dataset]

        if verbose_init:
            print('Extracting stories for the challenge:', self.dataset)
        with tarfile.open(path) as tar:
            train_stories = self.get_stories(tar.extractfile(challenge.format('train')))
            test_stories = self.get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
        DataParams = namedtuple('data_params', 'vocab_size story_maxlen query_maxlen')
        self.data_params = DataParams(vocab_size, story_maxlen, query_maxlen)

        if verbose_init:
            print('-')
            print('Vocab size:', vocab_size, 'unique words')
            print('Story max length:', story_maxlen, 'words')
            print('Query max length:', query_maxlen, 'words')
            print('Number of training stories:', len(train_stories))
            print('Number of test stories:', len(test_stories))
            print('-')
            print('Here\'s what a "story" tuple looks like (input, query, answer):')
            print(train_stories[0])
            print('-')
            print('Vectorizing the word sequences...')

        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        self.inputs_train, self.queries_train, self.answers_train = self.vectorize_stories(train_stories, word_idx)
        self.inputs_test, self.queries_test, self.answers_test = self.vectorize_stories(test_stories, word_idx)

        if verbose_init:
            print('-')
            print('inputs: integer tensor of shape (samples, max_length)')
            print('inputs_train shape:', self.inputs_train.shape)
            print('inputs_test shape:', self.inputs_test.shape)
            print('-')
            print('queries: integer tensor of shape (samples, max_length)')
            print('queries_train shape:', self.queries_train.shape)
            print('queries_test shape:', self.queries_test.shape)
            print('-')
            print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
            print('answers_train shape:', self.answers_train.shape)
            print('answers_test shape:', self.answers_test.shape)
            print('-')

    def tokenize(self, sent):
        '''Return the tokens of a sentence including punctuation.

        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''
        return [x.strip() for x in re.split(r'(\W+)+', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format

        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.

        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data
                if not max_length or len(flatten(story)) < max_length]
        return data

    def vectorize_stories(self, data, word_idx):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([word_idx[w] for w in story])
            queries.append([word_idx[w] for w in query])
            answers.append(word_idx[answer])
        return (pad_sequences(inputs, maxlen=self.data_params.story_maxlen),
                pad_sequences(queries, maxlen=self.data_params.query_maxlen),
                np.array(answers))

    def get_no_epochs(self) -> int:
        return 100
