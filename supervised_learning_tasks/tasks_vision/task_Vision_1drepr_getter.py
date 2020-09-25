import numpy as np
import time
import os


class Vision1dRepr():
    def __init__(self, dataset: str = "MNIST", _type: str = "PCA", n_components: int = 200):
        allowed_types = ["PCA", "tSNE", "resnet"]
        if _type not in allowed_types:
            print(f"ERROR: type must be one of {allowed_types}, but is {_type}.")
            raise ValueError
        if _type == 'tSNE':
            if n_components >= 4:
                error_string = f"ERROR: n_components for tSNE must be smaller than 4, but is {n_components}. "
                error_string += f"Using 3 components instead."
                print(error_string)
                n_components = 3
        if _type == 'resnet':
            n_components = 2048  # resnet always has 2048 components

        self.type = _type
        self.n_components = n_components
        self.dataset = dataset

        dirname = os.path.dirname(__file__)
        dirname = os.path.join(dirname, f'1d_reprs_{self.type}')
        filename = os.path.join(dirname, f'{self.dataset}_x_train_repr_1d_{n_components}components.npy')
        self.filename = filename

    def get_repr_computed(self, x_train: np.ndarray):
        start = time.time()
        print("Starting encoding to 1d-repr at time 0.")

        if self.type == "PCA":
            repr = self.compute_repr_PCA(x_train)
        elif self.type == "tSNE":
            repr = self.compute_repr_tSNE(x_train)
        elif self.type == "resnet":
            repr = self.compute_repr_resnet(x_train, start)

        print(f'Ended encoding at time {time.time() - start}.')

        return repr

    def generate_repr_to_file(self, x_train):
        repr = self.get_repr_computed(x_train)
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        np.save(self.filename, repr)
        return repr

    def get_repr_from_file(self, x_train):
        try:
            repr = np.load(self.filename)
        except Exception as e:
            print(f"Error opening file {self.filename}: {e}")
            repr = self.generate_repr_to_file(x_train)

        return repr

    def plot_repr(self, y_train):
        n_samples = len(y_train)
        # n_samples = 7000
        sampleIDs = list(range(n_samples))

        embeddings = self.get_repr_from_file([])
        labels_one_hot = y_train
        labels = np.argmax(labels_one_hot, axis=1)
        N_classes = labels_one_hot.shape[1]

        print("Ended loading data, starting plotting")

        import matplotlib.pyplot as plt
        if embeddings.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D

        cmap = plt.cm.tab10

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(10):
            x = embeddings[labels == i]
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=np.asarray(cmap(int(i)))[np.newaxis, :], label=str(int(i)))
        ax.legend()
        ax.grid(True)

        filename = os.path.splitext(self.filename)[0] + 'png'
        plt.savefig(filename)
        plt.show()

    def compute_repr_tSNE(self, x_train: np.ndarray):

        x_train_withPCA = self.compute_repr_PCA(x_train, 200)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=self.n_components, verbose=True)
        x_train_withTSNE = tsne.fit_transform(x_train_withPCA)

        return x_train_withTSNE

    def compute_repr_PCA(self, x_train: np.ndarray, n_components: int = -1):
        if n_components == -1:
            n_components = self.n_components

        # flatten to 2 dimensions
        x_train_flattened = np.reshape(x_train, (x_train.shape[0], -1))

        # standard scaling
        from sklearn.preprocessing import StandardScaler
        x_train_standardized = StandardScaler().fit_transform(x_train_flattened)

        # perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        x_train_withPCA = pca.fit_transform(x_train_standardized)

        return x_train_withPCA

    def compute_repr_resnet(self, x_train: np.ndarray, startTime: float):
        from tensorflow.keras.applications import resnet50
        from tensorflow import stack, concat, squeeze
        from tensorflow.compat.v1.image import resize_images
        from tensorflow.keras.models import Model

        # resize function
        target_size = 224

        def encode_samples(x_sample):
            batch_size = 1000
            no_samples = x_sample.shape[0]
            index = 0
            x_features = []
            while index < no_samples:
                print("encoding sample %d at time %f" % (index, (time.time() - startTime)))
                end = min(index + batch_size, no_samples)
                x = x_sample[index:end]
                if x.shape[-1] == 1:  # add 2 dimensions to make rgb sample out of black-white-sample
                    x = concat((x,) * 3, axis=-1)
                x = squeeze(x)
                x = resize_images(x, size=(target_size, target_size))
                x = resnet50.preprocess_input(x)
                x_feat = intermediate_layer_model.predict(x, workers=16)
                x_features += [x_feat]
                index += batch_size
            x_feat = concat(x_features, axis=0)
            return x_feat

        # encode function
        model = resnet50.ResNet50(weights='imagenet', input_shape=(target_size, target_size, 3))
        layer_name = 'avg_pool'
        intermediate_layer_model = Model(inputs=model.inputs,
                                         outputs=model.get_layer(layer_name).output)

        # intermediate_layer_model.compile(loss="mse",optimizer="sgd")

        def encode(x_samples):
            encoded_samples = encode_samples(x_samples)
            return encoded_samples

        # encode training data
        x_train_repr_1d = encode(x_train)

        return x_train_repr_1d
