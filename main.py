"""
Title: main.py

Created on: 2021-07-31

Author: FriscianViales

Encoding: UTF-8

Description: Machine Learning Pipeline implemented using Luigi. The workfow downloads
a Kaggle dataset for image classification between Cats and Dogs, splits the data, build
different Neural Networks and train them in parallel. Finally, it saves the training
results for all the models and build Accuracy/Loss plots for quick evaluation.
"""

# Import libraries:
import luigi
import sys
import os
import zipfile
import requests
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from luigi.format import Nop
from luigi.format import UTF8
from matplotlib import pyplot as plt
from shutil import copyfile

# File configurations:
DISABLE_GPU = False
TRAINING_DATA = os.path.join(os.getcwd(), 'Data', 'Training')
VALIDATION_DATA = os.path.join(os.getcwd(), 'Data', 'Validation')
logging.basicConfig(stream=sys.stdout, level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')
plt.style.use('dark_background')

if DISABLE_GPU:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


class ImportKaggleDataset(luigi.ExternalTask):
    """ Import the Kaggle Dogs vs. Cats Training Data. """
    def output(self):
        return luigi.LocalTarget(path=os.path.join(os.getcwd(), 'Data', 'Raw', 'train.zip'), format=Nop)

    def run(self):
        logging.info(f"Downloading dataset, please wait...")
        data = self.get_data()
        self.write_output(data)

    def get_data(self):
        with requests.Session() as session:
            return session.get(url=f"{self.dataset_url()}")

    def write_output(self, data: requests.Response):
        with self.output().open('wb') as zipped:
            for chunk in data.iter_content(chunk_size=512):
                zipped.write(chunk)

    @staticmethod
    def dataset_url():
        return f"https://storage.googleapis.com/open-ml-datasets/cats-versus-dogs-dataset/train.zip"


class UnZipDataset(luigi.Task):
    """ Unzip dataset in a specific filepath for further processing. """
    def requires(self):
        return ImportKaggleDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(os.getcwd(), 'Checkpoints', f"{self.get_task_family()}.txt"))

    def run(self):
        logging.info('Unzipping dataset, please wait...')
        zipped = zipfile.ZipFile(self.input().path)
        zipped.extractall(os.path.join(os.getcwd(), 'Data', 'Raw'))
        self.write_output()

    def write_output(self):
        with self.output().open('w') as checkpoint:
            checkpoint.write("DONE")


class SplitTrainingDataset(luigi.Task):
    """ Split training dataset into training and validation based on predetermined parameters. """
    TRAINING_SPLIT_PCT = 0.8
    SUBSET_DATA_PCT = 0.5

    def requires(self):
        return UnZipDataset()

    def output(self):
        return luigi.LocalTarget(os.path.join(os.getcwd(), 'Checkpoints', f"{self.get_task_family()}.txt"))

    def write_output(self, df):
        with self.output().open('w') as csv:
            csv.write(df.to_csv(index=False, line_terminator='\n'))

    def run(self):
        logging.info('Processing files, please wait...')
        self.make_folders()
        self.empty_folders()
        df = self.split_dataset()
        self.write_output(df)

    def split_dataset(self, seed=69) -> pd.DataFrame:
        # List with all file names:
        training_images = os.listdir(os.path.join(os.getcwd(), 'Data', 'Raw', 'train'))
        df = pd.DataFrame()
        df['Doggos'] = [x for x in training_images if str(x.split('.')[0]).upper() == 'DOG']
        df['Kitties'] = [x for x in training_images if str(x.split('.')[0]).upper() == 'CAT']
        x = df.shape[0]

        # Split data:
        df = df.sample(int(x * self.SUBSET_DATA_PCT), random_state=seed)
        x = df.shape[0]
        training_batch = df.sample(round(x * self.TRAINING_SPLIT_PCT), random_state=seed).sort_values('Doggos')
        validation_batch = df.loc[~df.index.isin(training_batch.index)].sort_values('Doggos')

        # Move files and return df with all file names:
        self.copy_files(training_batch)
        self.copy_files(validation_batch, is_train=False)
        training_batch['class'] = 'Training'
        validation_batch['class'] = 'Validation'
        return pd.concat([training_batch, validation_batch], ignore_index=True).reset_index(drop=True)

    @staticmethod
    def make_folders():
        # Create folders for splitting training data:
        for i in {'Training', 'Validation'}:
            try:
                os.mkdir(os.path.join(os.getcwd(), 'Data', i))
            except FileExistsError:
                pass

        # Split data by labels too:
        for a in {'Training', 'Validation'}:
            for b in {'Dogs', 'Cats'}:
                try:
                    os.mkdir(os.path.join(os.getcwd(), 'Data', a, b))
                except FileExistsError:
                    pass

    @staticmethod
    def empty_folders():
        # Make sure there are any files on the folders created by this task:
        for a in {'Training', 'Validation'}:
            for b in {'Dogs', 'Cats'}:
                for f in os.listdir(os.path.join(os.getcwd(), 'Data', a, b)):
                    os.remove(os.path.join(os.getcwd(), 'Data', a, b, f))

    @staticmethod
    def copy_files(df: pd.DataFrame, is_train=True):
        if is_train:
            for i, row in df.iterrows():
                old = os.path.join(os.getcwd(), 'Data', 'Raw', 'train', row['Doggos'])
                new = os.path.join(TRAINING_DATA, 'Dogs', row['Doggos'])
                copyfile(old, new)
                old = os.path.join(os.getcwd(), 'Data', 'Raw', 'train', row['Kitties'])
                new = os.path.join(TRAINING_DATA, 'Cats', row['Kitties'])
                copyfile(old, new)
        else:
            for i, row in df.iterrows():
                old = os.path.join(os.getcwd(), 'Data', 'Raw', 'train', row['Doggos'])
                new = os.path.join(VALIDATION_DATA, 'Dogs', row['Doggos'])
                copyfile(old, new)
                old = os.path.join(os.getcwd(), 'Data', 'Raw', 'train', row['Kitties'])
                new = os.path.join(VALIDATION_DATA, 'Cats', row['Kitties'])
                copyfile(old, new)


class ConvolutionalNeuralNetwork(luigi.ExternalTask):
    """ Task that builds a model with a given set of parameters. """
    MODEL_NAME = luigi.Parameter()
    LOSS_FUNCTION = luigi.Parameter()
    PERFORMANCE_METRIC = luigi.Parameter()
    OPTIMZER = luigi.Parameter()
    OPTIMZER_MOMENTUM = luigi.IntParameter()
    OPTIMZER_LR = luigi.IntParameter()
    IMAGE_SIZE = luigi.TupleParameter()
    PREDICTION_CLASSES = luigi.IntParameter()
    CHANNELS = luigi.IntParameter()
    CONV_LAYERS = luigi.IntParameter()
    CONV_FILTERS = luigi.IntParameter()
    NEURONS_DENSE = luigi.IntParameter()
    BATCH_NORM = luigi.BoolParameter()
    DROPOUT = luigi.BoolParameter()

    def output(self):
        model_name = f"{self.get_task_family()}-{self.MODEL_NAME}-V{self.CONV_LAYERS}.h5"
        return luigi.LocalTarget(os.path.join('Models', self.MODEL_NAME, model_name))

    def run(self):
        model = self.build_network()
        model.save(self.output().path)
        logging.info(f"CNN-{self.MODEL_NAME}-V{self.CONV_LAYERS} built. ")

    def build_network(self) -> Sequential:
        # Start with an empty network:
        model = Sequential()

        # Build feature extractor layers:
        for i in range(self.CONV_LAYERS):
            model = self.add_layer(model, offset=i + 1)

        # Add Neural Network to the model:
        model.add(Flatten())
        model.add(Dense(self.NEURONS_DENSE, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=self.get_optimizer(),
            loss=self.LOSS_FUNCTION,
            metrics=[self.PERFORMANCE_METRIC]
        )
        return model

    def input_shape(self) -> tuple:
        return self.IMAGE_SIZE[0], self.IMAGE_SIZE[1], self.CHANNELS

    def add_layer(self, model: Sequential, offset: int) -> Sequential:
        if offset == 1:
            model.add(Conv2D(
                filters=self.CONV_FILTERS,
                kernel_size=(3, 3),
                activation='relu',
                kernel_initializer='he_uniform',
                input_shape=self.input_shape())
            )
        else:
            model.add(Conv2D(
                filters=self.CONV_FILTERS * offset,
                kernel_size=(3, 3),
                activation='relu',
                kernel_initializer='he_uniform')
            )

        if self.DROPOUT:
            model.add(Dropout(rate=0.2))
            model.add(MaxPooling2D(2, 2))
        elif self.BATCH_NORM:
            model.add(BatchNormalization())
            model.add(MaxPooling2D(2, 2))
        else:
            model.add(MaxPooling2D(2, 2))

        return model

    def get_optimizer(self):
        if self.OPTIMZER == 'SGD':
            return optimizers.SGD(
                momentum=self.OPTIMZER_MOMENTUM,
                learning_rate=self.OPTIMZER_LR
            )
        elif self.OPTIMZER == 'ADAM':
            return optimizers.Adam(
                learning_rate=self.OPTIMZER_LR
            )
        else:
            return optimizers.RMSprop(
                momentum=self.OPTIMZER_MOMENTUM,
                learning_rate=self.OPTIMZER_LR
            )


class TrainModelOnCPU(luigi.Task):
    """ Build and train a Convolutional Neural Network. """
    STEPS_PER_EPOCH = luigi.IntParameter(default=100)
    EPOCHS = luigi.IntParameter(default=500)
    VALIDATION_STEPS = luigi.IntParameter(default=100)
    CLASS_MODE = luigi.Parameter(default='binary')
    MODEL_NAME = luigi.Parameter(default='Baseline')
    LOSS_FUNCTION = luigi.Parameter(default='binary_crossentropy')
    PERFORMANCE_METRIC = luigi.Parameter(default='accuracy')
    OPTIMZER = luigi.Parameter(default='RMSprop')
    OPTIMZER_MOMENTUM = luigi.IntParameter(default=0.9)
    OPTIMZER_LR = luigi.IntParameter(default=0.001)
    IMAGE_SIZE = luigi.TupleParameter(default=(150, 150))
    PREDICTION_CLASSES = luigi.IntParameter(default=2)
    CHANNELS = luigi.IntParameter(default=3)
    CONV_LAYERS = luigi.IntParameter(default=3)
    CONV_FILTERS = luigi.IntParameter(default=32)
    NEURONS_DENSE = luigi.IntParameter(default=512)
    BATCH_NORM = luigi.BoolParameter(default=False)
    DROPOUT = luigi.BoolParameter(default=False)

    def requires(self):
        return {
            "Data": SplitTrainingDataset(),
            "Model": ConvolutionalNeuralNetwork(
                MODEL_NAME=self.MODEL_NAME,
                LOSS_FUNCTION=self.LOSS_FUNCTION,
                PERFORMANCE_METRIC=self.PERFORMANCE_METRIC,
                OPTIMZER=self.OPTIMZER,
                OPTIMZER_MOMENTUM=self.OPTIMZER_MOMENTUM,
                OPTIMZER_LR=self.OPTIMZER_LR,
                IMAGE_SIZE=self.IMAGE_SIZE,
                PREDICTION_CLASSES=self.PREDICTION_CLASSES,
                CHANNELS=self.CHANNELS,
                CONV_LAYERS=self.CONV_LAYERS,
                CONV_FILTERS=self.CONV_FILTERS,
                NEURONS_DENSE=self.NEURONS_DENSE,
                BATCH_NORM=self.BATCH_NORM,
                DROPOUT=self.DROPOUT
            )
        }

    def output(self):
        return luigi.LocalTarget(self.model_name(trained=True))

    def run(self):
        model = load_model(self.model_name())
        training = self.train_model(model)
        self.save_training_metrics(training)
        model.save(self.output().path)

    def train_model(self, model: Sequential) -> Sequential:
        return model.fit(
            x=self.training_data_feed(),
            validation_data=self.validation_data_feed(),
            steps_per_epoch=self.STEPS_PER_EPOCH,
            epochs=self.EPOCHS,
            validation_steps=self.VALIDATION_STEPS,
            callbacks=[self.early_stop()]
        )

    def training_data_feed(self) -> image.ImageDataGenerator:
        return image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True
        ).flow_from_directory(
            directory=TRAINING_DATA,
            batch_size=int(self.batch_size()[0] / self.STEPS_PER_EPOCH),
            class_mode=self.CLASS_MODE,
            target_size=self.IMAGE_SIZE
        )

    def validation_data_feed(self) -> image.ImageDataGenerator:
        return image.ImageDataGenerator(
            rescale=1.0 / 255.0
        ).flow_from_directory(
            directory=VALIDATION_DATA,
            batch_size=int(self.batch_size()[1] / self.STEPS_PER_EPOCH),
            class_mode=self.CLASS_MODE,
            target_size=self.IMAGE_SIZE
        )

    def batch_size(self) -> tuple:
        with self.input().get('Data').open('r') as csv:
            df = pd.read_csv(csv)
            return df.loc[df['class'] == 'Training'].shape[0], df.loc[df['class'] == 'Validation'].shape[0]

    def save_training_metrics(self, training_results):
        # Fetch metrics:
        df = pd.DataFrame(training_results.history)
        df.insert(0, 'model_name', self.MODEL_NAME)
        df = df.reset_index().rename(columns={"index": "epoch"})

        try:
            os.mkdir(os.path.join(os.getcwd(), 'TrainingResults'))
        except FileExistsError:
            pass

        # Save results:
        model_name = f"ConvolutionalNeuralNetwork-{self.MODEL_NAME}-V{self.CONV_LAYERS}-Trained.csv"
        df.to_csv(os.path.join(os.getcwd(), 'TrainingResults', model_name), index=False, line_terminator='\n')

    def model_name(self, trained=False) -> str:
        if trained:
            model_name = f"ConvolutionalNeuralNetwork-{self.MODEL_NAME}-V{self.CONV_LAYERS}-Trained.h5"
            return os.path.join('Models', self.MODEL_NAME, model_name)
        else:
            model_name = f"ConvolutionalNeuralNetwork-{self.MODEL_NAME}-V{self.CONV_LAYERS}.h5"
            return os.path.join('Models', self.MODEL_NAME, model_name)

    @staticmethod
    def early_stop():
        return tf.keras.callbacks.EarlyStopping(
            patience=25
        )


class TrainAllModels(luigi.WrapperTask):
    """ Task that triggers parallel training. """
    def requires(self):
        yield TrainModelOnCPU(
            MODEL_NAME='BaselineNet-50x50Res',
            IMAGE_SIZE=(50, 50)
        )
        yield TrainModelOnCPU(
            MODEL_NAME='CNN-50x50Res-BatchNorm-RMSprop',
            IMAGE_SIZE=(50, 50),
            BATCH_NORM=True,
        )
        yield TrainModelOnCPU(
            MODEL_NAME='CNN-50x50Res-BatchNorm-Adam',
            IMAGE_SIZE=(50, 50),
            BATCH_NORM=True,
            OPTIMZER='ADAM'
        )
        yield TrainModelOnCPU(
            MODEL_NAME='CNN-50x50-BatchNorm-SGD',
            IMAGE_SIZE=(50, 50),
            BATCH_NORM=True,
            OPTIMZER='SGD'
        )


class EvaluateAllModels(luigi.Task):
    """Evaluate models after they have completed training. """
    def requires(self):
        return TrainAllModels()

    def output(self):
        return luigi.LocalTarget(
            path=os.path.join(os.getcwd(), 'TrainingSummary', f"{self.get_task_family()}.csv"),
            format=UTF8
        )

    def write_output(self, df: pd.DataFrame):
        with self.output().open('w') as checkpoint:
            checkpoint.write(df.to_csv(index=False, line_terminator='\n'))

    def run(self):
        rows = []
        for x in os.listdir(os.path.join(os.getcwd(), 'TrainingResults')):
            for i, row in self.read_csv(os.path.join(os.getcwd(), 'TrainingResults', x)).iterrows():
                rows.append(row.to_dict())

        df = pd.DataFrame(rows)
        self.plot_results(df)
        self.write_output(df)

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        with open(path, 'r') as csv:
            return pd.read_csv(csv)

    @staticmethod
    def plot_results(df: pd.DataFrame):
        models = df.model_name.unique()
        colors = plt.cm.get_cmap('Paired').colors

        try:
            os.mkdir(os.path.join(os.getcwd(), 'TrainingSummary'))
        except FileExistsError:
            pass

        # Plot #1:
        plt.figure(figsize=(20, 10))
        for model, color in zip(models, colors):
            plt.title("Training and Validation Accuracy")
            epochs = sorted(df.loc[df.model_name == model, 'epoch'].unique())
            y1 = df.loc[df.model_name == model, 'accuracy']
            y2 = df.loc[df.model_name == model, 'val_accuracy']
            plt.plot(epochs, y1, c=color, ls='--', label=f"{model}-Accuracy")
            plt.plot(epochs, y2, c=color, ls='-', label=f"{model}-ValAccuracy")
            plt.legend(loc="lower right")
        plt.savefig(os.path.join(os.getcwd(), 'TrainingSummary', 'Accuracy.png'))

        # Plot #1:
        plt.figure(figsize=(20, 10))
        for model, color in zip(models, colors):
            plt.title("Training and Validation Loss")
            epochs = sorted(df.loc[df.model_name == model, 'epoch'].unique())
            y1 = df.loc[df.model_name == model, 'loss']
            y2 = df.loc[df.model_name == model, 'val_loss']
            plt.plot(epochs, y1, c=color, ls='--', label=f"{model}-Loss")
            plt.plot(epochs, y2, c=color, ls='-', label=f"{model}-ValLoss")
            plt.legend(loc="upper right")
        plt.savefig(os.path.join(os.getcwd(), 'TrainingSummary', 'Loss.png'))
        plt.show()


class RunAll(luigi.WrapperTask):
    """ Pipeline entrypoint."""
    def requires(self):
        yield EvaluateAllModels()


if __name__ == '__main__':
    luigi.build(tasks=[RunAll()], local_scheduler=True, workers=1)
