import numpy as np
import math
from scipy import sparse
from tensorflow.keras.layers import  Input, Conv2D, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Metrics import PrecisionAngle, RecallAngle, F1ScoreAngle, AccuracyAngle, DiceAngle
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import argparse

class Network(object):
    def __init__(self, epoch, batchSize, sparse_matrix, thre, angle_interval=30, img_rows = 128, img_cols = 128, nClass=2):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.nClass = nClass
        self.batchSize = batchSize
        self.epochs = epoch
        'generate a fixed binary mask with center [64,64]'
        mask = np.ones((self.img_rows,self.img_cols))
        for i in range(self.img_rows):
            for j in range(self.img_cols):
                d = math.sqrt((i-int(self.img_rows/2))*(i-int(self.img_rows/2))+(j-int(self.img_cols/2))*(j-int(self.img_cols/2)))
                r =int(np.round(d))
                if r<=5:
                    mask[i,j] = 0
                elif r>=int(self.img_rows/2):#40:
                    mask[i,j] = 0
        mask = np.repeat(np.expand_dims(mask,axis=0),self.batchSize,axis=0)
        self.metric_mask = mask
        'generate angle range split matrix (interval=30deg)'
        self.angle_interval = angle_interval
        angle_range_split_matrix = []
        for ite in range(int(360/angle_interval)):
            matrix = np.zeros((360,self.batchSize))
            matrix[ite*angle_interval:(ite+1)*angle_interval,] = 1
            angle_range_split_matrix.append(matrix)
        angle_range_split_matrix = np.array([sample for sample in angle_range_split_matrix]) # 12*360*batchSize
        self.angle_range_split_matrix = angle_range_split_matrix
        'sparse matrix'
        self.sparse_matrix = sparse.csr_matrix.todense(sparse_matrix['[64, 64]'])
        self.thre = thre
        'initialize metrics'
        self.PrecisionAngle1 = PrecisionAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='lipid')
        self.RecallAngle1 = RecallAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='lipid')
        self.F1ScoreAngle1 = F1ScoreAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='lipid')
        self.AccuracyAngle1 = AccuracyAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='lipid')
        self.DiceAngle1 = DiceAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='lipid')
        self.PrecisionAngle2 = PrecisionAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='calcium')
        self.RecallAngle2 = RecallAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='calcium')
        self.F1ScoreAngle2 = F1ScoreAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='calcium')
        self.AccuracyAngle2 = AccuracyAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='calcium')
        self.DiceAngle2 = DiceAngle(self.sparse_matrix,self.batchSize,self.img_rows,self.thre,self.metric_mask,self.angle_range_split_matrix,self.angle_interval,name='calcium')

    def net(self):
        'build your network'
        inputs = Input(shape=(self.img_rows, self.img_cols ,self.subVolSlice,1),name='data') # None*128*128*7*1
        conv1_1 = Conv2D(64, kernel_size=3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',kernel_regularizer=regularizers.l2(1e-4))(inputs) # None*128*128*7*64 
        BatchNorm1_1 = BatchNormalization(axis=-1, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(conv1_1)
        outputs_1 = Conv2D(1, 2, activation = 'sigmoid', padding = 'valid', kernel_initializer = 'he_normal')(BatchNorm1_1)#sigmoid    # None*128*128*1*1
        outputs_2 = Conv2D(1, 2, activation = 'sigmoid', padding = 'valid', kernel_initializer = 'he_normal')(BatchNorm1_1)#sigmoid    # None*128*128*1*1

        model = Model(inputs = [inputs], outputs = [outputs_1,outputs_2])
        self.model = model

        predImg1_metrics = [self.PrecisionAngle1,self.RecallAngle1,self.F1ScoreAngle1,self.AccuracyAngle1,self.DiceAngle1]
        predImg2_metrics = [self.PrecisionAngle2,self.RecallAngle2,self.F1ScoreAngle2,self.AccuracyAngle2,self.DiceAngle2]
        model.compile(optimizer = Adam(learning_rate = 1e-4), loss = ['binary_crossentropy','binary_crossentropy'], loss_weights=[1.0,1.0], metrics=[predImg1_metrics,predImg2_metrics])
        
        return model

    def train(self, model_save_path, logs_save_path, data_path_train, data_path_validation):

            model = self.net()

            'your data generator used for training'
            training_generator = DataGenerator(data_path_train,**kwargs)
            'your data generator used for validation'
            validation_generator = DataGenerator_test(data_path_validation, **kwargs)

            my_callbacks = [
                TensorBoard(log_dir=logs_save_path, histogram_freq=5, write_graph=True),
                ModelCheckpoint(os.path.join(model_save_path, 'model_{epoch:03d}.hdf5'),
                            monitor='val_loss',verbose=1,save_best_only=True)
            ]
            

            History = model.fit(training_generator,  epochs=self.epochs, validation_data=validation_generator, validation_freq=1, callbacks=my_callbacks
                        ,max_queue_size=12,workers=6, verbose=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to train the model.')

    parser.add_argument(
        '--model_save_path',
        type=str,
        required=True,
        help='The save path of model.'
    )

    parser.add_argument(
        '--logs_save_path',
        type=str,
        required=True,
        help='The save path of metrics logs.'
    )

    parser.add_argument(
        '--data_path_train',
        type=str,
        required=True,
        help=' The path of training dataset '
    )

    parser.add_argument(
        '--data_path_validation',
        type=str,
        required=True,
        help=' The path of validation dataset '
    )

    parser.add_argument(
        '-e',
        '--epoch',
        type=int,
        default=100,
        required=False,
        help='Number of epochs to train the model')

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=16,
        required=False,
        help='batch size.'
    )

    parser.add_argument(
        '--sparse_matrix',
        default=np.load(r'.\lumenCenter_sampleLine_sparseMetrix.npy',allow_pickle=True).item(),
        required=False,
        help='sparse metrix used for angle determination'
    )

    parser.add_argument(
        '--thre',
        type=float,
        default=0.8,
        required=False,
        help='probability threshold of pixels on predicted CT image'
    )

    args = parser.parse_args()

    umodel = Network(epochs=args.epoch, 
                        batchSize=args.batch_size,
                        sparse_matrix=args.sparse_matrix,
                        thre=args.thre
                        )

    umodel.train2(args.model_save_path, args.logs_save_path, args.data_path_train, args.data_path_validation)

