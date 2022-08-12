import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

class PrecisionAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(PrecisionAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.img_size = img_size
        self.thre = thre
        self.metric_mask =  metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.precision_angle = self.add_weight(name='precision_angle',initializer='zeros')
        self.true_positive = self.add_weight(name='true_positive',initializer='zeros')
        self.pred_positive = self.add_weight(name='pred_positive',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128        
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize binary 0/1
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize binary 0/1
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize

        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate precision'
        true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        predicted_positives = K.sum(K.round(chemo_pred))
        # precision = (true_positives) / (predicted_positives + K.epsilon())
        # self.precision_angle.assign(precision)
        self.true_positive.assign_add(true_positives)
        self.pred_positive.assign_add(predicted_positives)
        self.precision_angle.assign(self.true_positive/(self.pred_positive+K.epsilon()))

    def result(self):
        return self.precision_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.precision_angle.assign(0.)
        self.true_positive.assign(0.)
        self.pred_positive.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_positive':K.eval(self.true_positive),
                'pred_positive':K.eval(self.pred_positive),
                'precision_angle':K.eval(self.precision_angle)
            }
        )
        return config

class RecallAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(RecallAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.recall_angle = self.add_weight(name='recall_angle',initializer='zeros')
        self.true_positive = self.add_weight(name='true_positive',initializer='zeros')
        self.label_positive = self.add_weight(name='label_positive',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate recall'
        true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        possible_positives = K.sum(K.round(chemo_true))
        # recall = (true_positives) / (possible_positives + K.epsilon())
        # self.recall_angle.assign(recall)
        self.true_positive.assign_add(true_positives)
        self.label_positive.assign_add(possible_positives)
        self.recall_angle.assign(self.true_positive/(self.label_positive + K.epsilon()))

    def result(self):
        return self.recall_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.recall_angle.assign(0.)
        self.true_positive.assign(0.)
        self.label_positive.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_positive':K.eval(self.true_positive),
                'label_positive':K.eval(self.label_positive),
                'recall_angle':K.eval(self.recall_angle)
            }
        )
        return config

class F1ScoreAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(F1ScoreAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.f1_angle = self.add_weight(name='f1_angle',initializer='zeros')
        self.pred_positive = self.add_weight(name='pred_positive',initializer='zeros')
        self.true_positive = self.add_weight(name='true_positive',initializer='zeros')
        self.label_positive = self.add_weight(name='label_positive',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate recall'
        true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        possible_positives = K.sum(K.round(chemo_true))
        # recall = (true_positives) / (possible_positives + K.epsilon())
        'calculate precision'
        # true_positives = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        predicted_positives = K.sum(K.round(chemo_pred))
        # precision = (true_positives) / (predicted_positives + K.epsilon())
        'calculate f1-score'
        # f1 = 2*precision*recall/(precision+recall)

        # self.precision_angle.assign(precision)
        self.true_positive.assign_add(true_positives)
        self.pred_positive.assign_add(predicted_positives)
        self.label_positive.assign_add(possible_positives)
        self.f1_angle.assign(2*self.true_positive/(self.label_positive+self.pred_positive+K.epsilon()))

    def result(self):
        return self.f1_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.f1_angle.assign(0.)
        self.true_positive.assign(0.)
        self.label_positive.assign(0.)
        self.pred_positive.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_positive':K.eval(self.true_positive),
                'label_positive':K.eval(self.label_positive),
                'pred_positive':K.eval(self.pred_positive),
                'f1_angle':K.eval(self.f1_angle)
            }
        )
        return config

class AccuracyAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(AccuracyAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.accuracy_angle = self.add_weight(name='accuracy_angle',initializer='zeros')
        self.true_pred = self.add_weight(name='true_pred',initializer='zeros')
        self.total_element = self.add_weight(name='total_element',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'calculate accuracy'
        true_pred = K.sum(tf.where(tf.math.equal(K.round(chemo_true), K.round(chemo_pred)),1.0,0.0))
        total_element = K.sum(K.round(chemo_true))+K.sum(1-K.round(chemo_true))
        # acc = (true_pred) / (total_element + K.epsilon())
        self.true_pred.assign_add(true_pred)
        self.total_element.assign_add(total_element)
        # self.accuracy_angle.assign(acc)
        self.accuracy_angle.assign(self.true_pred/(self.total_element+K.epsilon()))
        
    def result(self):
        self.accuracy_angle

        return self.accuracy_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accuracy_angle.assign(0.)
        self.true_pred.assign(0.)
        self.total_element.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'true_pred':K.eval(self.true_pred),
                'total_element':K.eval(self.total_element),
                'accuracy_angle':K.eval(self.accuracy_angle)
            }
        )
        return config

class DiceAngle(Metric):
    def __init__(self, sparse_matrix, batch_size, img_size, thre, metric_mask, angle_range_split_matrix, angle_interval, **kwargs):
        super(DiceAngle, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.batch_size = batch_size
        self.thre = thre
        self.img_size = img_size
        self.metric_mask = metric_mask # batchSize*128*128
        self.angle_interval = angle_interval
        self.angle_range_split_matrix = angle_range_split_matrix # 12*360*batchSize
        self.dice_angle = self.add_weight(name='dice_angle',initializer='zeros')
        self.intersection = self.add_weight(name='intersection',initializer='zeros')
        self.union = self.add_weight(name='union',initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '# numpy calculation'
        # sparseM = []
        # sparseM = tf.sparse.from_dense(K.constant(sparse.csr_matrix.todense(self.sparse_matrix[str([64,64])]))) # 360*16384
        sparseM = tf.sparse.from_dense(K.constant(self.sparse_matrix))
        '# tensor calculation'
        y_pred_ = tf.squeeze(y_pred)*K.constant(self.metric_mask) # batchSize*128*128
        y_true_ = tf.squeeze(y_true)*K.constant(self.metric_mask) # batchSize*128*128      
        y_pred_ = tf.where(tf.math.greater_equal(y_pred_,self.thre), y_pred_, 0) # batchSize*128*128
        y_true_ = tf.where(tf.math.greater_equal(y_true_,self.thre), y_true_, 0) # batchSize*128*128
        y_pred_ = tf.transpose(y_pred_, perm=(1,2,0)) # 128*128*batchSize
        y_true_ = tf.transpose(y_true_, perm=(1,2,0)) # 128*128*batchSize

        chemo_pred = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_pred_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        chemo_true = tf.sparse.sparse_dense_matmul(sparseM, tf.reshape(y_true_,[self.img_size*self.img_size,self.batch_size])) # 360*batchSize
        full_ones = tf.ones(tf.shape(chemo_pred))
        chemo_pred = tf.where(tf.math.greater(chemo_pred,0), full_ones, chemo_pred) # 360*batchSize
        chemo_true = tf.where(tf.math.greater(chemo_true,0), full_ones, chemo_true) # 360*batchSize
        chemo_pred = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_pred, axis=1)/self.angle_interval # 12*batchSize
        chemo_true = tf.reduce_sum(K.constant(self.angle_range_split_matrix)*chemo_true, axis=1)/self.angle_interval # 12*batchSize
        # chemo_pred_storage.append(chemo_pred)
        # chemo_true_storage.append(chemo_true)
        # chemo_pred_ = tf.reshape(chemo_pred_storage,(self.batch_size,360,1))  # batchSize*360*1
        # chemo_true_ = tf.reshape(chemo_true_storage,(self.batch_size,360,1))  # batchSize*360*1
        'dice accuracy'
        intersection = K.sum(K.round(chemo_true) * K.round(chemo_pred))
        union = K.sum(K.round(chemo_pred)) + K.sum(K.round(chemo_true))
        # dice = (2*intersection) / (union + K.epsilon())
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
        # self.dice_angle.assign(dice)
        self.dice_angle.assign((2*self.intersection)/(self.union+K.epsilon()))

    def result(self):
        return self.dice_angle

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice_angle.assign(0.)
        self.intersection.assign(0.)
        self.union.assign(0.)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sparse_matrix':self.sparse_matrix,
                'batch_size':self.batch_size,
                'thre':self.thre,
                'img_size':self.img_size,
                'metric_mask':self.metric_mask,
                'angle_interval':self.angle_interval,
                'angle_range_split_matrix':self.angle_range_split_matrix,
                'intersection':K.eval(self.intersection),
                'union':K.eval(self.union),
                'dice_angle':K.eval(self.dice_angle)
            }
        )
        return config