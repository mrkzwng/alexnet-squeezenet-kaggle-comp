import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt

'''
preprocessing
'''
def is_num(obj):
    # checks if item is number
    try:
        float(obj)
        return(True)
    except:
        return(False)


def is_int(k):
    # checks if number is integer for random_search
    return(True if int(k) == k else False)


def bands_to_imgs(band_series):
    '''
    takes a pandas series object of image bands
    and transforms it to Nx75x75 numpy array,

    N = number of images
    '''
    bands = band_series.reset_index(drop=True)
    bands = bands.apply(lambda x: np.array(x))
    bands = bands.apply(lambda x: x.reshape([1, 75, 75]))
    imgs = np.concatenate(bands, axis=0)

    return(imgs)


def bands_to_channels(band1_series, band2_series):
    '''
    takes pandas series representations of image
    bands and transforms them into an Nx75x75x2
    numpy array

    N = number of observations
    '''
    band1 = bands_to_imgs(band1_series)
    band2 = bands_to_imgs(band2_series)
    b1 = band1.reshape([band1.shape[0], 75, 75, 1])
    b2 = band2.reshape([band2.shape[0], 75, 75, 1])
    bands = np.concatenate([b1, b2], axis=3)

    return(bands)


def imgs_to_channels(band1, band2):
    '''
    for augmented data
    takes numpy arrays of image bands
    with dimensions of Nx75x75 and 
    transforms them into Nx75x75x2

    N = number of observations
    '''
    band1 = band1.reshape([band1.shape[0], 75, 75, 1])
    band2 = band2.reshape([band2.shape[0], 75, 75, 1])
    bands = np.concatenate([band1, band2], axis=3)

    return(bands)
    

def get_rotated_imgs(band_series, angle=22.5):
    '''
    takes a pandas series object of image bands
    and create rotated images to augment data
    '''
    imgs = bands_to_imgs(band_series)
    rotated_imgs = []

    for img in imgs:
        rotated = [rotate(img, 
                          angle=i, 
                          mode='reflect', 
                          reshape=False).reshape([1, 75, 75])
                   for i in np.arange(0, 360, angle)]
        rotated_imgs = rotated_imgs + rotated

    rotated_imgs = np.concatenate(rotated_imgs, axis=0)

    return(rotated_imgs)


def augment_by_angle(data, angle=22.5):
    '''
    augment data via rotation
    '''
    # get rotated images in numpy array form
    rotated_band1 = get_rotated_imgs(data.band_1, angle)
    rotated_band2 = get_rotated_imgs(data.band_2, angle)
    # prepare matching images with labels
    augment_factor = rotated_band1.shape[0] / data.shape[0]
    y = data.is_iceberg.as_matrix()
    y = np.repeat(y, augment_factor, axis=0)
    angles = data.inc_angle.replace(to_replace='na', value=0)
    angles = angles.as_matrix()
    angles = np.repeat(angles, augment_factor, axis=0)
    X = imgs_to_channels(rotated_band1, rotated_band2)
    
    return(X, y, angles)

'''
training functions
'''
def cv_split(X_train, y_train, k_partitions=5):
    '''
    splits data into lists of k partitions
    '''
    part_size = np.ceil(X_train.shape[0] / float(k_partitions))
    part_size = part_size.astype(np.int32)
    split_X = [X_train[i * part_size:(i+1) * part_size] 
               for i in xrange(k_partitions)]
    split_y = [y_train[i * part_size:(i+1) * part_size] 
               for i in xrange(k_partitions)]
    
    return(split_X, split_y)


def get_minibatches(X, y, batch_size=32, n_steps=64):
    '''
    splits data into lists of batches via given batch_size
    '''
    part_size = batch_size * n_steps
    n_batches = np.ceil(X.shape[0] / float(part_size))
    n_batches = n_batches.astype(np.int32)
    batches_X = [X[i * part_size: (i+1) * part_size] 
                 for i in xrange(n_batches)]
    batches_y = [y[i * part_size: (i+1) * part_size]
                 for i in xrange(n_batches)]
    
    return(batches_X, batches_y)


def cross_validate(model_fn, X_train, y_train, 
                   model_hyperparams, model_dir,
                   k_folds=2, epochs=1, batch_size=32,
                   summary_steps=64):
    '''
    cross-validates a tf.estimator.Estimator
    collects 
    ---Arguments---
    
    + model_fn:
      tf.estimator.Estimator model function
      in which the model's graph is defined
      
    + X_train, y_train: 
      training data
    
    + model_hyperparams: 
      dict of {parameter: (min, max)}
      values for random search
      
    + model_dir: 
      directory to save each fold's model
      and performance data
      named after hyperparameter settings
      
    + k_folds:
      number of folds for cross-validation
    
    + epochs:
      number of epochs of training

    + batch_size:
      batch size, duh
      
    + summary_steps:
      number of steps between each round of
      model evaluation
    '''    
    # grab cross-validation partitions
    X_split, y_split = cv_split(X_train, y_train, k_folds)
    
    # set performance history accumulators
    header = ['train_loss', 'valid_loss'] + \
             ['train_true_positives', 'train_false_positives', 
              'train_false_negatives', 'train_precision', 'train_recall', 
              'train_AUC', 'train_mean_per_class_acc', 
              'train_accuracy'] + \
             ['valid_true_positives', 'valid_false_positives', 
              'valid_false_negatives', 'valid_precision', 'valid_recall', 
              'valid_AUC', 'valid_mean_per_class_acc', 
              'valid_accuracy'] + \
             ['n_batches', 'n_epochs', 'n_folds'] + \
             model_hyperparams.keys()
    performance = pd.DataFrame(columns=header)

    # for each fold of cross-validation
    for cv_id, (X_valid, y_valid) in enumerate(zip(X_split, y_split)):
        # get training partitions
        X, y = zip(*[(X_spl, y_spl) for i, (X_spl, y_spl)
                     in enumerate(zip(X_split, y_split)) 
                     if i != cv_id])
        X, y = np.concatenate(X, axis=0), \
               np.concatenate(y, axis=0)
            
        # shuffle training partition X, y
        rand_indices = np.random.choice(list(range(X.shape[0])),
                                        size=X.shape[0],
                                        replace=False)
        X = X[rand_indices, :, :, :]
        y = y[rand_indices]
        
        # create minibatches from X, y
        X_batches, y_batches = get_minibatches(X, y, batch_size, 
                                               n_steps=summary_steps)
        
        # set cv directory
        directory = model_dir + '/cv_' + str(cv_id)
        # initialize model
        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=directory,
                                       params=model_hyperparams)
        # for each epoch
        for epoch_id in xrange(epochs):
        
            # for each minibatch:
            for part_id, (X_batch, y_batch) \
                           in enumerate(zip(X_batches, y_batches)):
                # define training function
                train_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={'x': X_batch},
                                    y=y_batch,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_epochs=1)
                # train
                model.train(input_fn=train_fn)

                # define evaluation function for train/validation data
                train_eval_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={'x': X},
                                    y=y,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_epochs=1)
                valid_eval_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={'x': X_valid},
                                    y=y_valid,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_epochs=1)
                # define predict function for grabbing losses
                train_pred_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={'x': X},
                                    y=y,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_epochs=1)
                valid_pred_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={'x': X_valid},
                                    y=y_valid,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_epochs=1)
                # grab per-minibatch summaries from model.evaluate()
                train_metrics = model.evaluate(
                                    input_fn=train_eval_fn)
                valid_metrics = model.evaluate(
                                    input_fn=valid_eval_fn)
                # compute loss from predictions
                tr_raw_preds = list(model.predict(
                                    input_fn=train_pred_fn))
                val_raw_preds = list(model.predict(
                                    input_fn=valid_pred_fn))
                train_preds = np.array([raw for raw
                              in tr_raw_preds]).reshape(
                                        [X.shape[0], -1])
                valid_preds = np.array([raw for raw
                              in val_raw_preds]).reshape(
                                        [X_valid.shape[0], -1])
                n_labels = 2 if len(y.shape) == 1 \
                             else y.shape[1]
                train_loss = log_loss(y_true=y,
                                      y_pred=train_preds,
                                      labels=np.arange(n_labels))
                valid_loss = log_loss(y_true=y_valid,
                                      y_pred=valid_preds,
                                      labels=np.arange(n_labels))
                # write to performance dataframe
                perf_data = {'train_'+metric: train_metrics[metric]
                             for metric in train_metrics.keys()}
                perf_data.update({'valid_'+metric: valid_metrics[metric]
                                  for metric in valid_metrics.keys()})
                perf_data.update(model_hyperparams)
                perf_data.update(train_loss=train_loss,
                                 valid_loss=valid_loss,
                                 n_epochs=epoch_id+1,
                                 n_batches=(epoch_id+1)*((1+part_id)*summary_steps),
                                 n_folds=cv_id+1)
                performance = performance.append(perf_data,
                                                 ignore_index=True)
        
        # save summary to disk
        performance.to_csv(path_or_buf=directory+'performance.csv', 
                           index=False)
            
    return(performance)


def learning_curve_info(model, train_X, train_y,
                        train_IA, valid_IA,
                        valid_X, valid_y, model_name,
                        num_epochs=30, step_size=2):
    '''
    grab learning curve info via defined 
    tf.estimator.Estimator model
    '''
    header = ['train_loss', 'valid_loss',
              'train_true_positives', 'train_false_positives', 
              'train_false_negatives', 'train_precision', 
              'train_recall', 'train_AUC', 
              'train_mean_per_class_acc', 'train_accuracy',
              'valid_true_positives', 'valid_false_positives', 
              'valid_false_negatives', 'valid_precision', 
              'valid_recall', 'valid_AUC', 
              'valid_mean_per_class_acc', 'valid_accuracy',
              'n_epochs']
    performance = pd.DataFrame(columns=header)

    for ep_id in np.arange(0, num_epochs+1, step_size):
        train_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': train_X,
                           'angles': train_IA},
                        y=train_y,
                        batch_size=32,
                        shuffle=True,
                        num_epochs=step_size)
        model.train(input_fn=train_fn)
        
        train_eval_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': train_X,
                           'angles': train_IA},
                        y=train_y,
                        batch_size=32,
                        shuffle=False,
                        num_epochs=1)
        valid_eval_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': valid_X,
                           'angles': valid_IA},
                        y=valid_y,
                        batch_size=32,
                        shuffle=False,
                        num_epochs=1)
        
        train_metrics = model.evaluate(input_fn=train_eval_fn)
        valid_metrics = model.evaluate(input_fn=valid_eval_fn)
        
        metrics = {'train_'+metric: train_metrics[metric] 
                   for metric in train_metrics.keys()}
        metrics.update({'valid_'+metric: valid_metrics[metric]
                        for metric in train_metrics.keys()})
        metrics.update(n_epochs=ep_id)
        performance = performance.append(metrics, ignore_index=True)

    
    performance.to_csv(path_or_buf='./performance/'+model_name+'.csv', 
                       index=False)
        
    return(model, performance)


def learning_curve_info_no_IA(model, train_X, train_y,
                             valid_X, valid_y, model_name,
                             num_epochs=30, step_size=2):
    '''
    grab learning curve info via defined 
    tf.estimator.Estimator model
    '''
    header = ['train_loss', 'valid_loss',
              'train_true_positives', 'train_false_positives', 
              'train_false_negatives', 'train_precision', 
              'train_recall', 'train_AUC', 
              'train_mean_per_class_acc', 'train_accuracy',
              'valid_true_positives', 'valid_false_positives', 
              'valid_false_negatives', 'valid_precision', 
              'valid_recall', 'valid_AUC', 
              'valid_mean_per_class_acc', 'valid_accuracy',
              'n_epochs']
    performance = pd.DataFrame(columns=header)

    for ep_id in np.arange(0, num_epochs+1, step_size):
        train_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': train_X},
                        y=train_y,
                        batch_size=32,
                        shuffle=True,
                        num_epochs=step_size)
        model.train(input_fn=train_fn)
        
        train_eval_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': train_X},
                        y=train_y,
                        batch_size=32,
                        shuffle=False,
                        num_epochs=1)
        valid_eval_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': valid_X},
                        y=valid_y,
                        batch_size=32,
                        shuffle=False,
                        num_epochs=1)
        
        train_metrics = model.evaluate(input_fn=train_eval_fn)
        valid_metrics = model.evaluate(input_fn=valid_eval_fn)
        
        metrics = {'train_'+metric: train_metrics[metric] 
                   for metric in train_metrics.keys()}
        metrics.update({'valid_'+metric: valid_metrics[metric]
                        for metric in train_metrics.keys()})
        metrics.update(n_epochs=ep_id)
        performance = performance.append(metrics, ignore_index=True)

    
    performance.to_csv(path_or_buf='./performance/'+model_name+'.csv', 
                       index=False)
        
    return(model, performance)


def random_search(n_samples, hyperparams):
    '''
    samples uniformly from hyperparameter hyperrectangle
    
    hyperparams:
    a dict of 'parameter': (min value, max value)
    '''
    samples_dict = {param: map(lambda val:
                                   np.random.randint(
                                   low=hyperparams[param][0],
                                   high=hyperparams[param][1] + 1,
                                   size=n_samples) 
                                   if is_int(val) else
                                   np.random.uniform(
                                   low=hyperparams[param][0],
                                   high=hyperparams[param][1],
                                   size=n_samples),
                               [hyperparams[param][0]])[0]
                    for param in hyperparams.keys()}
    samples_list = [{param: samples_dict[param][i] for param
                    in hyperparams.keys()} for i
                    in xrange(n_samples)]
    
    return(samples_list)
    

def random_search_iter(n_samples, hyperparams):
    '''
    samples uniformly from hyperparam rectangle
    for list of integer-valued parameters
    
    intended use: layer-wise parameter search
    
    hyperparams:
    dict = {'parameter': [(min1, max1), 
                          (min2, max2), 
                          (min_n, max_n)]}
    '''
    # get param: [[samp_1j, samp2j, ...samp_nj], ... [..., samp_nk]]
    # dict for n samples and k intervals
    samples_dict = {param: map(lambda (min_x, max_x):
                                   np.random.randint(
                                   low=min_x,
                                   high=max_x + 1,
                                   size=n_samples),
                               hyperparams[param])
                    for param in hyperparams.keys()}
    # select samples from list of list of samples
    samples_list = [{param: map(lambda samples:
                                    samples[i],
                                samples_dict[param])
                     for param in hyperparams.keys()}
                    for i in xrange(n_samples)]
    
    return(samples_list)

'''
plotting functions
'''
def plot_bands(b1_series, b2_series):
    '''
    plot N band one and 2 images side by side
    in a Kx4 grid

    K = N / 2
    '''
    n = len(b1_series)
    k = int(np.ceil(n / 2.0))
    h = int(np.ceil(float(k) * 30 / 4))
    b1_arrays = bands_to_imgs(b1_series)
    b2_arrays = bands_to_imgs(b2_series)

    fig = plt.figure(1, figsize=[30, h])
    for i in range(n):
        ax_b1 = fig.add_subplot(k, 4, i*2 + 1)
        ax_b2 = fig.add_subplot(k, 4, i*2 + 2)
        ax_b1.imshow(b1_arrays[i, :, :])
        ax_b2.imshow(b2_arrays[i, :, :])

    plt.show()


def plot_band(band):
    '''
    plots a single instance of a band assumed to 
    be a list
    '''
    img = band
    img = np.array(img)
    img = img.reshape([75, 75])
    plt.figure(figsize=[4, 4])
    plt.imshow(img)


def plot_imgs(imgs):
    '''
    plot N images of type numpy array
    '''
    n = imgs.shape[0]
    k = int(np.ceil(n / 4.0))
    h = int(np.ceil(float(k) * 30 / 4))

    fig = plt.figure(figsize=[30, h])
    for i in range(n):
        ax = fig.add_subplot(k, 4, i+1)
        ax.imshow(imgs[i, :, :])

    plt.show()


def plot_img(img):
    '''
    plots a single instance of a band assumed to
    be an numpy array
    '''
    plt.figure(figsize=[4, 4])
    plt.imshow(img)


'''
model performance
'''
def get_perf_summaries(n_hyperparams, n_folds):
    '''
    assumes working dir contains 'hyperparam<h_id>'
    directories with 'cv_<k>performance.csv'
    
    grabs perf summaries and hyperparameters
    '''
    performances = []
    hparam_settings = []
    non_hparams = ['train_loss',
                   'valid_loss',
                   'train_true_positives',
                   'train_false_positives',
                   'train_false_negatives',
                   'train_precision',
                   'train_recall',
                   'train_AUC',
                   'train_mean_per_class_acc',
                   'train_accuracy',
                   'valid_true_positives',
                   'valid_false_positives',
                   'valid_false_negatives',
                   'valid_precision',
                   'valid_recall',
                   'valid_AUC',
                   'valid_mean_per_class_acc',
                   'valid_accuracy',
                   'n_batches',
                   'n_epochs',
                   'n_folds',
                   'train_global_step',
                   'valid_global_step']
    for h_id in xrange(n_hyperparams):
        perf_dir = './hyperparam'+str(h_id)+'/cv_'+str(n_folds-1)+ \
                   'performance.csv'
        performance = pd.read_csv(perf_dir)
        performance['h_param_id'] = h_id
        performances.append(performance)
        
        hparam = performance.drop(non_hparams, axis=1)
        hparam_settings.append(hparam)
        
    df = pd.concat(performances, axis=0, ignore_index=True)
    hp = pd.concat(hparam_settings, axis=0, ignore_index=True)
    hp.drop_duplicates(inplace=True)
    
    return(df, hp)


def plot_learning_curves(n_hyperparams, fold, max_loss_lim=5):
    '''
    assumes working dir contains 'hyperparam<h_id>'
    directories with 'cv_<fold>performance.csv'
    
    plots learning curves for each hyperparameter setting
    given a fold and a list of hyperparameter names
    '''
    height = int(np.ceil(n_hyperparams / 3.0))
    fig = plt.figure(1, figsize=[30, 10 * height])
    for h_id in xrange(n_hyperparams):
        perf_dir = './hyperparam'+str(h_id)+'/cv_'+str(fold - 1)+ \
                   'performance.csv'
        performance = pd.read_csv(perf_dir)
        
        ax = fig.add_subplot(height, 3, h_id + 1)
        ax.set_title('Hyperparameter setting: '+ str(h_id))
        ax.set_ylabel('Log loss')
        df = performance[performance.n_folds == fold]
        ax.plot(df.train_global_step, df.train_loss)
        ax.plot(df.train_global_step, df.valid_loss)
        ax.set_ylim(0, max_loss_lim)
        ax.legend(loc='upper right')
        
    plt.show()


def plot_hyperparam_perf(data_df, param):
    '''
    plots hyperparameter of choice against validation metrics
    '''
    metrics = [ 'valid_true_positives',
                'valid_false_positives',
                'valid_false_negatives',
                'valid_precision',
                'valid_recall',
                'valid_AUC',
                'valid_mean_per_class_acc',
                'valid_accuracy']
    
    fig = plt.figure(1, figsize=[30, 40])
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(4, 2, i + 1)
        ax.set_ylabel(metric)
        # grab last batch id
        batch_id =  data_df.n_batches.unique()[-2]
        ax.plot(sorted(data_df[param].unique()), 
                data_df[data_df.n_batches >= batch_id].groupby(
                    by='h_param_id')[metric].mean())
        
    plt.show()   


'''
prepare submission
'''
def create_submission(test_X, predictions, name):
    '''
    saves submission csv
    
    expects test_X as a dataframe, and predictions
    as a generator returned by Estimator.predict()
    '''
    predictions = list(predictions)
    predictions = np.array([p for p 
                    in predictions]).reshape([test_X.shape[0], -1])
    predictions = predictions[:, 1:]
    submission = test_X.drop(['band_1', 'band_2', 'inc_angle'],
                             axis=1)
    submission['is_iceberg'] = predictions
    submission.to_csv(name+'_submission.csv', index=False)
    
    return(submission)

