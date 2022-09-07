import os
#from keras import backend as K
import keras
from datetime import datetime
import six
import csv
from collections import OrderedDict, Iterable
import numpy as np


class MyLogger(keras.callbacks.Callback):
    def __init__(self, n):
        self.n = n   # print loss & acc every n epochs
    def on_train_begin(self, logs = None):
        print("Starting training...")     
    def on_epoch_begin(self, epochs, logs=None):
        print(f"Starting epoch {epochs}")
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: Starting batch {batch}")
        
    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}\n loss is {logs['loss']}, accuracy is {logs['accuracy']}, F1-Score is {logs['f1_score']}, Precision is {logs['precision']}, Recall is {logs['recall']}")
        
    def on_epoch_end(self, epochs, logs=None):
        print(f"Finished epoch {epochs}\n loss is {logs['loss']}, accuracy is {logs['accuracy']}, F1-Score is {logs['f1_score']}, Precision is {logs['precision']}, Recall is {logs['recall']}")
        
    def on_train_end(self, logs=None):
        print("Finished training")
    
class NBatchCSVLogger(keras.callbacks.Callback):
    """Callback that streams every batch results to a csv file.
    """
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(NBatchCSVLogger, self).__init__()
        
    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            
    def on_epoch_end(self, epochs, logs=None):
        logs = logs or {}
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        if self.keys is None:
            self.keys = sorted(logs.keys())
        if self.model.stop_training:
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        row_dict = OrderedDict({'epoch': epochs})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        


        
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        
class NEpochCSVLogger(keras.callbacks.Callback):
    """Callback that streams every batch results to a csv file.
    """
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(NEpochCSVLogger, self).__init__()
        
    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        if self.keys is None:
            self.keys = sorted(logs.keys())
        if self.model.stop_training:
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        
class TestingCallback(keras.callbacks.Callback):
    
    def on_test_begin(self, logs=None):
        print("Starting testing ...")
        
    def on_test_batch_begin(self, batch, logs=None):
        print(f"Testing: Starting batch {batch}")
    
    def on_test_batch_end(self, batch, logs=None):
        print(f"Testing: Finished batch {batch}\n loss is {logs['loss']}, accuracy is {logs['accuracy']}, F1-Score is {logs['f1_score']}, Precision is {logs['precision']}, Recall is {logs['recall']}")
        
    def on_test_end(self, logs=None):
        print("Finished testing")

class NTestBatchCSVLogger(keras.callbacks.Callback):
    """Callback that streams every batch results to a csv file.
    """
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(NTestBatchCSVLogger, self).__init__()
    def on_test_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
      
    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        if self.keys is None:
            self.keys = sorted(logs.keys())
        if self.model.stop_training:
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
