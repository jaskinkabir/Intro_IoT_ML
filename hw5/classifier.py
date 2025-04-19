import tf_keras
from tf_keras import models, layers

class Classifier:
    def __init__(self, model: tf_keras.Model, optimizer: tf_keras.optimizers.Optimizer, loss: tf_keras.losses.Loss, metrics: tf_keras.metrics.Metric):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            jit_compile=True,
        )
    def fit(
        self,
        save_path = None,
        best_path = 'training/bestacc.txt',
        reset_best_acc = False,
        stop_patience = 10,
        fit_kwargs = {},
        ):
        checkpoint_callback = lambda: None
        if reset_best_acc:
            best_accuracy = 0
        else:
            with open(best_path, 'r+') as f:    
                best_accuracy = float(f.read())
        if save_path is not None:
            checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
                save_path,
                save_best_only=False,
                # monitor='val_accuracy',
                # mode='max',
                # initial_value_threshold=best_accuracy
            )
        early_stop = tf_keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=stop_patience,
            mode='max'
        )

        
        hist = self.model.fit(**fit_kwargs, callbacks=[checkpoint_callback, early_stop])
        
        best_accuracy_new = max(hist.history['val_accuracy'])
        print(f"Best Validation Accuracy: {best_accuracy_new}")
        if best_accuracy_new > best_accuracy:
            with open(best_path, 'w+') as f:
                f.write(str(best_accuracy_new))
        return hist
        
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = models.load_model(path)