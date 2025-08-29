from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate



metric = evaluate.load('accuracy')



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)



def get_class_weights(df):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(df['label'].unique().tolist())),
        y=df['label'].values   # directly use numpy array instead of .tolist()
    )
    return class_weights
