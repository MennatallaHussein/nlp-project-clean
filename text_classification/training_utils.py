from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate



metric = evaluate.load('accuracy')



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = np.array(logits)
    preds = np.argmax(logits, axis=-1)
    labels = np.array(labels)
    acc = float((preds == labels).astype(np.float32).mean())
    return {"accuracy": acc}




def get_class_weights(df):
    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}

