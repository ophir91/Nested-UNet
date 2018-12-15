from keras import backend as K

# Define metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def precision(y_true, y_pred):  # PPV - Positive Predictive Value
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + K.epsilon()) / (K.sum(y_pred_f) + K.epsilon())


def tversky_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)  # True positives
    fp = K.sum(y_pred_f) - tp  # False positives
    fn = K.sum(y_true_f) - tp  # False negatives

    return (tp + K.epsilon()) / (tp + 0.9 * fp + (1 - 0.9) * fn + K.epsilon())


# Define custom loss
def dice_coef_loss():
    def calculate_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    return calculate_loss


def tversky_index_loss():
    def calculate_loss(y_true, y_pred):
        return -tversky_index(y_true, y_pred)

    return calculate_loss