import torch


def generate_bigrams(x):
    for i in range(0, len(x)-2):
        x.append(' '.join(x[i], x[i+1]))
    return x


def multi_cat_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])
