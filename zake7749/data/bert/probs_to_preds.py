import pandas as pd
import numpy as np


if __name__ == '__main__':
    test = pd.read_csv('../dataset/test.csv')
    ids = list(test['id'])
    probs = pd.read_csv('bert.csv')
    preds = []
    labels = ['agreed', 'disagreed', 'unrelated']
    for i, row in probs.iterrows():
        p = [row['agreed'], row['disagreed'], row['unrelated']]
        j = np.argmax(p)
        preds.append(labels[j])
    df = pd.DataFrame({'Id': ids, 'Category': preds})
    df.to_csv('preds.csv', index=False)

