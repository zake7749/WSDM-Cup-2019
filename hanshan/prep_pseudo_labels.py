"""Script to prepare new pseudo-labels for training."""
import os
import pandas as pd


if __name__ == '__main__':
    # file paths
    labels_path = \
        '../zake7749/data/ensemble/second_level/FirstLevelPseudoLabels.csv'
    data_folder = '../zake7749/data/dataset'

    # open data
    print('Opening files...')
    train = pd.read_csv(os.path.join(data_folder, 'train.csv'))
    test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
    labels = pd.read_csv(labels_path)

    print('Processing...')

    # concat pseudo label columns to test data to get the dev set
    dev_pl = pd.concat([test, labels], axis=1)

    # create a new data frame from both train and pseudo-labeled test
    train_pl = pd.concat([train, dev_pl], axis=0)

    print('Saving...')

    # create folder if doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), 'pl_data')):
        os.mkdir(os.path.join(os.getcwd(), 'pl_data'))

    # save the augmented training data
    train_pl.to_csv(os.path.join('pl_data', 'train.csv'), index=False)
    print('The new training set has %s records, looking like this:'
          % len(train_pl))
    print(train_pl.head())  # look at it and make sure it's right

    # save the pseudo-labeled test data as the dev data
    dev_pl.to_csv(os.path.join('pl_data', 'dev.csv'), index=False)
    print('The new dev set has %s records, looking like this:' % len(dev_pl))
    print(dev_pl.head())
