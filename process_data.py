import pandas as pd

train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')

if __name__ == '__main__':
    print(train, test)
    print()
