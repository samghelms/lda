import glob
import pandas as pd

def _read_reviews(glb):
    reviews = []
    for n in glb:
        id, rating = n.strip('.txt').split('_')
        text = open(n, 'r').read()
        reviews.append((text, rating))
    return pd.DataFrame(reviews, columns=['review', 'rating'])

def get_reviews_data():

    neg = glob.glob("aclImdb/train/neg/*")
    pos = glob.glob("aclImdb/train/neg/*")
    df_neg = _read_reviews(neg)
    df_pos = _read_reviews(pos)
    df = pd.concat([df_neg, df_pos], ignore_index=True)

    assert df.shape[0] == 25000 and df.shape[1] == 2
    return df


if __name__ == '__main__':
    print(get_reviews_data())