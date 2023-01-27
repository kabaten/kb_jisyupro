import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import cv2

import sys, os
sys.path.append('..')

# 以下のひらがなデータセットを利用する
# https://github.com/ndl-lab/hiragana_mojigazo

# 以下のサイトを参考にした。
# https://tkstock.site/2022/05/29/python-pytorch-mydataset-dataloader/

class HiraganaDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        image_paths = df["path"].to_list()
        # self.input_size = input_size
        self.len = len(image_paths)
        # self.transform = transform
        # self.phase = phase
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):  
        image_path = self.df["path"].to_list()[index]
        # 画像の読込
        #image = cv2.imread(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 2値化
        image = np.array(image)
        image = ~image
        _, image = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)
        image = image.astype(np.float32)

        img_size, _ = image.shape
        ax = np.array(range(1, img_size+1))
        X, Y = np.meshgrid(ax, ax)
        X = X*image
        Y = Y*image
        data = np.stack([X, Y, image]).astype(np.float32)
        
        # label はここでは自分自身
        label = image
        return data, label

rom2kana = {
            'a'  :'あ', 'i'  :'い', 'u'  :'う', 'e'  :'え', 'o'  :'お',
            'ka' :'か', 'ki' :'き', 'ku' :'く', 'ke' :'け', 'ko' :'こ',
            'sa' :'さ', 'shi':'し', 'su' :'す', 'se' :'せ', 'so' :'そ',
            'ta' :'た', 'chi':'ち', 'tu' :'つ', 'te' :'て', 'to' :'と',
            'na' :'な', 'ni' :'に', 'nu' :'ぬ', 'ne' :'ね', 'no' :'の',
            'ha' :'は', 'hi' :'ひ', 'fu' :'ふ', 'he' :'へ', 'ho' :'ほ',
            'ma' :'ま', 'mi' :'み', 'mu' :'む', 'me' :'め', 'mo' :'も',
            'ya' :'や', 'yu' :'ゆ', 'yo' :'よ',
            'ra' :'ら', 'ri' :'り', 'ru' :'る', 're' :'れ', 'ro' :'ろ',
            'wa' :'わ', 'wi' :'ゐ', 'we' :'ゑ', 'wo' :'を', 'n'  :'ん',
            'ga' :'が', 'gi' :'ぎ', 'gu' :'ぐ', 'ge' :'げ', 'go' :'ご',
            'za' :'ざ', 'ji' :'じ', 'zu' :'ず', 'ze' :'ぜ', 'zo' :'ぞ',
            'da' :'だ', 'di' :'ぢ', 'du' :'づ', 'de' :'で', 'do' :'ど',
            'ba' :'ば', 'bi' :'び', 'bu' :'ぶ', 'be' :'べ', 'bo' :'ぼ',
            'pa' :'ぱ', 'pi' :'ぴ', 'pu' :'ぷ', 'pe' :'ぺ', 'po' :'ぽ',
            'si' :'し', 'ti': 'ち', 'tsu': 'つ', 'hu': 'ふ', 'zi': 'じ'
            }

def kana2uni(kana):
    uni = 'U' + format(ord(kana), '04x').upper()
    return uni

def rom2uni(rom):
    kana = rom2kana[rom]
    uni = kana2uni(kana)
    return uni

def make_hiragana_dataset(DATADIR, CATEGORIES=None, ratio=0.7, num_sample=1000):
    """
    DATADIR: データのあるディレクトリへのパス. (i.e. "../data/hiragana73")
    CATEGORIES: ひらがな(ローマ字表記)のリスト(i.e. ['a', 'i', 'u',...]). None の場合は73種類すべて.
    ratio: データを分割する際の訓練データの割合
    num_sample: 各フォルダから取り出す画像の数
    """
    data = []

    for CATEGORY in os.listdir(DATADIR):
        if CATEGORY != ".DS_Store":
            for img_path in os.listdir(DATADIR + "/" + CATEGORY)[:num_sample]:
                data.append([(DATADIR + "/" + CATEGORY + "/" + img_path), CATEGORY])

    df = pd.DataFrame(data, columns=['path', 'category'])
    if CATEGORIES:
        CATEGORIES = [rom2uni(rom) for rom in CATEGORIES]
        df = df[df['category'].isin(CATEGORIES)]
    #breakpoint()
    hiragana_dataset = HiraganaDataset(df)

    train_dataset, valid_dataset = torch.utils.data.random_split(hiragana_dataset, [int(len(hiragana_dataset)*ratio), int(len(hiragana_dataset)*(1-ratio))])

    return train_dataset, valid_dataset

# その他の画像を利用する場合
def make_input(imgs):
    """
    画像から入力データを作る関数(np[N,48,48] -> np[N,3,48,48])
    以下のような利用を想定している。

    >> path = 'image.jpg'
    >> image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    >> src = (~image)/255
    >> inputs = torch.from_numpy(make_input(src)).type(torch.float32)
    >> labels = inputs[:, -1]
    """
    if len(imgs.shape) > 2:
        N, img_size, _ = imgs.shape
    else:
        N = 1
        img_size, _ = imgs.shape
    
    imgs = imgs.reshape(N, 1, img_size, img_size)
    ax = np.array(range(0, img_size))
    X, Y = np.meshgrid(ax, ax)
    Xs = np.tile(X, (N, 1)).reshape(N, 1, img_size, img_size)
    Ys = np.tile(Y, (N, 1)).reshape(N, 1, img_size, img_size)
    Xs = Xs*imgs
    Ys = Ys*imgs

    data = np.concatenate([Xs, Ys, imgs], 1)
    return data