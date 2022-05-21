import os
from PIL import Image
import numpy as np
import random
import torch

class VideoDataset(torch.utils.data.Dataset):
    """
    動画のDataset
    """

    def __init__(self, video_list, videoid_labelid_dict, num_segments, phase, transform, img_tmpl='{:05d}.jpg', random_frame=False):
        self.video_list = video_list  # 動画画像のフォルダへのパスリスト
        self.videoid_labelid_dict = videoid_labelid_dict
        self.num_segments = num_segments  # 動画を何分割して使用するのかを決める
        self.phase = phase  # train or val
        self.transform = transform  # 前処理
        self.img_tmpl = img_tmpl  # 読み込みたい画像のファイル名のテンプレート
        self.random_frame = random_frame  # 取得する動画のインデックスをランダムにするか

    def __len__(self):
        '''動画の数を返す'''
        return len(self.video_list)

    def __getitem__(self, index):
        '''
        前処理をした画像たちのデータとラベル、ラベルIDを取得
        '''
        imgs_transformed_list, label_id = self.pull_item(index)
        
        return imgs_transformed_list, label_id

    def pull_item(self, index):
        '''前処理をした画像たちのデータとラベル、ラベルIDを取得'''

        # 1. 画像たちをリストに読み込む
        dir_path = self.video_list[index]  # 画像が格納されたフォルダ

        img_group_list = []
        
        number_of_frames_to_add = random.randint(-5, 5)

        for side in ("left", "right"):
            directory_path = os.path.join(dir_path, side)
            indices = self._get_indices(directory_path, number_of_frames_to_add)  # 読み込む画像idxを求める
            img_group = self._load_imgs(directory_path, self.img_tmpl, indices)  # リストに読み込む
            img_group_list.append(img_group)

        # 2. ラベルの取得し、idに変換する
        # csv rowにindexを指定してlabelをゲット

        video_id = int(dir_path.split('/')[-1])
        label_id = self.videoid_labelid_dict[video_id]

        # 3. 前処理を実施
        imgs_transformed_list = [self.transform(img_group, phase=self.phase) for img_group in img_group_list]

        return imgs_transformed_list, label_id

    def _load_imgs(self, dir_path, img_tmpl, indices):
        '''画像をまとめて読み込み、リスト化する関数'''
        img_group = []  # 画像を格納するリスト

        for idx in indices:
            # 画像のパスを取得
            file_path = os.path.join(dir_path, img_tmpl.format(idx))

            # 画像を読み込む
            img = Image.open(file_path).convert('RGB')

            # リストに追加
            img_group.append(img)

        return img_group
    
    def _get_indices(self, dir_path, number_of_frames_to_add):
        """
        動画全体をself.num_segmentに分割した際に取得する動画のidxのリストを取得する
        """
        # 動画のフレーム数を求める
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)

        # 動画の取得間隔幅を求める
        tick = (num_frames) / float(self.num_segments)

        # 動画の取得間隔幅で取り出す際のidxをリストで求める
        indices = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])+1
        
        if self.random_frame:
            indices = indices + number_of_frames_to_add
            indices = np.clip(indices, 1, num_frames)

        return indices