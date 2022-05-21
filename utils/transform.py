from PIL import Image
import torch
import torchvision

class VideoTransform():
    def __init__(self, resize, ccrop_size, rcrop_size, mean, std):
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                # DataAugumentation()
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
#                 GroupCenterCrop(ccrop_size),  # 画像をまとめてセンタークロップ
                GroupRandomCrop(ccrop_size, rcrop_size),  # 画像をまとめてランダムクロップ
                GroupGrayScale(),  # 画像をまとめてグレースケールに
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ]),
            'val': torchvision.transforms.Compose([
                GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                GroupCenterCrop(ccrop_size),  # 画像をまとめてセンタークロップ
                GroupGrayScale(),  # 画像をまとめてグレースケールに
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ])
        }

    def __call__(self, img_group, phase):
        return self.data_transform[phase](img_group)


class GroupResize():
    ''' 画像をまとめてリスケールするクラス。
    画像の短い方の辺の長さがresizeに変換される。
    アスペクト比は保たれる。
    '''

    def __init__(self, resize, interpolation=Image.BILINEAR):
        '''リスケールする処理を用意'''
        self.rescaler = torchvision.transforms.Resize(resize, interpolation)

    def __call__(self, img_group):
        '''リスケールをimg_group(リスト)内の各imgに実施'''
        return [self.rescaler(img) for img in img_group]


class GroupCenterCrop():
    ''' 画像をまとめてセンタークロップするクラス。
        （ccrop_size, ccrop_size）の画像を切り出す。
    '''

    def __init__(self, ccrop_size):
        '''センタークロップする処理を用意'''
        self.ccrop = torchvision.transforms.CenterCrop(ccrop_size)

    def __call__(self, img_group):
        '''センタークロップをimg_group(リスト)内の各imgに実施'''
        return [self.ccrop(img) for img in img_group]
    
class GroupRandomCrop():
    ''' 画像をまとめてランダムクロップするクラス。
        （rcrop_size[0], rcrop_size[1]）の画像を切り出した後に
        （ccrop_size, ccrop_size）の画像を切り出す。
    '''

    def __init__(self, ccrop_size, rcrop_size):
        '''ランダムクロップする処理を用意'''
        self.ccrop = torchvision.transforms.CenterCrop(rcrop_size)
        self.ccrop_size = ccrop_size

    def __call__(self, img_group):
        '''ランダムクロップをimg_group(リスト)内の各imgに実施'''  
        ccrop_img_group = [self.ccrop(img) for img in img_group]
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(ccrop_img_group[0], output_size=(self.ccrop_size, self.ccrop_size))
        
        return [torchvision.transforms.functional.crop(img, i, j, h, w) for img in ccrop_img_group]


class GroupGrayScale():
    ''' 画像をまとめてグレースケールにするクラス。
    '''

    def __init__(self):
        '''グレースケールにする処理を用意'''
        self.gscale = torchvision.transforms.Grayscale()

    def __call__(self, img_group):
        '''グレースケールをimg_group(リスト)内の各imgに実施'''
        return [self.gscale(img) for img in img_group]


class GroupToTensor():
    ''' 画像をまとめてテンソル化するクラス。
    '''

    def __init__(self):
        '''テンソル化する処理を用意'''
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        '''テンソル化をimg_group(リスト)内の各imgに実施
        '''

        return [self.to_tensor(img) for img in img_group]


class GroupImgNormalize():
    ''' 画像をまとめて標準化するクラス。
    '''

    def __init__(self, mean, std):
        '''標準化する処理を用意'''
        self.normlize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        '''標準化をimg_group(リスト)内の各imgに実施'''
        return [self.normlize(img) for img in img_group]


class Stack():
    ''' 画像を一つのテンソルにまとめるクラス。
    '''

    def __call__(self, img_group):
        '''img_groupはtorch.Size([1, 96, 96])を要素とするリスト
        '''
#         ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
#                          for x in img_group], dim=0)  # frames次元で結合
        ret = torch.cat([x.unsqueeze(dim=0) for x in img_group], dim=0)

        return ret