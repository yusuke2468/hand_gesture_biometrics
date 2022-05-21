from .layers import ECO_2D, ECO_3D
import torch
from torch import nn
# from torch.nn import init

class ECO(nn.Module):
    def __init__(self):
        super().__init__()

        # 2D Netモジュール
        self.eco_2d = ECO_2D()

        # 3D Netモジュール
        self.eco_3d = ECO_3D()

        # クラス分類の全結合層
        self.fc_final = nn.Linear(in_features=512, out_features=128, bias=True)

    def forward(self, x):
        '''
        入力xはtorch.Size([batch_num, num_segments=32, 1, 96, 96]))
        '''

        bs, ns, c, h, w = x.shape

        # xを(bs*ns, c, h, w)にサイズ変換する
        out = x.view(-1, c, h, w)

        # 2D Netモジュール 出力torch.Size([batch_num×32, 96, 12, 12])
        out = self.eco_2d(out)

        # 2次元画像をテンソルを3次元用に変換する
        out = out.view(-1, ns, 96, 12, 12)

        # 3D Netモジュール 出力torch.Size([batch_num, 512])
        out = self.eco_3d(out)

        # クラス分類の全結合層　出力torch.Size([batch_num, 128])
        out = self.fc_final(out)

        return out



class HBM(nn.Module):
    def __init__(self):
        super().__init__()

        self.eco_left = ECO()

        self.eco_right = ECO()

    def forward(self, x):
        # input: x_left
        out_left = self.eco_left(x[0])

        # input: x_right
        out_right = self.eco_right(x[1])

        # out_leftとout_rightを結合
        out = torch.cat((out_left, out_right), 1)

        return out