import torch, os, cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from collections import deque


# 轻量级车道平滑器 - 折中核心：简单、稳定、有效
class SimpleLaneSmoother:
    def __init__(self, maxlen=3):
        self.history = deque(maxlen=maxlen)

    def smooth(self, points):
        self.history.append(points.copy())
        return np.mean(np.array(self.history), axis=0)


# 全局平滑对象
smoother = SimpleLaneSmoother(maxlen=2)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    dist_print('start testing...')

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {k[7:]: v for k, v in state_dict.items()} if 'module.' in list(state_dict.keys())[
        0] else state_dict
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.45, 0.406), (0.229, 0.224, 0.225)),
    ])

    if cfg.dataset == 'CULane':
        splits = ['my_test2.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'list/test_split/' + split),
                                    img_transform=img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform=img_transforms) for
                    split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = split[:-4] + '.avi'
        print(f"Output: {out_path}")
        vout = cv2.VideoWriter(out_path, fourcc, 30.0, (img_w, img_h))

        for data in tqdm.tqdm(loader):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))

            # 平滑处理 + 过滤噪声
            out_j = smoother.smooth(out_j)

            # 绘制：干净、不抖动、无杂线
            for lane_idx in range(out_j.shape[1]):
                if np.sum(out_j[:, lane_idx] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        x = int(out_j[k, lane_idx] * col_sample_w * img_w / 800) - 1
                        y = int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1

                        # 边缘过滤：不贴边、不交叉
                        if 100 < x < img_w - 100:
                            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)

            vout.write(vis)
        vout.release()