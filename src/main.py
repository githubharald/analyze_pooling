import argparse
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from path import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
side_len = 224


def render_object(obj: str,
                  x: int,
                  y: int,
                  r: Optional[int] = None,
                  w: Optional[int] = None,
                  h: Optional[int] = None) -> np.ndarray:
    img = np.zeros([side_len, side_len], dtype=np.uint8)
    if obj == 'circle':
        assert r is not None
        cv2.circle(img, (x, y), r, 255, -1)
    elif obj == 'rectangle':
        assert w is not None and h is not None
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), 255, -1)
    else:
        raise Exception(f'{obj} not implemented')

    return img


def img_list_to_batch(imgs: List[np.ndarray]) -> torch.Tensor:
    imgs = np.array(imgs)
    imgs = imgs[..., None]
    imgs = torch.tensor(imgs).float() - 255 / 2
    imgs = imgs.permute([0, 3, 1, 2])
    return imgs


def label_list_to_batch(labels: List[int]) -> torch.Tensor:
    return torch.as_tensor(labels).float()


def get_samples(batch_size: int = 250) -> Tuple[List[np.ndarray], List[int]]:
    imgs = []
    labels = []

    for _ in range(batch_size):
        x = np.random.randint(0, side_len)
        y = np.random.randint(0, side_len)
        if np.random.random() < 0.5:
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            img = render_object('rectangle', x, y, w=w, h=h)
            labels.append(0)
        else:
            r = np.random.randint(5, 15)
            img = render_object('circle', x, y, r=r)
            labels.append(1)

        imgs.append(img)

    return imgs, labels


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_pooling_layers: int = 2) -> None:
        super(NeuralNetwork, self).__init__()

        num_layers = 3
        assert num_pooling_layers < num_layers

        channels = [1] + ([3] * (num_layers - 1)) + [1]
        op_list = []
        for i in range(num_layers):
            op_list.append(torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1))

            if i < (num_layers - 1):
                op_list.append(torch.nn.ReLU())

            if i < num_pooling_layers:
                op_list.append(torch.nn.MaxPool2d(2, stride=2))

        self.net = torch.nn.Sequential(*op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.net(x), dim=(1, 2, 3))


def train(net: NeuralNetwork) -> None:
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    num_batches = 1000
    for i in range(num_batches):
        # get batch
        img, label = get_samples()

        # forward pass
        optimizer.zero_grad()
        y = net(img_list_to_batch(img).to(device))
        loss = F.binary_cross_entropy_with_logits(y, label_list_to_batch(label).to(device))
        print(f'{i + 1}/{num_batches}: {loss.data}')

        # backward pass, optimize loss
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), 'weights')


def infer(net: NeuralNetwork) -> None:
    assert Path('weights').exists()
    net.load_state_dict(torch.load('weights', map_location=device))
    net.eval()

    imgs, _ = get_samples()
    with torch.no_grad():
        preds = torch.sigmoid(net(img_list_to_batch(imgs).to(device))).to('cpu')

    for img, pred in zip(imgs, preds):
        plt.imshow(img)
        msg = 'circle' if pred > 0.5 else 'rectangle'
        msg += f' {pred:.2f}'
        plt.title(msg)
        plt.waitforbuttonpress()


def analyze(net: NeuralNetwork) -> None:
    net.load_state_dict(torch.load('weights', map_location=device))
    net.eval()

    score = np.zeros([side_len, side_len])
    for x in range(0, side_len):
        print(f'{x + 1}/{side_len}')
        for y in range(0, side_len):
            img = render_object('rectangle', x, y, w=6, h=6)
            with torch.no_grad():
                preds = torch.sigmoid(net(img_list_to_batch([img]).to(device))).to('cpu')
            score[y, x] = float(preds[0].to('cpu'))

    plt.imshow(score)
    plt.colorbar()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'infer', 'analyze'], required=True)
    parser.add_argument('--num_pooling_layers', choices=[0, 1, 2], required=True, type=int)
    return parser.parse_args()


def main():
    parsed = parse_args()
    net = NeuralNetwork(parsed.num_pooling_layers).to(device)

    if parsed.mode == 'train':
        train(net)
    elif parsed.mode == 'infer':
        infer(net)
    elif parsed.mode == 'analyze':
        analyze(net)


if __name__ == '__main__':
    main()
