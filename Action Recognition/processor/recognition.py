#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """
    
    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
      self.model.eval()
      loader = self.data_loader['test']
      loss_value = []
      result_frag = []
      label_frag = []

      # 실제 레이블과 예측 결과를 저장할 리스트
      all_labels = []
      all_preds = []

      for data, label in loader:
          # get data
          data = data.float().to(self.dev)
          label = label.long().to(self.dev)

          # inference
          with torch.no_grad():
              output = self.model(data)
          result_frag.append(output.data.cpu().numpy())

          # get loss
          if evaluation:
              loss = self.loss(output, label)
              loss_value.append(loss.item())
              label_frag.append(label.data.cpu().numpy())

          # 예측 결과와 실제 레이블을 리스트에 저장
          _, predicted = torch.max(output, 1)  # 예측된 클래스를 가져오기
          all_labels.append(label.cpu().numpy())
          all_preds.append(predicted.cpu().numpy())

      self.result = np.concatenate(result_frag)
      if evaluation:
          self.label = np.concatenate(label_frag)
          self.epoch_info['mean_loss'] = np.mean(loss_value)
          self.show_epoch_info()

          # show top-k accuracy
          for k in self.arg.show_topk:
              self.show_topk(k)

          # confusion matrix 계산
          all_labels = np.concatenate(all_labels)
          all_preds = np.concatenate(all_preds)

          cm = confusion_matrix(all_labels, all_preds)  # confusion matrix 계산
          self.plot_confusion_matrix(cm)
          class_names = [f"Class {i}" for i in range(cm.shape[0])]

          # 클래스별 정확도 시각화 및 저장
          plot_class_accuracy(cm, class_names, save_path='classwise_accuracy.jpg')
    
    
    # confusion matrix를 시각화하는 함수
    def plot_confusion_matrix(self, cm, class_names=None, filename="confusion_matrix.jpg"):
        if class_names is None:
            class_names = ['Sitting', 'Walking', 'FallDown']  # 클래스 이름을 기본값으로 설정

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(filename, format='jpg', dpi=300)
        plt.close()  # 현재 플롯을 닫아서 메모리 해제

    def plot_class_accuracy(cm, class_names, save_path='classwise_accuracy.jpg'):
        """
        클래스별 정확도를 막대그래프로 시각화하여 저장하는 함수

        :param cm: confusion matrix (class x class)
        :param class_names: 클래스 이름 리스트
        :param save_path: 결과를 저장할 이미지 파일 경로
        """
        # 클래스별 정확도 계산
        class_accuracies = np.diagonal(cm) / np.sum(cm, axis=1)

        # 막대그래프 생성
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, class_accuracies, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy')
        
        # 이미지 파일로 저장
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
