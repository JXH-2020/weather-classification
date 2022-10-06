import argparse
from datetime import datetime
import logging
import os
import traceback
import time

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
import utils
from datasets import BatchDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def modelSet():
    resnet50 = models.resnet50(pretrained=True)

    # 在PyTorch中加载模型时，所有参数的‘requires_grad’字段默认设置为true。这意味着对参数值的每一次更改都将被存储，以便在用于训练的反向传播图中使用。
    # 这增加了内存需求。由于预训练的模型中的大多数参数已经训练好了，因此将requires_grad字段重置为false。
    for param in resnet50.parameters():
        param.requires_grad = False

    # 为了适应自己的数据集，将ResNet-50的最后一层替换为，将原来最后一个全连接层的输入喂给一个有256个输出单元的线性层，接着再连接ReLU层和Dropout层，然后是256 x 6的线性层，输出为6通道的softmax层。
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.Linear(256, 14),
    )
    return resnet50


def main(args):
    # weather_backbone = model.modelNet.MyAlexNet().to(device)
    model_ = modelSet()
    weather_backbone = model_.to(device)
    weather_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        [{
            'params': weather_backbone.parameters()
        }],
        lr=args.base_lr,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience,
                                                           verbose=True)
    transform1 = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.6, p=0.3),
        transforms.RandomRotation(degrees=(0, 60)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([args.img_size, args.img_size]),
    ])
    train_dataset = BatchDataset(args.train_dir, transform=transform1)
    val_dataset = BatchDataset(args.val_dir, transform=transform2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))
    meter = utils.ListMeter()
    best_val = None
    for epoch in range(args.epochs):
        # 训练
        loss, acc = train(train_loader, weather_backbone, weather_criterion, optimizer, epoch, args)
        if np.isnan(loss):
            print("ERROR! Loss is Nan. Break.")
            break
        meter.add({"loss": loss, "acc": acc})
        # 验证
        val_loss, val_acc = validate(val_loader, weather_backbone, weather_criterion, epoch, args)
        meter.add({"val_loss": val_loss, "val_acc": val_acc})
        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch + 1, args.epochs) +
            "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) +
            "loss:{:.6f} val_loss:{:.6f} ".format(loss, val_loss) +
            "acc:{:.6f} val_acc:{:.6f}".format(acc, val_acc)
        )
        utils.plot_history(meter.get("loss"), meter.get("acc"), meter.get("val_loss"), meter.get("val_acc"),
                           history_save_path)

        # 保存
        if best_val is None or val_acc > best_val:
            best_val = val_acc
            save_checkpoint({
                'epoch': epoch,
                'weather_backbone': weather_backbone.state_dict(),
            }, model_save_path)
            logging.info("Saved best model.")
        else:
            save_checkpoint({'epoch': epoch, 'weather_backbone': weather_backbone.state_dict()
                             }, (model_save_path + '-' + str(val_loss) + '-' + str(1)))
            logging.info("Saved model.")
        scheduler.step(val_loss)
    utils.plot_history(meter.pop("loss"), meter.pop("acc"), meter.pop("val_loss"), meter.pop("val_acc"),
                       history_save_path)
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))


def train(train_loader, weather_backbone, weather_criterion, optimizer, epoch, args):
    meter = utils.AverageMeter()
    total = len(train_loader)
    le = 0
    correct = 0
    for i, (image, label, filename) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        weather_backbone = weather_backbone.to(device)

        outputs = weather_backbone(image)
        loss = weather_criterion(outputs, label)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum().item()
        le += label.size(0)
        acc = correct / le

        meter.add({"weather_loss": loss, "loss": loss.item(), "acc": acc})

        if i % args.log_step == 0:
            logging.info(
                "Trainning epoch:{}/{} batch:{}/{} ".format(epoch + 1, args.epochs, i + 1, total) +
                "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) +
                "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return meter.pop("loss"), meter.pop("acc")


def validate(val_loader, weather_backbone, weather_criterion, epoch, args):
    weather_backbone.eval()
    meter = utils.AverageMeter()
    with torch.no_grad():
        total = len(val_loader)
        le = 0
        correct = 0
        for i, (image, label, filename) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = weather_backbone(image)
            loss = weather_criterion(outputs, label)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
            le += label.size(0)
            acc = correct / le

            meter.add({"loss": loss.item(), "acc": acc})

            if i % args.log_step == 0:
                logging.info(
                    "Validating epoch:{}/{} batch:{}/{} ".format(epoch + 1, args.epochs, i + 1, total) +
                    "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
                )

    return meter.pop("loss"), meter.pop("acc")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Swin-Transformer FG simple predict:")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="training epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers parameter of dataloader")
    parser.add_argument("--log_step", type=int, default=50, help="log accuracy each log_step batchs")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    # parser.add_argument("--lr", type=float, default=0.001, help="backbone initial learning rate")
    # parser.add_argument("--warmup_steps", type=int, default=10, help="use warmup cosine schedule")
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, required=True, help="experiment name")
    parser.add_argument("--train_dir", type=str, required=True, help="train .txt file path")
    parser.add_argument("--val_dir", type=str, required=True, help="val .txt file path")
    args = parser.parse_args()

    model_save_path = "./middle/models/{}-best.pth".format(args.experiment_name)
    log_path = "./middle/logs/{}-{}.log".format(args.experiment_name, datetime.now()).replace(":", ".")
    history_save_path = "./middle/history/{}-{}.png".format(args.experiment_name, datetime.now()).replace(":", ".")

    os.makedirs("./middle/models/", exist_ok=True)
    os.makedirs("./middle/logs/", exist_ok=True)
    os.makedirs("./middle/history/", exist_ok=True)
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
    )
    try:
        main(args)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)
