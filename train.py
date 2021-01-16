import torch
from metric import *
from utils import decode_segmap
from tqdm import tqdm
from losses import FocalLoss
from transform import *
from scheduler import Poly
from dataset import SemDataset
from changeNet import ChangeNet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

lr = 1e-2
max_epoch = 50
num_class = 2
writer = SummaryWriter(comment='changenet')
model = ChangeNet(num_class=num_class, input_size=(224, 224))

param_group = [
    {'params': model.backbone.parameters(), 'lr': lr / 10, 'weight_decay': 1e-4, 'momentum': 0.9},
    {'params': model.cp3_Deconv.parameters(), 'lr': lr, 'weight_decay': 1e-4, 'momentum': 0.9},
    {'params': model.cp4_Deconv.parameters(), 'lr': lr, 'weight_decay': 1e-4, 'momentum': 0.9},
    {'params': model.cp5_Deconv.parameters(), 'lr': lr, 'weight_decay': 1e-4, 'momentum': 0.9},
    {'params': model.FC.parameters(), 'lr': lr, 'weight_decay': 1e-4, 'momentum': 0.9},
]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
model = torch.nn.parallel.DataParallel(model.cuda(), output_device=0)

transformTrain = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomGaussianBlur(),
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
])

transformTest = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
])

trainDataSet = SemDataset(split='train', transform=transformTrain)
testDataSet = SemDataset(split='test', transform=transformTest)
trainLoader = DataLoader(trainDataSet, batch_size=30, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
testLoader = DataLoader(testDataSet, batch_size=2, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)

weight = [0.25, 0.75]
# weight = [0.15, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.15]
crition = FocalLoss(gamma=2.0, alpha=weight)
scheduler = Poly(optimizer, num_epochs=max_epoch, iters_per_epoch=len(trainLoader))


def evalute(epoch):
    cnt = 0
    test_loss = 0
    total_inter, total_union = 0, 0
    for refer, test, label in tqdm(testLoader):
        refer = refer.float().cuda()
        test = test.float().cuda()
        label = label.cuda()

        label[label == 11] = 0
        label[label > 0] = 1

        output = model(refer, test)
        loss = crition(output, label)
        test_loss += loss.item()
        cnt += 1
        result = eval_metrics(output=output, target=label, num_classes=2)
        total_inter += result[0]
        total_union += result[1]

    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()
    print({
        'Epoch': epoch,
        'test_loss': {test_loss / cnt},
        "Mean_IoU": np.round(mIoU, 3),
    })

##############train step##############
for epoch in range(max_epoch):
    train_loss = 0
    refer, test, label, output = None, None, None, None
    for refer, test, label in tqdm(trainLoader):
        refer = refer.float().cuda()
        test = test.float().cuda()
        label = label.cuda()

        label[label == 11] = 0
        label[label > 0] = 1

        output = model(refer, test)
        loss = crition(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        scheduler.step(epoch=epoch)

    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(trainLoader)}')

    model.eval()
    evalute(epoch)
    model.train()
