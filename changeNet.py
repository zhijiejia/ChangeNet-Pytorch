from torch import nn
import resnet
import torch

class Deconvolution(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super(Deconvolution, self).__init__()
        self.FC = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_classes),
        )

        self.Deconv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, 16, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, num_classes, 3, stride=1),
            nn.UpsamplingBilinear2d(size=input_size)
        )

        self.initial(self.FC)
        self.initial(self.Deconv)

    def initial(self, *models):
        for m in models:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        x1 = self.FC(x1)
        x1 = self.Deconv(x1)
        x2 = self.FC(x2)
        x2 = self.Deconv(x2)
        return torch.cat([x1, x2], dim=1)


class ChangeNet(nn.Module):
    def __init__(self, input_size=(224, 224), num_class=12):
        super(ChangeNet, self).__init__()
        self.backbone = resnet.resnet50(pretrained=True)
        self.cp3_Deconv = Deconvolution(512, num_class, input_size)
        self.cp4_Deconv = Deconvolution(1024, num_class, input_size)
        self.cp5_Deconv = Deconvolution(2048, num_class, input_size)
        self.FC = nn.Conv2d(2 * num_class, num_class, kernel_size=1)
        self.initial(self.FC)

    def initial(self, *models):
        for m in models:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, refer, test):
        layer1_r, layer2_r, layer3_r, layer4_r = self.backbone(refer)
        layer1_t, layer2_t, layer3_t, layer4_t = self.backbone(test)

        output2 = self.FC(self.cp3_Deconv(layer2_r, layer2_t))
        output3 = self.FC(self.cp4_Deconv(layer3_r, layer3_t))
        output4 = self.FC(self.cp5_Deconv(layer4_r, layer4_t))

        output = output2.add(output3)
        output = output.add(output4)

        return output
        #return torch.softmax(output, dim=1)
