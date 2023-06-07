import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNClassifier(nn.Module):
    # constructor
    def __init__(self, n_classes):
        super(DNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc = nn.Linear(256*75*75, n_classes, bias=True)

        self.init_conv()

        print('INFO: Load classifier model.')


    def init_conv(self):
        # Initialize convolution parameters.
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

        # Initialize fc parameters.
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: prediction
        """
        out = F.relu(self.bn1(self.conv1(image)))   # (N, 64, 300, 300)
        out = self.pool1(out)                       # (N, 64, 150, 150)

        out = F.relu(self.bn2(self.conv2(out)))     # (N, 128, 150, 150)
        out = self.pool2(out)                       # (N, 128, 75, 75)

        out = F.relu(self.bn3(self.conv3(out)))     # (N, 256, 75, 75)
    
        out = out.view(out.size(0), -1)             # (N, 256 * 75 * 75)
        out = self.fc(out)     

        return out