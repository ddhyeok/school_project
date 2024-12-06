import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out =out+ self.shortcut(x)
        out = F.relu(out)
        return out
class ResidualSEBlockFC(nn.Module):
    def __init__(self, input_channels, reduction_ratio=8, hidden_dims=[1024, 512], output_dim=256, dropout_prob=0.4):
        super(ResidualSEBlockFC, self).__init__()
        
        # SEBlock
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(input_channels, input_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // reduction_ratio, input_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),  
            nn.Conv2d(input_channels // reduction_ratio, input_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # fc
        self.flatten = nn.Flatten()  # Flattening 512x12x12 to 73728
        self.fc1 = nn.Linear(input_channels * 6 * 12, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
        # Dropout 
        self.dropout = nn.Dropout(dropout_prob)
        
        # residual
        self.residual_connection1 = nn.Linear(input_channels * 6 * 12, output_dim)
        self.residual_connection2 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        se_weight = self.se_block(x)
        x = x * se_weight

        # Flatten
        x = self.flatten(x)
        
        #Residual
        residual1 = self.residual_connection1(x)

        #fc
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        #Residual
        residual2 = self.residual_connection2(x)

        # fc
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x =x+ residual1 + residual2
        
        return x
class CAEClassifier(nn.Module):
    def __init__(self,model,dropout=0.4,encoder_grad=True):
        super(CAEClassifier, self).__init__()
        self.encoder = nn.Sequential(
        model.encoder.conv1,
        model.encoder.bn1,
        model.encoder.pool,
        model.encoder.layer1,
        model.encoder.layer2,
        model.encoder.layer3,
        model.encoder.layer4,
        model.encoder.layer5,
        model.encoder.layer6,
        model.encoder.layer7,
        model.encoder.layer8
        )
        
        for param in self.encoder.parameters():
            param.requires_grad = encoder_grad
        

        self.fc1 = ResidualSEBlockFC(512)
        
        self.fc2 =nn.Sequential(
            
                    nn.Linear(256, 128),      
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),      
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 2)
                )
        self.bn=nn.BatchNorm2d(512)
        self.he_init()
        
        
    def forward(self, x):
        x=self.encoder(x)
        x=self.bn(x)
        
        x=self.fc1(x)
        x=self.fc2(x)
        
        return x
    def he_init(self):
        for linear in self.fc2:
            if isinstance(linear, nn.Linear):
                nn.init.kaiming_normal_(linear.weight, nonlinearity='leaky_relu')
                linear.bias.data.fill_(0.01)
