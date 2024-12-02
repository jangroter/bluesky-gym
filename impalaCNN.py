from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A Residual Block used in the IMPALA CNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.skip:
            identity = self.skip(x)
        x = self.conv1(self.relu(x))
        x = self.conv2(self.relu(x))
        return x + identity

class ResidualBlockLight(nn.Module):
    """
    A Residual Block used in the IMPALA CNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlockLight, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.skip:
            identity = self.skip(x)
        x = self.relu(self.conv1(x))
        return x + identity

class IMPALACNN(BaseFeaturesExtractor):
    """
    Custom IMPALA CNN architecture for Stable-Baselines3.
    """
    def __init__(self, observation_space, features_dim=256):
        super(IMPALACNN, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]
        
        # Define IMPALA CNN layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(16, 16)
        self.residual_block2 = ResidualBlock(16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block3 = ResidualBlock(32, 32)
        self.residual_block4 = ResidualBlock(32, 32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block5 = ResidualBlock(32, 32)
        self.residual_block6 = ResidualBlock(32, 32)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the number of features after convolutions
        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)

        # Final linear layer to produce the desired feature dimension
        self.fc = nn.Linear(n_flatten, features_dim)

    def _get_flattened_size(self, observation_space):
        sample_input = th.as_tensor(observation_space.sample()[None]).float()

        x = self.conv1(sample_input)
        x = self.max1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.relu(x)
        x = self.flatten(x)
        print(x.shape[1])
        return x.shape[1]

    def forward(self, observations):
        x = self.conv1(observations)
        x = self.max1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.relu(self.fc(x))

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        print(n_flatten)

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
class ImpalaLight(BaseFeaturesExtractor):
    """
    Custom IMPALA CNN architecture for Stable-Baselines3.
    """
    def __init__(self, observation_space, features_dim=256):
        super(ImpalaLight, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]
        
        # Define IMPALA CNN layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4, padding=0)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.residual_block1 = ResidualBlockLight(16, 16)
        self.residual_block2 = ResidualBlockLight(16, 16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the number of features after convolutions
        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)

        # Final linear layer to produce the desired feature dimension
        self.fc = nn.Linear(n_flatten, features_dim)

    def _get_flattened_size(self, observation_space):
        sample_input = th.as_tensor(observation_space.sample()[None]).float()
        x = self.conv1(sample_input)
        x = self.max1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.flatten(x)
        print(x.shape[1])
        return x.shape[1]

    def forward(self, observations):
        x = self.conv1(observations)
        x = self.max1(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.flatten(x)
        return self.relu(self.fc(x))
    
class ImpalaLight_v2(BaseFeaturesExtractor):
    """
    Custom IMPALA CNN architecture for Stable-Baselines3. Combines NatureCNN's channels, kernel and stride with half the Resnets used in Impala and Maxpooling
    """
    def __init__(self, observation_space, features_dim=256):
        super(ImpalaLight_v2, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]
        
        # Define IMPALA CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.residual_block1 = ResidualBlock(32, 32)
        # self.residual_block2 = ResidualBlock(32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.residual_block3 = ResidualBlock(64, 64)
        # self.residual_block4 = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.residual_block5 = ResidualBlock(64, 64)
        # self.residual_block6 = ResidualBlock(64, 64)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the number of features after convolutions
        with th.no_grad():
            n_flatten = self._get_flattened_size(observation_space)

        # Final linear layer to produce the desired feature dimension
        self.fc = nn.Linear(n_flatten, features_dim)

    def _get_flattened_size(self, observation_space):
        sample_input = th.as_tensor(observation_space.sample()[None]).float()

        x = self.conv1(sample_input)
        x = self.max1(x)
        x = self.residual_block1(x)
        # x = self.residual_block2(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.residual_block3(x)
        # x = self.residual_block4(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.residual_block5(x)
        # x = self.residual_block6(x)
        x = self.relu(x)
        x = self.flatten(x)
        print(x.shape[1])
        return x.shape[1]

    def forward(self, observations):
        x = self.conv1(observations)
        x = self.max1(x)
        x = self.residual_block1(x)
        # x = self.residual_block2(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.residual_block3(x)
        # x = self.residual_block4(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.residual_block5(x)
        # x = self.residual_block6(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.relu(self.fc(x))