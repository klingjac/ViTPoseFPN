#import torch
#import torchvision
#import torch.nn as nn
#from mmcv.cnn import normal_init, constant_init

#from ..builder import NECKS


#@NECKS.register_module()


#class FeaturePyramidNetwork(nn.Module):
    
#    def __init__(self, in_channels=256, num_scales = 4, strides = [4,8,16,32]):
#        super().__init__()
        
#        assert num_scales == len(strides)
        
#        self.in_channels = in_channels
#        self.num_scales = num_scales
        
        #Bottom-up Pathway
        #making separate layers in case we need to keep track of the weights and biases
#        self.bottom_convs = nn.ModuleList([
#            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=strides[i])
#            for i in range(num_scales)
#        ])
        
        # Top-down pathway
#        self.lateral_convs = nn.ModuleList([
#            nn.Conv2d(in_channels, in_channels, kernel_size=1)
#            for _ in range(num_scales)
#        ])
#        self.output_convs = nn.ModuleList([
#            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#            for _ in range(num_scales)
#        ])
        
        
#    def bottom_up_pathway(self, x):
        #building the ResNet feature
        #here with increasing spatial size
        
#        residual_blocks = []
#        input = x
#        for i in range(self.num_scales):
#            ci = self.bottom_convs[i](input)
#            residual_blocks.append(ci)
#            input = ci  
        
#        return residual_blocks
    
#    def top_down_pathway(self, residual_blocks):
        #go down and combine the residual blocks with the lateral connections
        
#        merged_maps = []
        
#        for i in range(self.num_scales-1, -1, -1):
#            if i< self.num_scales-1:
                #Upsample the previous result to match the next result in the resnet
#                upsampled = nn.functional.interpolate(
#                                merged_maps[-1], mode='nearest', size = residual_blocks[i].shape[-2:]
#                                )
#                mi = self.lateral_convs[i](residual_blocks[i]) + upsampled
#            else:
#                mi = self.lateral_convs[i](residual_blocks[i]) #the last conv layer 
#            merged_maps.insert(0, mi)
          
#        feature_maps = []
#        for map in merged_maps:
#            pi = self.output_convs[i](map)
#            feature_maps.append(pi)
        
#        return feature_maps
    
#    def init_weights(self):
        # Initialize convolutional layers
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                normal_init(m, std=0.001)
#            elif isinstance(m, nn.BatchNorm2d):
#                constant_init(m, 1)

#    def forward(self, x):
        
        #residuals = self.bottom_up_pathway(x)
        #feature_maps = self.top_down_pathway(residuals)
        
        #return feature_maps
	#def forward(self, x):
#    	residuals   = self.bottom_up_pathway(x)
#    	pyramid    = self.top_down_pathway(residuals)
    	# assume pyramid[0] is highest-res; upsample & sum
#    	out = pyramid[0]
#    	for p in pyramid[1:]:
#        	out = out + nn.functional.interpolate(p, size=out.shape[-2:], mode='nearest')
#    	return out





import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import normal_init, constant_init

from ..builder import NECKS


@NECKS.register_module()
class FeaturePyramidNetwork(BaseModule):
    """
    A custom Feature Pyramid Network (FPN) neck with bottom-up and top-down pathways.
    Inherits from BaseModule to integrate MMDetection's init_weights logic.
    """
    def __init__(
        self,
        in_channels=256,
        num_scales=4,
        strides=(4, 8, 16, 32),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001)
    ):
        super().__init__(init_cfg=init_cfg)
        assert num_scales == len(strides), \
            "num_scales must match length of strides"

        self.in_channels = in_channels
        self.num_scales = num_scales
        self.strides = strides

        # Bottom-up pathway: reduce feature map sizes
        self.bottom_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=strides[i]
            )
            for i in range(num_scales)
        ])

        # Top-down lateral and output convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(num_scales)
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(num_scales)
        ])

    def bottom_up_pathway(self, x):
        """
        Builds a sequence of feature maps by applying 1x1 convs with increasing strides.
        """
        residual_blocks = []
        out = x
        for conv in self.bottom_convs:
            out = conv(out)
            residual_blocks.append(out)
        return residual_blocks

    def top_down_pathway(self, residual_blocks):
        """
        Merges feature maps from bottom-up via lateral connections and upsampling.
        """
        merged_maps = []
        # Iterate from highest-resolution (last) to lowest (first)
        for i in range(self.num_scales - 1, -1, -1):
            if i < self.num_scales - 1:
                prev = merged_maps[-1]
                up = F.interpolate(prev,
                                   size=residual_blocks[i].shape[-2:],
                                   mode='nearest')
                merged = self.lateral_convs[i](residual_blocks[i]) + up
            else:
                merged = self.lateral_convs[i](residual_blocks[i])
            merged_maps.insert(0, merged)

        # Apply final 3x3 conv to each merged map
        feature_maps = []
        for idx, m in enumerate(merged_maps):
            feature_maps.append(self.output_convs[idx](m))
        return feature_maps

    def init_weights(self):
        """
        Initialize all Conv2d with normal init and BatchNorm2d with constant init.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x):
        """
        Forward pass: x is the highest-resolution feature map.
        Returns a list of feature maps at multiple scales.
        """
        residuals = self.bottom_up_pathway(x)
        return self.top_down_pathway(residuals)