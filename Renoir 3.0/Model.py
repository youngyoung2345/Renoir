from Backbone import VGG19
from CosFace import CosFace
import torch.nn as nn

class Recognizer(nn.Module):
    def __init__(self):
        super(Recognizer, self).__init__()
        self.VGG19 = VGG19()
        self.CosFace = CosFace(in_dim = 25088, out_dim = 20, s = 64, m = 0.6)
    
    def forward(self, x):
        x = self.VGG19(x)
        x = self.CosFace(x)
        
        return x
