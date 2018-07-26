
from Encoder.Encoder import Resnet_v2_101, Inception_v3
from Encoder.Attention import Classical_Attention
class Model_Zoo:
    def __init__(self):
        pass
    def __call__(self, model):
        if model == 'Resnet_v2_101':
            encoder = Resnet_v2_101()
            
        if model=='Inception_v3':
            encoder = Inception_v3()
        decoder = Classical_Attention()
        return encoder, decoder







