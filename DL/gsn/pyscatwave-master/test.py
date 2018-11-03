import torch
from scatwave.scattering import Scattering

scat = Scattering(M=32, N=32, J=2).cuda()
x = torch.randn(1, 3, 32, 32).cuda()

print (scat(x).size())



from torch.utils.data import DataLoader
import torch
# import torch.utils.data