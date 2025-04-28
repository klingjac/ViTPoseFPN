import torch
ckpt = torch.load('vitpose-b.pth', map_location='cpu') #TODO: Edit the path 
sd   = ckpt.get('state_dict', ckpt.get('model', ckpt))

backbone_sd = {k: v for k, v in sd.items() if k.startswith('backbone.')}
torch.save(backbone_sd, 'backbone_only_with_prefix.pth') #TODO: Edit the path 
