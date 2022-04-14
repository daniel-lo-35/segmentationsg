import torch

pretrain_model = torch.load('../output-step1/best_model_final.pth.pth') # <Path to Pretrained Model Weights (example: model_final.pth)>
pretrain_weight = {}
pretrain_weight['model'] = pretrain_model['model']
with open('../output-step1/model_weights.pth', 'wb') as f: # <Weight Save Path (example: model_weights.pth)>
    torch.save(pretrain_weight, f)
