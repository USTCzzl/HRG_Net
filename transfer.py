import torch
state_dict = torch.load("/home/zzl/Pictures/swin_ggcnn/output/models/220530_1030_d_hr/epoch_65_iou_0.95")
torch.save(state_dict, "epoch65_0.95.pth",_use_new_zipfile_serialization=False)
