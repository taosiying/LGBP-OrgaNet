import ml_collections
import os
import wget


os.makedirs('./weights', exist_ok=True)

def get_LGBPOrga_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 2
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
            "./weights/swin_tiny_patch4_window7_224.pth")
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm = [256,512,1024]
    cfg.resnet_pretrained = True



    return cfg