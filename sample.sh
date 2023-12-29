cd ./nn/data/
python3 tinypix2pix_mod.py --wgan --patchgan_lr 1e-5 --patchgan_wt 5 --unet_lr 1e-4 --unet_mae 2 --data_label ./