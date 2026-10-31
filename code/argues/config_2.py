import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')


parser.add_argument('--n_threads', type=int, default=1,
                    help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--base_width', type=int, default=8,
                    help='base width of network')


parser.add_argument('--train_path',
                    default='/home/user/ldr_code/seg/fda_seg/seam_patches',
                    help='training npz folder (full labels 0/1 or sparse 0/1/-1)')
parser.add_argument('--val_path',
                    default='/home/user/ldr_code/seg/fda_seg/test',
                    help='validation npz folder — MUST be fully labeled (0/1 only)')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size of trainset')


parser.add_argument('--save', default='models',
                    help='save path of trained models')


parser.add_argument('--restore', default=None,
                    help='pretrained model checkpoint path (.pth). '
                         'None = train from scratch (default). '
                         'Example: --restore output/seam/checkpoints/epoch_020.pth')


parser.add_argument('--optim', default='AdamW',
                    help='optimizer: AdamW or SGD')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')


parser.add_argument('--da_type', default='common', choices=['common', 'pasa'],
                    help='Data augmentation type (common or pasa)')


parser.add_argument('--use_recon', type=int, default=1, choices=[0, 1],
                    help='1 = enable reconstruction auxiliary loss, 0 = disable')
parser.add_argument('--recon_weight', type=float, default=0.08,
                    help='overall weight for reconstruction auxiliary loss')
parser.add_argument('--recon_rampup', type=int, default=20,
                    help='epochs to ramp reconstruction loss from 0 to full weight')
parser.add_argument('--recon_horizontal_ratio', type=float, default=0.18,
                    help='ratio of H/W slices masked for anisotropic reconstruction')
parser.add_argument('--recon_vertical_ratio', type=float, default=0.10,
                    help='ratio of D slices masked for anisotropic reconstruction')
parser.add_argument('--recon_block_ratio', type=float, default=0.12,
                    help='size ratio of the random 3D block mask')
parser.add_argument('--recon_edge_weight', type=float, default=0.25,
                    help='relative weight for edge-aware gradient reconstruction')
parser.add_argument('--recon_freq_weight', type=float, default=0.10,
                    help='relative weight for frequency amplitude reconstruction')
parser.add_argument('--recon_low_freq_ratio', type=float, default=0.08,
                    help='radius ratio of low-frequency window used in FFT reconstruction loss')

args = parser.parse_args()
