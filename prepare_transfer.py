import sys
import os
import torch


def prepare_transfer(ckpt_path, out_path):
    if not os.path.isfile(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[*] Loading pre-trained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    changed = False


    if 'optimizer' in ckpt:
        del ckpt['optimizer']
        print("  -> Removed 'optimizer' state.")
        changed = True


    if 'epoch' in ckpt:
        del ckpt['epoch']
        print("  -> Removed 'epoch' (will restart from epoch 1).")
        changed = True


    if not isinstance(ckpt, dict):
        ckpt = {'net': ckpt}
        changed = True

    if changed:
        torch.save(ckpt, out_path)
        print(f"[*] Successfully saved clean transfer checkpoint to: {out_path}\n")
    else:

        import shutil
        shutil.copy(ckpt_path, out_path)
        print(f"[*] Checkpoint is already clean. Copied to: {out_path}\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python prepare_transfer.py <input_ckpt_path> <output_ckpt_path>")
        sys.exit(1)

    in_ckpt = sys.argv[1]
    out_ckpt = sys.argv[2]
    prepare_transfer(in_ckpt, out_ckpt)
