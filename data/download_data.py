import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'f3', 'seam'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows = json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if args.dataset != 'all' and not row['file'].lower().startswith(args.dataset):
            continue
        target = args.output_dir / row['file']
        if target.exists():
            if target.stat().st_size == row['bytes'] and digest(target) == row['sha256']:
                print(f'Already verified: {target.name}')
                continue
            raise FileExistsError(f'Existing file has a different size/hash: {target}')
        temporary = target.with_suffix(target.suffix + '.part')
        print(f'Downloading {target.name} ({row["bytes"]:,} bytes)', flush=True)
        request = urllib.request.Request(row['url'], headers={'User-Agent': 'FAMT-Salt3D-data-downloader'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if temporary.stat().st_size != row['bytes'] or digest(temporary) != row['sha256']:
            raise RuntimeError(f'Download verification failed: {temporary}')
        temporary.replace(target)
        print(f'Verified: {target.name}', flush=True)


if __name__ == '__main__':
    main()
