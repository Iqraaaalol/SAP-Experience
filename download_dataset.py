from __future__ import annotations

import argparse
import shutil
import bz2
import sys
from pathlib import Path

try:
    import requests
except Exception:
    print("Missing dependency 'requests'. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        if total is None:
            with open(dest, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            total = int(total)
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded * 100 / total
                    print(f"\r{pct:5.1f}% ({downloaded}/{total})", end="")
            print()





def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download Wikivoyage and pull Ollama model")
    p.add_argument("--wiki-url", help="Wikivoyage dump URL", default="https://dumps.wikimedia.org/enwikivoyage/20260201/enwikivoyage-20260201-pages-articles.xml.bz2")
    p.add_argument("--out", help="Output path for the downloaded file", default="knowledge_base/wikivoyage.xml.bz2")
    p.add_argument("--skip-extract", help="Skip extracting downloaded .bz2 file", action="store_true")
    p.add_argument("--skip-wiki", help="Skip downloading the wikivoyage file", action="store_true")
    args = p.parse_args(argv)

    if not args.skip_wiki:
        try:
            download(args.wiki_url, Path(args.out))
        except Exception as e:
            print(f"Error downloading file: {e}", file=sys.stderr)
            return 1

    # extract xml from bz2
    if not args.skip_wiki and not args.skip_extract:
        out_path = Path(args.out)
        if out_path.name.endswith('.bz2'):
            decompressed_name = out_path.name[:-4]
            decompressed_path = out_path.with_name(decompressed_name)
            try:
                print(f"Extracting {out_path} -> {decompressed_path}")
                decompressed_path.parent.mkdir(parents=True, exist_ok=True)
                with bz2.open(out_path, 'rb') as src, open(decompressed_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                print(f"Extracted to {decompressed_path}")
            except Exception as e:
                print(f"Failed to extract {out_path}: {e}", file=sys.stderr)
                return 4

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
