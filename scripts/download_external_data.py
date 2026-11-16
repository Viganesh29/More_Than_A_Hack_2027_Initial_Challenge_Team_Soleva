"""scripts/download_external_data.py

Deterministic external dataset fetcher for More Than A Hack 2027 Phase 1.
Downloads Metrica Sports Sample Games 1, 2, and 3 from the official repository
with SHA-256 validation, exponential backoff retries, and atomic writes.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

# Pinned to a fixed commit of metrica-sports/sample-data so the external evaluation is reproducible.
SAMPLE_DATA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
BASE_URL = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{SAMPLE_DATA_COMMIT}/data"

DATASET_INVENTORY = [
    # Sample Game 1 (CSV tracking + CSV events)
    {
        "game": "Sample_Game_1",
        "relative_path": "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "description": "Game 1 Home Tracking Data (CSV)",
        "sha256": "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    },
    {
        "game": "Sample_Game_1",
        "relative_path": "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "description": "Game 1 Away Tracking Data (CSV)",
        "sha256": "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    },
    {
        "game": "Sample_Game_1",
        "relative_path": "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "description": "Game 1 Raw Events Data (CSV)",
        "sha256": "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    },
    # Sample Game 2 (CSV tracking + CSV events)
    {
        "game": "Sample_Game_2",
        "relative_path": "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "description": "Game 2 Home Tracking Data (CSV)",
        "sha256": "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    },
    {
        "game": "Sample_Game_2",
        "relative_path": "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "description": "Game 2 Away Tracking Data (CSV)",
        "sha256": "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    },
    {
        "game": "Sample_Game_2",
        "relative_path": "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "description": "Game 2 Raw Events Data (CSV)",
        "sha256": "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    },
    # Sample Game 3 (FIFA EPTS tracking + EPTS XML metadata + JSON events)
    {
        "game": "Sample_Game_3",
        "relative_path": "Sample_Game_3/Sample_Game_3_tracking.txt",
        "description": "Game 3 EPTS Tracking Data (TXT)",
        "sha256": "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
    },
    {
        "game": "Sample_Game_3",
        "relative_path": "Sample_Game_3/Sample_Game_3_metadata.xml",
        "description": "Game 3 EPTS Metadata (XML)",
        "sha256": "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    },
    {
        "game": "Sample_Game_3",
        "relative_path": "Sample_Game_3/Sample_Game_3_events.json",
        "description": "Game 3 Raw Events Data (JSON)",
        "sha256": "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    },
]


def calculate_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    target_path: Path,
    expected_sha256: str = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> bool:
    """Downloads a URL to target_path atomically using a temporary file.

    Validates SHA-256 if expected_sha256 is provided.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    headers = {"User-Agent": "MTAH-2027-External-Data-Fetcher/1.0"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (Attempt {attempt}/{max_retries})...")
            hasher = hashlib.sha256()
            start_time = time.time()
            total_bytes = 0

            with urllib.request.urlopen(req, timeout=timeout) as response, open(
                temp_path, "wb"
            ) as out_file:
                content_len = response.headers.get("Content-Length")
                total_expected = int(content_len) if content_len else None

                while chunk := response.read(65536):
                    out_file.write(chunk)
                    hasher.update(chunk)
                    total_bytes += len(chunk)
                    if total_expected:
                        pct = (total_bytes / total_expected) * 100
                        print(
                            f"\r  Transferred: {total_bytes / (1024*1024):.2f}MB / {total_expected / (1024*1024):.2f}MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )

            elapsed = max(time.time() - start_time, 0.001)
            speed_mb = (total_bytes / (1024 * 1024)) / elapsed
            print(
                f"\n  Finished: {total_bytes / (1024*1024):.2f}MB in {elapsed:.2f}s ({speed_mb:.2f} MB/s)"
            )

            actual_sha256 = hasher.hexdigest()
            if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
                print(
                    f"  [ERROR] SHA-256 mismatch!\n    Expected: {expected_sha256}\n    Actual:   {actual_sha256}"
                )
                if temp_path.exists():
                    temp_path.unlink()
                return False

            # Atomic replace
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            print(f"  [OK] Saved to {target_path} (SHA-256: {actual_sha256[:12]}...)")
            return True

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"\n  [WARN] Download error: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                backoff = 2**attempt
                print(f"  Retrying in {backoff} seconds...")
                time.sleep(backoff)
            else:
                print(f"  [ERROR] Failed to download {url} after {max_retries} attempts.")
                return False

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify Metrica Sports Sample Data Games 1-3."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ext_data"),
        help="Target base directory (default: ext_data)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify presence and checksums without downloading",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Download timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Max retry attempts (default: 3)"
    )
    parser.add_argument(
        "--compute-hashes",
        action="store_true",
        help="Calculate and print SHA-256 hashes of downloaded files",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    print(f"Metrica Sports External Data Tooling")
    print(f"Base Directory: {output_dir}")
    print(f"Mode: {'Verify Only' if args.verify_only else 'Download & Verify'}")

    # Check for checksums file if exists
    checksums_file = output_dir / "checksums.json"
    cached_checksums = {}
    if checksums_file.exists():
        try:
            with open(checksums_file, "r", encoding="utf-8") as f:
                cached_checksums = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse {checksums_file}: {e}")

    all_passed = True
    computed_hashes = {}

    for item in DATASET_INVENTORY:
        rel_path = item["relative_path"]
        target_path = output_dir / rel_path
        expected_hash = item["sha256"] or cached_checksums.get(rel_path)
        url = f"{BASE_URL}/{rel_path}"

        print(f"\n--- Checking: {item['description']} ---")
        print(f"Path: {target_path}")

        if args.verify_only:
            if not target_path.exists():
                print(f"  [FAIL] Missing file: {target_path}")
                all_passed = False
                continue
            actual_hash = calculate_sha256(target_path)
            computed_hashes[rel_path] = actual_hash
            if expected_hash and actual_hash.lower() != expected_hash.lower():
                print(
                    f"  [FAIL] SHA-256 mismatch for {rel_path}:\n    Expected: {expected_hash}\n    Actual:   {actual_hash}"
                )
                all_passed = False
            else:
                print(f"  [PASS] Verified OK (SHA-256: {actual_hash[:12]}...)")
            continue

        # Download workflow
        if target_path.exists() and not args.force:
            actual_hash = calculate_sha256(target_path)
            computed_hashes[rel_path] = actual_hash
            if expected_hash is None or actual_hash.lower() == expected_hash.lower():
                print(
                    f"  [PASS] Already exists and verified (SHA-256: {actual_hash[:12]}...)"
                )
                continue
            else:
                print(f"  [WARN] Existing file hash mismatch. Re-downloading...")

        success = download_file(
            url=url,
            target_path=target_path,
            expected_sha256=expected_hash,
            timeout=args.timeout,
            max_retries=args.retries,
        )
        if not success:
            all_passed = False
        else:
            computed_hashes[rel_path] = calculate_sha256(target_path)

    if args.compute_hashes:
        print("\n=== Computed SHA-256 Hashes ===")
        print(json.dumps(computed_hashes, indent=2))
        with open(checksums_file, "w", encoding="utf-8") as f:
            json.dump(computed_hashes, f, indent=2)
        print(f"Saved manifest to {checksums_file}")

    if not all_passed:
        print("\n[ERROR] External data verification or download failed.")
        sys.exit(1)

    print("\n[SUCCESS] All external datasets ready and verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
