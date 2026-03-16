from __future__ import annotations

import json
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


DATASET_URL = "https://www.openslr.org/resources/1/waves_yesno.tar.gz"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "external" / "yesno"
    extract_root = dataset_root / "waves_yesno"
    archive_path = dataset_root / "waves_yesno.tar.gz"
    manifest_path = project_root / "data" / "manifests" / "yesno_manifest.json"

    dataset_root.mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading {DATASET_URL} ...")
        urlretrieve(DATASET_URL, archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_root)

    manifest: list[dict[str, object]] = []
    wav_files = sorted(extract_root.rglob("*.wav"))
    for wav_path in wav_files:
        transcript = " ".join("yes" if token == "1" else "no" for token in wav_path.stem.split("_"))
        Path(f"{wav_path}.trn").write_text(transcript + "\n", encoding="utf-8")
        manifest.append(
            {
                "sample_id": wav_path.stem,
                "audio_path": str(wav_path.resolve()),
                "transcript": transcript,
                "duration_sec": 0,
                "split": "test",
                "scene_tag": "yesno",
                "noise_tag": "clean",
                "accent_tag": "english",
            }
        )

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Downloaded {len(wav_files)} files to: {extract_root}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
