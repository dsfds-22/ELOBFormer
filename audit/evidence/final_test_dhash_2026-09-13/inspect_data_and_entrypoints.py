from pathlib import Path
from collections import Counter
import csv
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
REPO = ROOT / 'release/repository'
SOURCE = ROOT / 'release/audit_work/dataset_archive'
sys.path.insert(0, str(REPO))
from tools.check_split_visual_leakage import dhash64
from tools.convert_labelme_boundary_to_masks import draw_lines, extract_boundary_polylines, make_heatmap

def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(p, data, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)

def run(args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return {'command': subprocess.list2cmdline([sys.executable, *map(str, args)]), 'exit_code': r.returncode, 'stdout': r.stdout[-2500:], 'stderr': r.stderr[-2500:]}

def main():
    evidence = {}
    missing_rel = 'development_labelme/train_val/041_000339'
    m = {r['path']: r for r in read_csv(SOURCE / 'sha256_manifest.csv')}
    with zipfile.ZipFile(ROOT / 'release/packages/emergency_outer_boundary_dataset_v1.zip') as z:
        names = {n.lstrip('./'): n for n in z.namelist()}
        evidence['missing_pair_recoverability'] = {}
        for ext in ['.jpg', '.json']:
            rel = missing_rel + ext
            evidence['missing_pair_recoverability'][ext] = {
                'audit_work_exists': (SOURCE / rel).exists(),
                'audit_work_sha_matches': hashlib.sha256((SOURCE / rel).read_bytes()).hexdigest() == m[rel]['sha256'],
                'zip_sha_matches': hashlib.sha256(z.read(names[rel])).hexdigest() == m[rel]['sha256'],
            }
    records = {}
    failures = []
    image_sizes = Counter()
    label_mismatch = []
    oob = []
    for split in ['train', 'val', 'test', 'final_test']:
        manifest = REPO / ('dataset_manifests/final_test/splits/test.csv' if split == 'final_test' else f'dataset_manifests/development/splits/{split}.csv')
        records[split] = []
        for row in read_csv(manifest):
            jp = SOURCE / row['json'] if split != 'final_test' else SOURCE / f"final_test_labelme/{row['prefix']}/{row['sample_id']}.json"
            ip = jp.with_suffix('.jpg')
            try:
                actual_sha = hashlib.sha256(ip.read_bytes()).hexdigest()
                if row.get('image_sha256') and actual_sha != row['image_sha256']:
                    failures.append({'id': row['sample_id'], 'issue': 'image hash'})
                j = json.loads(jp.read_text(encoding='utf-8-sig'))
                polys = extract_boundary_polylines(j, ['right_boundary'])
                if bool(polys) != bool(int(row['has_boundary'])):
                    label_mismatch.append(row['sample_id'])
                for points in polys:
                    for x,y in points:
                        if not (0 <= x < j['imageWidth'] and 0 <= y < j['imageHeight']):
                            oob.append(row['sample_id'])
                            break
                with Image.open(ip) as im:
                    image_sizes[str(im.size)] += 1
                    im.load()
                records[split].append({'id': row['sample_id'], 'path': str(ip), 'hash': dhash64(ip), 'sha256': actual_sha})
            except Exception as e:
                failures.append({'id': row['sample_id'], 'issue': str(e)})
        print(f'Decoded and fingerprinted {split}: {len(records[split])}', flush=True)
    evidence['source_validation'] = {'decoded_sizes': dict(image_sizes), 'failures': failures, 'label_presence_mismatch': label_mismatch, 'boundary_out_of_bounds_samples': sorted(set(oob))}
    candidates = []
    summaries = []
    for a,b in itertools.combinations(records, 2):
        minimum = 64
        near = 0
        for x in records[a]:
            for y in records[b]:
                d = (x['hash'] ^ y['hash']).bit_count()
                minimum = min(minimum, d)
                if d <= 2:
                    near += 1
                    candidates.append({'left_split': a, 'right_split': b, 'left_id': x['id'], 'right_id': y['id'], 'distance': d})
        exact = {x['sha256'] for x in records[a]} & {x['sha256'] for x in records[b]}
        summaries.append({'left':a, 'right':b, 'minimum_dhash_distance': minimum, 'candidates_at_most_2': near, 'exact_hash_overlap': len(exact)})
    evidence['dhash_screen'] = summaries
    write_csv(OUT / 'dhash_candidates.csv', candidates, ['left_split','right_split','left_id','right_id','distance'])

    # Minimal real-data fixture, in audit output only.
    fixture = OUT / 'entrypoint_fixture'
    raw = fixture / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    row = read_csv(REPO / 'dataset_manifests/final_test/splits/test.csv')[0]
    jp = SOURCE / f"final_test_labelme/{row['prefix']}/{row['sample_id']}.json"
    for p in [jp, jp.with_suffix('.jpg')]:
        shutil.copy2(p, raw / p.name)
    for split in ['train', 'val', 'test']:
        write_csv(fixture / f'protocol/{split}.csv', [row] if split == 'test' else [], list(row))
    evidence['empty_split_conversion'] = run([REPO / 'tools/convert_labelme_boundary_to_masks.py', '--root', raw, '--out', fixture / 'converted', '--split-dir', fixture / 'protocol', '--fixed-band-width', '12', '--edge-width', '5', '--heatmap-radius', '9', '--heatmap-sigma', '3', '--copy-images', '--copy-json'])
    evidence['readme_gt_dir_argument'] = run([REPO / 'tools/evaluate_raw_boundary_metrics.py', '--gt-dir', 'unused', '--pred-dir', 'unused', '--out-dir', fixture / 'unused'])
    generated = dict(row)
    generated.pop('height', None)
    generated.pop('width', None)
    generated.update(image=f'images/{jp.stem}.jpg', json=f'labelme_json/{jp.name}', band_mask=f'masks_band/{jp.stem}.png', edge_mask=f'masks_edge/{jp.stem}.png', heatmap=f'heatmaps/{jp.stem}.png')
    write_csv(fixture / 'converted/splits/test.csv', [generated], list(generated))
    evidence['missing_prediction_fallback'] = run([REPO / 'tools/evaluate_raw_boundary_metrics.py', '--dataset-root', fixture / 'converted', '--pred-dir', fixture / 'empty_predictions', '--out-dir', fixture / 'metrics'])
    (OUT / 'data_entrypoint_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
