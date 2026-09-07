import json, os, re
root = r'd:\Escuela\3_Primer Semestre\Redes Neuronales\DeCloud\models'
pat = re.compile(r'psnr|ssim|iou|dice|miou|mIoU|PSNR|SSIM|IoU|Dice', re.I)
for fn in sorted(os.listdir(root)):
    if not fn.endswith('.ipynb'):
        continue
    p = os.path.join(root, fn)
    try:
        nb = json.load(open(p, 'r', encoding='utf-8'))
    except Exception as e:
        print('ERR', fn, e)
        continue
    hits = []
    for ci, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        out = ''.join(''.join(o.get('text', [])) if isinstance(o, dict) else '' for o in cell.get('outputs', []))
        txt = src + '\n' + out
        if pat.search(txt):
            hits.append((ci, txt[:500]))
    if hits:
        print(f'\nFILE {fn} HITS {len(hits)}')
        for ci, txt in hits[:10]:
            print('CELL', ci)
            print(re.sub(r'\s+', ' ', txt)[:300])
            print('---')
