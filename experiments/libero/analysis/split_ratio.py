"""Route / value norm ratios of read_split_probe.py records (layer 0 and 2; renormalized over image keys; signed).
Usage: python split_ratio.py <root>"""
import pickle, glob, os, sys, numpy as np
root = sys.argv[1]
for mdir in sorted(glob.glob(root + '/*/')):
    name = os.path.basename(mdir.rstrip('/')); acc = {}
    for f in sorted(glob.glob(mdir + 'task*.pkl')):
        for ep in pickle.load(open(f, 'rb'))['episodes']:
            for frame, pr in ep['probes'].items():
                if 'split_norms' in pr['features']:
                    acc.setdefault(frame, []).append(pr['features']['split_norms'][1:])
    out = []
    for frame in ('t0', 'transport'):
        if frame in acc:
            n = np.concatenate(acc[frame]).mean(0)
            cols = f'L0 {n[0,0]/n[0,1]:.2f}' + (f' renorm {n[0,3]/n[0,4]:.2f}' if n.shape[1] > 3 else '') + f' | L2 {n[2,0]/n[2,1]:.2f}' + (f' renorm {n[2,3]/n[2,4]:.2f}' if n.shape[1] > 3 else '') + (f' | signed L0 {n[0,5]/n[0,6]:.2f} L2 {n[2,5]/n[2,6]:.2f}' if n.shape[1] > 5 and n[0,6] > 0 else '')
            out.append(f'{frame}: {cols}')
    print(f'  {name:9s} route/value ratio  ' + '  ||  '.join(out))
