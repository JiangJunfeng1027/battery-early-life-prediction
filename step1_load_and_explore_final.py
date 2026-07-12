"""
Step 1: MIT Battery Dataset - Data Loading & Exploration (FINAL)
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

DATA_DIR = './data/MIT/'
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAT_FILES = {
    'batch1': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat',
}

def load_batch_from_mat(mat_filepath, batch_prefix):
    print(f"Loading {mat_filepath}...")
    f = h5py.File(mat_filepath, 'r')
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    print(f"  Found {num_cells} cells")

    bat_dict = {}
    for i in range(num_cells):
        print(f"  Processing cell {i+1}/{num_cells}...", end='\r')

        # Cycle life
        cl = f[batch['cycle_life'][i, 0]][()].flatten()

        # Charging policy
        try:
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()
        except Exception:
            policy = 'unknown'

        # Summary data - h5py 3.x returns values directly (no dereferencing needed)
        summary_grp = f[batch['summary'][i, 0]]
        summary = {}
        field_map = {'IR':'IR','QCharge':'QC','QDischarge':'QD','Tavg':'Tavg',
                     'Tmin':'Tmin','Tmax':'Tmax','chargetime':'chargetime','cycle':'cycle'}
        for mat_field, out_field in field_map.items():
            try:
                summary[out_field] = summary_grp[mat_field][0, :]
            except Exception as e:
                print(f"\n  Warning: {mat_field} cell {i}: {e}")
                summary[out_field] = np.array([])

        # Cycle data
        cycles_grp = f[batch['cycles'][i, 0]]
        cycle_dict = {}
        num_cycles = cycles_grp['I'].shape[0]
        for j in range(num_cycles):
            cd = {}
            for mat_f, out_f in {'I':'I','Qc':'Qc','Qd':'Qd','Qdlin':'Qdlin',
                                  'T':'T','Tdlin':'Tdlin','V':'V',
                                  'discharge_dQdV':'dQdV','t':'t'}.items():
                try:
                    cd[out_f] = cycles_grp[mat_f][j, 0]
                    # If it's a reference, dereference it
                    if isinstance(cd[out_f], h5py.h5r.Reference):
                        cd[out_f] = f[cd[out_f]][()].flatten()
                    elif isinstance(cd[out_f], (np.floating, float)):
                        # scalar - skip, might be a reference we need to handle
                        ref = cycles_grp[mat_f][j, 0]
                        cd[out_f] = np.array([ref])
                    else:
                        cd[out_f] = np.atleast_1d(cd[out_f])
                except Exception:
                    pass
            cycle_dict[str(j)] = cd

        bat_dict[f'{batch_prefix}{i}'] = {
            'cycle_life': cl, 'charge_policy': policy,
            'summary': summary, 'cycles': cycle_dict
        }

    f.close()
    print(f"\n  Done! Loaded {num_cells} cells.")
    return bat_dict

def merge_batches(all_batches):
    if 'batch1' not in all_batches:
        return {k: v for d in all_batches.values() for k, v in d.items()}
    batch1 = all_batches['batch1'].copy()
    # The public five-cell exclusion belongs to the 2017-05-12 batch, not this
    # separately loaded 2018-02-20 batch. Label/window validity is checked later.
    if 'batch2' in all_batches:
        batch2 = all_batches['batch2'].copy()
        b2k = ['b2c7','b2c8','b2c9','b2c15','b2c16']
        b1k = ['b1c0','b1c1','b1c2','b1c3','b1c4']
        add_len = [662, 981, 1060, 208, 482]
        for i, bk in enumerate(b1k):
            if bk in batch1 and b2k[i] in batch2:
                batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
                for j in batch1[bk]['summary']:
                    if j == 'cycle':
                        batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[b2k[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
                    else:
                        batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[b2k[i]]['summary'][j]))
                last = len(batch1[bk]['cycles'])
                for ji, jk in enumerate(batch2[b2k[i]]['cycles']):
                    batch1[bk]['cycles'][str(last + ji)] = batch2[b2k[i]]['cycles'][jk]
        for bk in b2k:
            batch2.pop(bk, None)
    else:
        batch2 = {}
    batch3 = all_batches.get('batch3', {}).copy()
    for bad in ['b3c37','b3c2','b3c23','b3c32','b3c42','b3c43']:
        batch3.pop(bad, None)
    bat_dict = {**batch1, **batch2, **batch3}
    print(f"Total cells after merging: {len(bat_dict)}")
    return bat_dict

def explore_and_plot(bat_dict):
    keys = list(bat_dict.keys())
    # Handle NaN cycle lives
    cycle_lives = []
    valid_keys = []
    for k in keys:
        cl_val = bat_dict[k]['cycle_life'].flatten()[0]
        if not np.isnan(cl_val):
            cycle_lives.append(int(cl_val))
            valid_keys.append(k)
    cycle_lives = np.array(cycle_lives)

    print(f"\n{'='*60}")
    print("DATA OVERVIEW")
    print(f"{'='*60}")
    print(f"Total cells: {len(cycle_lives)}")
    print(f"Cycle life range: {cycle_lives.min()} - {cycle_lives.max()}")
    print(f"Mean: {cycle_lives.mean():.0f}, Median: {np.median(cycle_lives):.0f}, Std: {cycle_lives.std():.0f}")

    ex = valid_keys[0]
    cell = bat_dict[ex]
    print(f"\nExample cell '{ex}':")
    print(f"  Cycle life: {int(cell['cycle_life'].flatten()[0])}")
    print(f"  Charge policy: {cell['charge_policy']}")
    print(f"  Cycles recorded: {len(cell['cycles'])}")
    print(f"  Summary fields: {list(cell['summary'].keys())}")
    print(f"  Summary QD length: {len(cell['summary']['QD'])}")
    print(f"  Cycle data fields: {list(cell['cycles']['0'].keys())}")

    # Plot 1: Capacity Degradation
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, cl in zip(valid_keys, cycle_lives):
        qd = bat_dict[key]['summary']['QD']
        if len(qd) == 0: continue
        cyc = bat_dict[key]['summary']['cycle'][:len(qd)]
        color = 'red' if cl < 500 else ('orange' if cl < 1000 else 'green')
        ax.plot(cyc, qd, color=color, alpha=0.3, linewidth=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],color='red',label='< 500 cycles'),
        Line2D([0],[0],color='orange',label='500-1000 cycles'),
        Line2D([0],[0],color='green',label='> 1000 cycles')
    ], fontsize=12)
    ax.set_xlabel('Cycle Number', fontsize=14)
    ax.set_ylabel('Discharge Capacity (Ah)', fontsize=14)
    ax.set_title('Capacity Degradation - All Cells', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_capacity_degradation.png'), dpi=150)
    plt.close()
    print("\nSaved fig1_capacity_degradation.png")

    # Plot 2: Cycle Life Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(cycle_lives, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(cycle_lives), color='red', linestyle='--', label=f'Median: {np.median(cycle_lives):.0f}')
    ax.set_xlabel('Cycle Life', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of Battery Cycle Lives', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_distribution.png'), dpi=150)
    plt.close()
    print("Saved fig2_distribution.png")

    # Plot 3: Example cells (short, mid, long life)
    sorted_idx = np.argsort(cycle_lives)
    for tag, si in [('short', sorted_idx[0]), ('medium', sorted_idx[len(sorted_idx)//2]), ('long', sorted_idx[-1])]:
        ck = valid_keys[si]
        cell = bat_dict[ck]
        cl = int(cell['cycle_life'].flatten()[0])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{tag.title()} Life: {ck} | Cycles: {cl} | Policy: {cell["charge_policy"]}', fontsize=14)

        ax = axes[0, 0]
        total = len(cell['cycles'])
        for idx in np.linspace(0, total-1, min(8, total), dtype=int):
            c = cell['cycles'][str(idx)]
            if 'Qd' in c and 'V' in c and len(np.atleast_1d(c['Qd'])) > 1:
                ax.plot(np.atleast_1d(c['Qd']), np.atleast_1d(c['V']), alpha=0.7, label=f'Cyc {idx}')
        ax.set_xlabel('Discharge Capacity (Ah)'); ax.set_ylabel('Voltage (V)')
        ax.set_title('Discharge Curves'); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        qd = cell['summary']['QD']
        if len(qd) > 0:
            ax.plot(cell['summary']['cycle'][:len(qd)], qd, 'b-', linewidth=1)
        ax.set_xlabel('Cycle'); ax.set_ylabel('Discharge Capacity (Ah)')
        ax.set_title('Capacity Degradation'); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ir = cell['summary']['IR']
        if len(ir) > 0:
            ax.plot(cell['summary']['cycle'][:len(ir)], ir, 'r-', linewidth=1)
        ax.set_xlabel('Cycle'); ax.set_ylabel('IR (Ohm)')
        ax.set_title('Internal Resistance'); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        tavg = cell['summary']['Tavg']
        if len(tavg) > 0:
            ax.plot(cell['summary']['cycle'][:len(tavg)], tavg, 'g-', linewidth=1, label='Avg')
        ax.set_xlabel('Cycle'); ax.set_ylabel('Temperature (C)')
        ax.set_title('Temperature'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'fig3_{tag}_life.png'), dpi=150)
        plt.close()
        print(f"Saved fig3_{tag}_life.png")

if __name__ == '__main__':
    print("="*60)
    print("MIT Battery Dataset - Load & Explore (FINAL)")
    print("="*60)

    pkl_ok = all(os.path.exists(os.path.join(OUTPUT_DIR, f'{b}.pkl')) for b in MAT_FILES)
    if pkl_ok:
        all_b = {}
        for bn in MAT_FILES:
            with open(os.path.join(OUTPUT_DIR, f'{bn}.pkl'), 'rb') as fp:
                all_b[bn] = pickle.load(fp)
            print(f"  Loaded {bn}: {len(all_b[bn])} cells")
    else:
        print("\nConverting .mat to .pkl (this takes a few minutes)...")
        all_b = {}
        for bn, mf in MAT_FILES.items():
            mp = os.path.join(DATA_DIR, mf)
            if not os.path.exists(mp):
                print(f"WARNING: {mp} not found"); continue
            prefix = {'batch1':'b1c','batch2':'b2c','batch3':'b3c'}[bn]
            all_b[bn] = load_batch_from_mat(mp, prefix)
            with open(os.path.join(OUTPUT_DIR, f'{bn}.pkl'), 'wb') as fp:
                pickle.dump(all_b[bn], fp)
            print(f"  Saved {bn}.pkl")

    bat_dict = merge_batches(all_b)
    with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'wb') as fp:
        pickle.dump(bat_dict, fp)
    print("Saved all_batteries.pkl")

    explore_and_plot(bat_dict)
    print(f"\n{'='*60}")
    print("DONE! Check the output/ folder for plots.")
    print("Next: use all_batteries.pkl for downstream feature engineering.")
    print("="*60)
