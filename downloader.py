"""
Datasets to use : GSE38417, GSE6011, GSE1764, GSE1007, GSE1004, GSE465 - Only DMD and Control
Datasets to use : GSE3307, GSE13608 - with Other Disorders Dropped
"""

import GEOparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict

SERIES_LIST = ["GSE38417", "GSE6011", "GSE1764", "GSE1007", "GSE1004", "GSE465", "GSE3307", "GSE13608" ]
DEST_DIR    = "."
OUTPUT_FILE = "DMD_combined_dataset.csv"

# Dictionary to store metadata analysis across datasets
metadata_analysis = defaultdict(list)

def process_series(series_id: str):

    print(f"\nProcessing {series_id}…")
    gse = GEOparse.get_GEO(geo=series_id, destdir=DEST_DIR, annotate_gpl=True)

    # Build raw expression (probes × samples) → transpose → (samples × probes)
    raw = gse.pivot_samples("VALUE").transpose()
    print(f"  Raw shape (samples × probes): {raw.shape}")

    # Extract labels by scanning each GSM's metadata entries
    labels_dict = {}
    dataset_metadata = []
    
    for gsm_id, gsm in gse.gsms.items():
        label = None
        all_meta = []
        for k, vals in gsm.metadata.items():
            if isinstance(vals, list):
                all_meta.extend(vals)
            else:
                all_meta.append(str(vals))
        
        sample_metadata = []
        for entry in all_meta:
            entry_lower = entry.lower()
            sample_metadata.append(entry_lower)
            print(f"  {gsm_id}: {entry_lower}")
            
            if "dmd" in entry_lower:
                label = 1
                break
            if ("control" in entry_lower) or ("normal" in entry_lower):
                label = 0
                break
        
        # Store metadata for this sample
        dataset_metadata.append({
            'gsm_id': gsm_id,
            'metadata_entries': sample_metadata,
            'assigned_label': label
        })
        
        if label is not None:
            labels_dict[gsm_id] = label
    
    # Store metadata analysis for this dataset
    metadata_analysis[series_id] = dataset_metadata

    labels = pd.Series(labels_dict, name="label")
    print(f"  Labeled samples: {labels.shape[0]} / {len(gse.gsms)}")

    # Probe → gene mapping via the series's GPL
    gpl_id = list(gse.gpls.keys())[0]
    df_gpl = gse.gpls[gpl_id].table.set_index("ID")
    print(f"  GPL columns: {df_gpl.columns.tolist()}")
    
    # Normalize "Gene Symbol" column name
    if "GeneSymbol" in df_gpl.columns and "Gene Symbol" not in df_gpl.columns:
        df_gpl = df_gpl.rename(columns={"GeneSymbol": "Gene Symbol"})
    if "Gene Symbol" not in df_gpl.columns:
        print(f"  WARNING: {series_id} GPL {gpl_id} lacks a 'Gene Symbol' column.")
        print(f"  Available columns: {df_gpl.columns.tolist()}")
        return None, None
    
    probe2gene = df_gpl["Gene Symbol"].dropna()

    # Keep only probes that map to a gene symbol
    common_probes = set(raw.columns).intersection(probe2gene.index)
    expr = raw.loc[:, sorted(common_probes)].rename(columns=probe2gene.to_dict())

    # Collapse duplicate gene symbols by taking the mean
    expr = expr.groupby(axis=1, level=0).mean()
    print(f"  After mapping → genes, shape: {expr.shape}")

    # Filter to only labeled samples
    valid = labels.index.intersection(expr.index)
    expr = expr.loc[valid]
    labels = labels.loc[valid]
    print(f"  After filtering to labeled: expr = {expr.shape}, labels = {labels.shape}")
    return expr, labels

def analyze_metadata():
    """Analyze and print metadata patterns across datasets"""
    print("\n" + "="*80)
    print("METADATA ANALYSIS ACROSS DATASETS")
    print("="*80)
    
    for dataset_id, samples_metadata in metadata_analysis.items():
        print(f"\n--- {dataset_id} ---")
        print(f"Total samples: {len(samples_metadata)}")
        
        # Count labels
        label_counts = {'DMD (1)': 0, 'Control (0)': 0, 'Unlabeled': 0}
        
        # Collect unique metadata patterns
        dmd_patterns = set()
        control_patterns = set()
        unlabeled_patterns = set()
        
        for sample in samples_metadata:
            label = sample['assigned_label']
            metadata_entries = sample['metadata_entries']
            
            if label == 1:
                label_counts['DMD (1)'] += 1
                for entry in metadata_entries:
                    if 'dmd' in entry:
                        dmd_patterns.add(entry)
            elif label == 0:
                label_counts['Control (0)'] += 1
                for entry in metadata_entries:
                    if 'control' in entry or 'normal' in entry:
                        control_patterns.add(entry)
            else:
                label_counts['Unlabeled'] += 1
                for entry in metadata_entries[:3]:  # Show first 3 entries for unlabeled
                    unlabeled_patterns.add(entry)
        
        print(f"Label distribution: {label_counts}")
        
        if dmd_patterns:
            print(f"DMD-related metadata patterns ({len(dmd_patterns)}):")
            for pattern in sorted(dmd_patterns):
                print(f"  - '{pattern}'")
        
        if control_patterns:
            print(f"Control-related metadata patterns ({len(control_patterns)}):")
            for pattern in sorted(control_patterns):
                print(f"  - '{pattern}'")
        
        if unlabeled_patterns:
            print(f"Sample unlabeled metadata patterns ({len(unlabeled_patterns)}):")
            for pattern in sorted(list(unlabeled_patterns)[:5]):  # Show max 5
                print(f"  - '{pattern}'")

# -----------------------------------------------------------------------------
#  Step 1: Process each series
# -----------------------------------------------------------------------------
expr_list = []
label_list = []

for gid in SERIES_LIST:
    try:
        expr_df, labs = process_series(gid)
        if expr_df is None or expr_df.shape[0] == 0:
            print(f"  → {gid} produced 0 labeled samples; skipping.")
            continue
        expr_list.append(expr_df)
        label_list.append(labs)
    except Exception as e:
        print(f"  → Error processing {gid}: {e}")
        continue

# -----------------------------------------------------------------------------
#  Step 2: Analyze metadata patterns
# -----------------------------------------------------------------------------
analyze_metadata()

if len(expr_list) == 0:
    print("\nNo labeled samples found in any series. Check metadata analysis above.")
else:
    # -----------------------------------------------------------------------------
    #  Step 3: Intersect gene symbols across DataFrames
    # -----------------------------------------------------------------------------
    common_genes = set(expr_list[0].columns)
    for df in expr_list[1:]:
        common_genes &= set(df.columns)
    common_genes = sorted([g for g in common_genes if isinstance(g, str)])
    print(f"\nCommon genes across {len(expr_list)} series: {len(common_genes)}")

    # Subset each expression DataFrame to the common genes
    expr_list = [df.loc[:, common_genes] for df in expr_list]

    # -----------------------------------------------------------------------------
    #  Step 4: Concatenate into one combined DataFrame and add target column
    # -----------------------------------------------------------------------------
    X_all = pd.concat(expr_list, axis=0)
    y_all = pd.concat(label_list, axis=0)
    
    # Combine features and target into single DataFrame
    combined_df = X_all.copy()
    combined_df['target'] = y_all
    combined_df.fillna(0, inplace=True)
    
    print(f"Combined shape: {combined_df.shape}")
    print("Target distribution:\n", combined_df['target'].value_counts())

    # -----------------------------------------------------------------------------
    #  Step 5: Save to single CSV file
    # -----------------------------------------------------------------------------
    combined_df.to_csv(OUTPUT_FILE, index=True)
    print(f"\nSaved combined dataset to: {OUTPUT_FILE}")
    print(f"Columns: {len(combined_df.columns)} ({len(combined_df.columns)-1} genes + 1 target)")
    print(f"Samples: {len(combined_df)}")