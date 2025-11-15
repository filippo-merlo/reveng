#!/usr/bin/env python3
"""Filter CSV to keep only columns needed for decoder training.

Removes all partial observability (po_*) columns and other unnecessary data,
keeping only full observability prompts and action sequences.
"""

import argparse
import pandas as pd
from pathlib import Path


def filter_csv(input_csv: str, output_csv: str):
    """Filter CSV to keep only decoder training columns.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    print(f"Reading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Original rows: {len(df)}")
    
    # Columns to keep for decoder training
    columns_to_keep = [
        'env_idx',
        'fo_observation',  # Full observability grid (for grid text decoder)
        'fo_prompt',       # Full observability prompt (for LLM activation decoder)
        'action_sequence', # Optimal action sequence (labels)
        'start_pos',       # Agent start position (metadata)
        'goal_pos',        # Goal position (metadata)
        'optimal_trajectory_length',  # Length of optimal path (metadata)
    ]
    
    # Filter to only keep columns that exist in the dataframe
    available_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    df_filtered = df[available_columns]
    
    print(f"\nFiltered columns: {list(df_filtered.columns)}")
    print(f"Filtered rows: {len(df_filtered)}")
    
    # Save filtered CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)
    
    print(f"\nFiltered CSV saved to: {output_path}")
    print(f"Removed columns: {set(df.columns) - set(available_columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter CSV to keep only decoder training columns"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: adds '_decoder_training' suffix)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_csv)
        output_path = input_path.parent / f"{input_path.stem}_decoder_training{input_path.suffix}"
        args.output = str(output_path)
    
    filter_csv(args.input_csv, args.output)

