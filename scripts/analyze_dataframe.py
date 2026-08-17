#!/usr/bin/env python3
"""
Script to analyze allmusic_youtube_discogs_reviews.pkl dataframe
- Counts NaN values per column
- Counts duplicated entries
"""

import pandas as pd
from pathlib import Path


def analyze_dataframe(pkl_path: Path):
    """Analyze a pickle dataframe for NaNs and duplicates."""

    print(f"Loading dataframe from: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    print(f"\n{'=' * 80}")
    print(f"DATAFRAME OVERVIEW")
    print(f"{'=' * 80}")
    print(f"Shape: {df.shape} (rows x columns)")
    print(f"Columns: {list(df.columns)}")

    print(f"\nColumn data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    # Count NaNs per column
    print(f"\n{'=' * 80}")
    print(f"NaN VALUES PER COLUMN")
    print(f"{'=' * 80}")
    nan_counts = df.isna().sum()
    nan_percentages = (df.isna().sum() / len(df)) * 100

    nan_summary = pd.DataFrame(
        {
            "Column": nan_counts.index,
            "NaN Count": nan_counts.values,
            "NaN %": nan_percentages.values,
        }
    ).sort_values("NaN Count", ascending=False)

    print(nan_summary.to_string(index=False))
    print(f"\nTotal NaN values in dataframe: {df.isna().sum().sum()}")

    # Print samples of review_text to validate data
    if "review_text" in df.columns:
        print(f"\n{'=' * 80}")
        print(f"REVIEW_TEXT SAMPLES (First 50 rows)")
        print(f"{'=' * 80}")
        sample_reviews = df[["review_text"]].head(50)
        for idx, (index, row) in enumerate(sample_reviews.iterrows(), 1):
            review = row["review_text"]
            if pd.isna(review):
                print(f"\n[{idx}] [NULL/NaN]")
            else:
                # Truncate at 150 chars for readability
                preview = str(review)[:150] + ("..." if len(str(review)) > 150 else "")
                print(f"\n[{idx}] {preview}")

    # Detailed analysis of missing/non-available values
    print(f"\n{'=' * 80}")
    print(f"MISSING/NON-AVAILABLE VALUES ANALYSIS")
    print(f"{'=' * 80}")

    # Count duplicated entries
    print(f"\n{'=' * 80}")
    print(f"DUPLICATE ENTRIES")
    print(f"{'=' * 80}")

    # Find hashable columns (exclude lists, dicts, etc.)
    hashable_cols = []
    unhashable_cols = []

    for col in df.columns:
        try:
            # Test if column is hashable by trying to hash a sample
            if len(df) > 0:
                sample = df[col].iloc[0]
                # Check if it's a scalar before using in boolean context
                if isinstance(sample, (list, dict, set)):
                    # These are unhashable types
                    raise TypeError(f"Column {col} contains unhashable type")
                if not pd.isna(sample):
                    hash(sample)
            hashable_cols.append(col)
        except (TypeError, AttributeError, ValueError):
            unhashable_cols.append(col)

    if unhashable_cols:
        print(f"Note: Columns with unhashable types (lists/dicts): {unhashable_cols}")
        print(f"Duplicate checking will be performed on hashable columns only.\n")

    # Check for complete row duplicates (only on hashable columns)
    if hashable_cols:
        try:
            duplicate_rows = df[hashable_cols].duplicated().sum()
            print(f"Complete duplicate rows (hashable columns): {duplicate_rows}")

            # Check for duplicates keeping first occurrence
            duplicate_rows_keep_first = df[hashable_cols].duplicated(keep="first").sum()
            print(
                f"Duplicate rows excluding first occurrence: {duplicate_rows_keep_first}"
            )
        except Exception as e:
            print(f"Error checking row duplicates: {e}")
    else:
        print("No hashable columns found for duplicate checking.")

    # Check for duplicates per column (if any obvious ID columns exist)
    print(f"\nDuplicates per column:")
    for col in df.columns:
        if col in hashable_cols:
            if "id" in col.lower() or col in ["track", "title", "url", "name"]:
                try:
                    dup_in_col = df[col].duplicated().sum()
                    if dup_in_col > 0:
                        print(f"  '{col}': {dup_in_col} duplicates")
                except Exception as e:
                    print(f"  '{col}': Error checking duplicates - {e}")

    # Convert list columns to sets and merge duplicates by index
    print(f"\n{'=' * 80}")
    print(f"MERGING DUPLICATES BY INDEX")
    print(f"{'=' * 80}")

    print(f"Using dataframe index for merging")
    df_merged = df.copy()

    # Convert list columns to frozensets
    print(f"\nConverting list columns to frozensets: {unhashable_cols}")
    for col in unhashable_cols:
        df_merged[col] = [
            frozenset(x) if isinstance(x, list) else x for x in df_merged[col]
        ]

    # Group by index and merge
    def merge_with_sets(group):
        """Merge rows with same index by unioning the frozensets."""
        result = {}
        for col in group.columns:
            if col in unhashable_cols:
                # Union all frozensets
                merged_set = set()
                for val in group[col]:
                    if isinstance(val, frozenset):
                        merged_set.update(val)
                result[col] = frozenset(merged_set) if merged_set else None
            else:
                # For other columns, take first non-null value
                non_null = group[col].dropna()
                result[col] = non_null.iloc[0] if len(non_null) > 0 else None
        return pd.Series(result)

    try:
        df_merged = df_merged.groupby(level=0).apply(merge_with_sets)

        # Replace empty frozensets with NaN
        print(f"\nReplacing empty frozensets with NaN...")
        for col in unhashable_cols:
            df_merged[col] = [
                None if isinstance(x, frozenset) and len(x) == 0 else x
                for x in df_merged[col]
            ]

        print(f"\nMerge completed:")
        print(f"  Original shape: {df.shape}")
        print(f"  Merged shape: {df_merged.shape}")
        print(f"  Rows merged: {df.shape[0] - df_merged.shape[0]}")

        # Compute stats on merged dataframe
        print(f"\n{'=' * 80}")
        print(f"STATS AFTER MERGING")
        print(f"{'=' * 80}")

        # NaN counts after merge
        nan_counts_merged = df_merged.isna().sum()
        nan_percentages_merged = (df_merged.isna().sum() / len(df_merged)) * 100

        nan_summary_merged = pd.DataFrame(
            {
                "Column": nan_counts_merged.index,
                "NaN Count": nan_counts_merged.values,
                "NaN %": nan_percentages_merged.values,
            }
        ).sort_values("NaN Count", ascending=False)

        print("\nNaN values after merge:")
        print(nan_summary_merged.to_string(index=False))

        # Save merged dataframe
        output_path = pkl_path.parent / f"{pkl_path.stem}_merged_sets.pkl"
        df_merged.to_pickle(output_path)
        print(f"\nMerged dataframe saved to: {output_path}")

        # Use the merged dataframe for further analysis
        df = df_merged

    except Exception as e:
        print(f"Error merging duplicates: {e}")
        print("Continuing with original dataframe.")

    # Memory usage
    print(f"\n{'=' * 80}")
    print(f"MEMORY USAGE")
    print(f"{'=' * 80}")
    memory_usage = df.memory_usage(deep=True)
    print(memory_usage.to_string())
    print(f"\nTotal memory: {memory_usage.sum() / 1024**2:.2f} MB")

    # Data types
    print(f"\n{'=' * 80}")
    print(f"DATA TYPES")
    print(f"{'=' * 80}")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    # Default path based on the project structure
    pkl_path = (
        Path(__file__).parent.parent
        / "notebooks"
        / "allmusic_youtube_discogs_reviews.pkl"
    )

    if not pkl_path.exists():
        print(f"Error: File not found at {pkl_path}")
        print("Please provide the correct path to the pickle file.")
        exit(1)

    analyze_dataframe(pkl_path)
