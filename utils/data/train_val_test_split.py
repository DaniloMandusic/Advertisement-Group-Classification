from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_test_split(df, val_size, test_size, target_col, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
        shuffle=True,
        stratify=df[target_col]
    )

    # second split: val + test
    val_ratio = val_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_ratio,
        random_state=random_state,
        stratify=temp_df[target_col]
    )

    return train_df, val_df, test_df

def data_integrity_check(splits, target_col):
    train_df = splits["train"]
    val_df   = splits["val"]
    test_df  = splits["test"]

    print("Dataset sizes:")
    for name, df_ in splits.items():
        print(f"{name:5s}: {len(df_):,}")

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\nTotal: {total:,}\n")

    # 2. Class coverage check
    print("Class coverage (number of classes per split):")
    all_classes = set(train_df[target_col].unique())

    for name, df_ in splits.items():
        classes = set(df_[target_col].unique())
        missing = all_classes - classes

        print(f"{name:5s}: {len(classes)} classes")

        if missing:
            print(f"Missing classes: {sorted(list(missing))}")
        else:
            print("All classes present")

    # 3. Distribution comparison
    print("\nClass distribution (top differences):")

    dist = pd.DataFrame({
        "train": train_df[target_col].value_counts(normalize=True),
        "val": val_df[target_col].value_counts(normalize=True),
        "test": test_df[target_col].value_counts(normalize=True),
    }).fillna(0)

    diff = (dist.max(axis=1) - dist.min(axis=1)).sort_values(ascending=False)

    print("\nTop distribution shifts:")
    print(diff.head(10))

    # 4. integrity check
    print("\nData integrity checks:")
    assert len(train_df) > 0, "Train set is empty!"
    assert len(val_df) > 0, "Validation set is empty!"
    assert len(test_df) > 0, "Test set is empty!"

    print("All checks passed")

    print("\n==========================================\n")