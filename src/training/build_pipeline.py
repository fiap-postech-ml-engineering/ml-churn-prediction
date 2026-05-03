import pandas as pd

from src.config.settings import (
    DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data.data_cleaning import clean_dataframe_for_modeling
from src.data.data_split import split_features_target, split_train_val_test
from src.data.feature_selection import select_tabular_raw_features
from src.data.tabular_pipeline import (
    fit_tabular_preprocessing_pipeline,
    save_tabular_preprocessing_pipeline,
)

RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"


def build_pipeline() -> None:
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df = select_tabular_raw_features(df, require_target=True)
    df = clean_dataframe_for_modeling(df)

    x, y = split_features_target(df)
    x_train, _, _, _, _, _ = split_train_val_test(
        x, y, test_size=TEST_SIZE, val_size=VALIDATION_SIZE, seed=RANDOM_SEED
    )

    pipeline, feature_names = fit_tabular_preprocessing_pipeline(x_train)
    output_path = save_tabular_preprocessing_pipeline(pipeline, feature_names)

    print(f"Pipeline salvo em: {output_path}")
    print(f"Features geradas: {len(feature_names)}")


if __name__ == "__main__":
    build_pipeline()
