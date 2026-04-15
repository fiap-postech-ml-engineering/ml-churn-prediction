import pandas as pd


def load_csv_data(file_path: str) -> pd.DataFrame | None:
    """
    Load data from a CSV file, with ";" as the separator.

    Parameters:
    file_path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame | None: A DataFrame containing the loaded data.
    """

    try:
        df = pd.read_csv(file_path, sep=";")
        print(f'1 file loaded: {file_path}')
        print(f'Dataframe shape: {df.shape}')
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"CSV parsing error in {file_path}: {e}")
        print("Maybe you're trying to force a \",\" separator?")
        return None
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {e}")
        return None
