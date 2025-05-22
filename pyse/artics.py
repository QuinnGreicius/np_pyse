import numpy as np
import pandas as pd

def populate_dataframe(artics_dir, artics_df, artic_cols):
    """
    Populates a DataFrame with data from .npy files in a specified directory.

    This function reads .npy files from the given directory, processes the data,
    and appends it to the provided DataFrame. It ensures that the data conforms
    to the expected number of columns by either truncating or padding with NaN
    values. The onset and offset values are extracted from the filenames and
    added as columns to the DataFrame.

    Args:
        artics_dir (Path or None): The directory containing .npy files. If None,
            the function returns the input DataFrame unchanged.
        artics_df (pd.DataFrame): The DataFrame to populate with data.
        artic_cols (list of str): The expected column names for the DataFrame,
            including 'onset' and 'offset'.

    Returns:
        pd.DataFrame: The updated DataFrame containing the concatenated and
        processed data from the .npy files.

    Notes:
        - The function assumes that the filenames of the .npy files contain
          onset and offset values as the second and third underscore-separated
          components, respectively.
        - If a .npy file has more columns than expected, the extra columns are
          truncated. If it has fewer columns, the missing columns are padded
          with NaN values.
        - The resulting DataFrame is sorted by the 'onset' column.
    """
    if not artics_dir is None:
        all_rows = []
        for file in artics_dir.glob("*.npy"):
            if file.name.endswith(".npy"):
                data = np.load(file)
                if data.shape[0] == 0:
                    continue

                n_expected = len(artic_cols)
                n_actual = data.shape[1]
                if n_actual != n_expected:
                    print(f"WARNING: {file.name} has shape {data.shape} but expected {n_expected} columns")
                    if n_actual > n_expected:
                        data = data[:, :n_expected]
                    else:
                        # Pad with NaNs for missing columns
                        pad_width = n_expected - n_actual
                        data = np.hstack([data, np.full((data.shape[0], pad_width), np.nan)])

                row_dict = {col: data[:, i] for i, col in enumerate(artic_cols)}
                row_dict['onset'] = float(file.name.split("_")[1])
                row_dict['offset'] = float(file.name.split("_")[2].split(".npy")[0])
                all_rows.append(row_dict)

        artics_df = pd.DataFrame(all_rows)
        artics_df = artics_df.sort_values(by='onset').reset_index(drop=True)
    return artics_df

def init_artics_new_df(se: "SE") -> None:
    """
    Initializes and populates the `artics_new_df` attribute of the given `SE` object.

    This function creates a DataFrame with specific articulatory columns and their
    corresponding velocity and acceleration columns. The DataFrame is populated
    using data from the `artics_new_dir` directory of the `SE` object.

    Args:
        se (SE): An instance of the `SE` class that contains the `artics_new_dir`
                 directory and where the resulting DataFrame will be stored.

    Returns:
        None: The function modifies the `se` object in place by setting its
              `artics_new_df` attribute.

    Attributes Modified:
        se.artics_new_df (pd.DataFrame): A DataFrame containing articulatory data
                                         along with velocity and acceleration
                                         columns for each articulatory feature.

    Notes:
        - The articulatory features include positions (e.g., 'tt_x', 'tt_y') and
          other metrics (e.g., 'ja', 'ttcc').
        - Velocity and acceleration are computed for each feature using numerical
          gradients.
    """
    artics_new_cols = ['tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y',
        'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y', 'loud', 'f0']

    artics_new_df = pd.DataFrame(columns=['onset', 'offset'] + artics_new_cols)
    artics_new_df = populate_dataframe(se.artics_new_dir, artics_new_df, artics_new_cols)
    for col in artics_new_cols:
        artics_new_df[col + '_vel'] = artics_new_df[col].apply(lambda x: np.gradient(x))
        artics_new_df[col + '_accel'] = artics_new_df[col + '_vel'].apply(lambda x: np.gradient(x))

    artics_new_df = artics_new_df[['onset', 'offset'] + [col for col in artics_new_df.columns if col not in ['onset', 'offset']]]
    se.artics_new_df = artics_new_df

def init_artics_old_df(se: "SE") -> None:
    """
    Initializes the `artics_old_df` attribute of the given `SE` object.

    This function creates a DataFrame with columns representing articulatory 
    features and their corresponding velocity and acceleration. The DataFrame 
    is populated using data from the `artics_old_dir` directory of the `SE` 
    object.

    Args:
        se (SE): An instance of the `SE` class. The `artics_old_dir` attribute 
                 of this object is used to populate the DataFrame, and the 
                 resulting DataFrame is assigned to the `artics_old_df` 
                 attribute.

    Columns in the DataFrame:
        - 'onset': Onset time of the articulatory feature.
        - 'offset': Offset time of the articulatory feature.
        - Articulatory features: [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y',
        'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y',
        'la', 'pro', 'ttcl', 'tbcl', 'v_x', 'v_y'].
        - Velocity columns: Derived as the gradient of each articulatory 
          feature column.
        - Acceleration columns: Derived as the gradient of each velocity 
          column.

    Returns:
        None: The function modifies the `se` object in place by setting its 
              `artics_old_df` attribute.mak
    """
    artics_old_cols = [
        'tt_x', 'tt_y', 'td_x', 'td_y', 'tb_x', 'tb_y',
        'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y',
        'la', 'pro', 'ttcl', 'tbcl', 'v_x', 'v_y'
    ]


    artics_old_df = pd.DataFrame(columns=['onset', 'offset'] + artics_old_cols)
    artics_old_df = populate_dataframe(se.artics_old_dir, artics_old_df, artics_old_cols)
    for col in artics_old_cols:
        artics_old_df[col + '_vel'] = artics_old_df[col].apply(lambda x: np.gradient(x))
        artics_old_df[col + '_accel'] = artics_old_df[col + '_vel'].apply(lambda x: np.gradient(x))
        
    artics_old_df = artics_old_df[['onset', 'offset'] + [col for col in artics_old_df.columns if col not in ['onset', 'offset']]]
    se.artics_old_df = artics_old_df

def init_artics_merged_df(se: "SE") -> None:
    """
    Initializes the `artics_merged_df` attribute for the given `SE` object by merging
    the `artics_old_df` and `artics_new_df` DataFrames. If either of these DataFrames
    is not already initialized, it calls their respective initialization functions.

    The merging is performed on the columns `onset` and `offset` using an outer join.
    For overlapping columns (other than `onset` and `offset`), values from 
    `artics_new_df` are preferred over those from `artics_old_df`. If a value is 
    missing in `artics_new_df`, the corresponding value from `artics_old_df` is used.

    Args:
        se (SE): An object that is expected to have the attributes `artics_old_df`, 
                 `artics_new_df`, and `artics_merged_df`. If `artics_old_df` or 
                 `artics_new_df` is not initialized, their respective initialization 
                 functions (`init_artics_old_df` and `init_artics_new_df`) are called.

    Returns:
        None: This function modifies the `se` object in place by setting its 
              `artics_merged_df` attribute.

    Raises:
        AttributeError: If the `se` object does not have the required attributes 
                        or if the initialization functions are not defined.

    Notes:
        - The `artics_old_df` and `artics_new_df` DataFrames must have the columns 
          `onset` and `offset` for the merge to work correctly.
        - The merged DataFrame will only retain one version of overlapping columns, 
          with preference given to the values from `artics_new_df`.
    """
    if not hasattr(se, 'artics_old_df') or se.artics_old_df is None:
        init_artics_old_df(se)
    if not hasattr(se, 'artics_new_df') or se.artics_new_df is None:
        init_artics_new_df(se)
    artics_old_df = se.artics_old_df
    artics_new_df = se.artics_new_df
    artics_merged_df = pd.merge(
        artics_old_df, 
        artics_new_df, 
        on=['onset', 'offset'], 
        how='outer', 
        suffixes=('_old', '_new')
    )
    # For overlapping columns, prefer values from artics_new_df
    for col in artics_new_df.columns:
        if col not in ['onset', 'offset']:
            if col in artics_old_df.columns:
                # Use the new value where available, else fallback to old
                artics_merged_df[col] = artics_merged_df[f'{col}_new'].combine_first(artics_merged_df[f'{col}_old'])
                artics_merged_df.drop([f'{col}_old', f'{col}_new'], axis=1, inplace=True)
    se.artics_merged_df = artics_merged_df