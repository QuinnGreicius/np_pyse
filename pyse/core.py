import numpy as np
from pathlib import Path
from . import syllabify, textgrid, enrich_features, artics, extract_waveform
import librosa
import soundfile as sf
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple, Dict
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.colors import to_hex
from tqdm import tqdm, trange

def compute_psth(spike_array, bins, sigma=2):
    """
    Compute the Peri-Stimulus Time Histogram (PSTH) for a given spike train.

    Parameters:
        spike_array (array-like): Array of spike times.
        bins (array-like): Array of bin edges for histogram computation.
        sigma (float, optional): Standard deviation for Gaussian smoothing. Default is 0.5.

    Returns:
        numpy.ndarray: Smoothed PSTH values in Hz.
    """
    bin_size = bins[1] - bins[0]
    spike_counts = np.histogram(spike_array, bins=bins)[0]
    spike_rates = spike_counts / (bin_size)  # Convert to Hz
    smoothed_psth = gaussian_filter1d(spike_rates, sigma=sigma)
    return smoothed_psth

def find_median_by_group(group_sizes, production_times):
    """
    Calculate the median production times for each group and replace the original production times with the median values.

    Args:
        group_sizes (list of int): A list of integers where each integer represents the size of a group.
        production_times (list of list of float): A list of lists where each inner list contains production times.

    Returns:
        list of list of float: The modified production_times list where each group's production times are replaced with the median production times of that group.

    Example:
        group_sizes = [2, 3]
        production_times = [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5], [2.5, 3.5], [3.0, 4.0]]
        result = find_median_by_group(group_sizes, production_times)
        # result will be [[1.5, 2.5], [1.5, 2.5], [2.5, 3.5], [2.5, 3.5], [2.5, 3.5]]
    """
    group_sizes = [0] + group_sizes
    for i in range(len(group_sizes)-1):
        start = group_sizes[i]
        end = group_sizes[i+1]
        group = production_times[start:end]
        max_len = max(len(arr) for arr in group)
        padded_arrays = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in group]
        stacked_group = np.vstack(padded_arrays)
        median_group = np.nanmedian(stacked_group, axis=0)
        production_times[start:end] = [median_group for _ in group]
    return production_times

class SE:
    """
    SE Class
    The SE class is designed to handle and process data related to neural recordings, audio, and associated metadata. 
    It provides methods for initializing directories, processing labels, extracting audio features, and analyzing neural data.
    """
    
    def __init__(self,
                 recid: str,
                 preproc_root: Path = Path("/data_store2/neuropixels/preproc/"),
                 ks_suffix: str = "KS4", # "KS4" or "KS4_Th=8" or "" for Kilosort 2.5
                 ks_dirs: Optional[Union[Path, List[Path]]] = None,
                 label_dir: Optional[Path] = None,
                 audio_file: Optional[Path] = None,
                 artics_old_dir: Optional[Path] = None,
                 artics_new_dir: Optional[Path] = None,
                 text_grid_dir: Optional[Path] = None
                 ):
        """
        Initializes the class with the provided parameters.
        Args:
            recid (str): The recording ID associated with the data.
            preproc_root (Path, optional): The root directory for preprocessed data. 
                Defaults to Path("/data_store2/neuropixels/preproc/").
            ks_suffix (str, optional): The suffix for Kilosort directories. 
                Can be "KS4", "KS4_Th=8", or an empty string for Kilosort 2.5. Defaults to "KS4".
            ks_dirs (Optional[Union[Path, List[Path]]], optional): The Kilosort directories. 
                If None, directories will be initialized automatically. Defaults to None.
            label_dir (Optional[Path], optional): The directory containing label files. Defaults to None.
            audio_file (Optional[Path], optional): The path to the associated audio file. Defaults to None.
            artics_old_dir (Optional[Path], optional): The directory containing articulation data. Defaults to None.
            artics_new_dir (Optional[Path], optional): The directory containing new articulation data. Defaults to None.
            text_grid_dir (Optional[Path], optional): The directory containing TextGrid files. Defaults to None.
        Attributes:
            recid (str): The recording ID.
            preproc_root (Path): The root directory for preprocessed data.
            ks_suffix (str): The suffix for Kilosort directories.
            ks_dirs (Union[Path, List[Path]]): The initialized Kilosort directories.
            task_table (Any): The task table initialized from label data.
            task_names (Any): The task names initialized from label data.
            task_timings (Any): The task timings initialized from label data.
            label_dir (Path): The directory containing label files.
            audio_file (Path): The path to the associated audio file.
            text_grid_dir (Path): The directory containing TextGrid files.
            artics_old_dir (Path): The directory containing articulation data.
            artics_new_dir (Path): The directory containing new articulation data.
        """
        self.recid = recid
        self.preproc_root = preproc_root

        self.ks_suffix = "_" + ks_suffix if len(ks_suffix) > 0 else ""
        self.ks_dirs = self._init_ks_dirs(ks_dirs)
        self.task_table, self.task_names, self.task_timings, self.label_dir = self._init_labels(label_dir)
        self.audio_file = self._init_audio(audio_file)
        self.text_grid_dir = self._init_text_grid(text_grid_dir)
        self.artics_old_dir = self._init_artics(artics_old_dir)
        self.artics_new_dir = self._init_artics_new(artics_new_dir)
        
        self._init_mpl_params()

    
    def _init_ks_dirs(self, ks_dirs: Optional[Union[Path, List[Path]]]) -> List[Path]:
        """
        Initializes and retrieves the Kilosort directories for the specified recording ID.
        This method determines the appropriate Kilosort directories based on the provided
        `ks_dirs` parameter or by searching the default directory structure. It handles
        cases where directories are missing or probes are excluded.
        Args:
            ks_dirs (Optional[Union[Path, List[Path]]]): 
                - If None, the method searches for Kilosort directories in the default 
                  location based on the recording ID.
                - If a single Path or string, it validates the existence of the directory.
                - If a list of Paths or strings, it validates the existence of each directory.
        Returns:
            List[Path]: A list of valid Kilosort directories.
        Raises:
            FileNotFoundError: If a required Kilosort directory does not exist.
        """
        if ks_dirs is None:
            ks_dirs = []
            kilosort_dir = self.preproc_root / self.recid / "kilosort"
            if not kilosort_dir.exists():
                raise FileNotFoundError(f"{self.recid} Kilosort directory does not exist.")
            excluded_probes = []
            if self.ks_suffix:
                pattern = f"{self.recid}_g*_imec*{self.ks_suffix}"
            else:
                pattern = f"{self.recid}_g*_imec[0-9]"  # only matches imec followed by a digit

            if (kilosort_dir / f"mc_{self.recid}_g0").exists():
                mc_dir = kilosort_dir / f"mc_{self.recid}_g0"
                mc_folders = glob.glob(str(mc_dir / pattern))
                # Extract the probe indices from the folder names and exclude them 
                probe_inds = [int(re.search(r"imec(\d+)", folder).group(1)) for folder in mc_folders]
                excluded_probes.extend(probe_inds)
                ks_dirs.extend(mc_folders)
            
            catgt_dir = kilosort_dir / f"catgt_{self.recid}_g0"
            catgt_folders = glob.glob(str(catgt_dir / pattern))
            catgt_folders = [folder for folder in catgt_folders if int(re.search(r"imec(\d+)", folder).group(1)) not in excluded_probes]
            ks_dirs.extend(catgt_folders)
            return [Path(folder) for folder in ks_dirs]
        elif isinstance(ks_dirs, (str, Path)):
            ks_dirs = Path(ks_dirs)
            if not ks_dirs.exists():
                raise FileNotFoundError(f"Kilosort directory {ks_dirs} does not exist.")
            ks_dirs = [ks_dirs]
        elif isinstance(ks_dirs, list):
            ks_dirs = [Path(folder) for folder in ks_dirs]
            for folder in ks_dirs:
                if not folder.exists():
                    raise FileNotFoundError(f"Kilosort directory {folder} does not exist.")
        return ks_dirs
    
    def _init_labels(self, label_dir: Optional[Path]) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, float, float]], Path]:
        """
        Initialize and process label data from the specified directory.
        This method reads timing and label information from various files in the
        provided `label_dir` directory. It processes the data into a unified
        DataFrame and extracts additional metadata such as task names and task
        timing information.
        Args:
            label_dir (Optional[Path]): The directory containing label files. If None,
            the method defaults to using the `preproc_root` directory combined
            with the `recid` and "labels" subdirectory.
        Returns:
            Tuple[pd.DataFrame, List[str], List[Tuple[str, float, float]], Path]:
            - A DataFrame (`task_table`) containing combined label data with columns:
              ['onset', 'offset', 'label', 'type', 'speech', 'task'].
            - A list of task names (`task_names`) extracted from subdirectories in `label_dir`.
            - A list of task timing information (`task_timing`) in the format:
              [(task_name, task_start, task_end)], or None if no timing file exists.
            - The `label_dir` Path used for processing.
        Notes:
            - The method expects specific files in each task subdirectory:
              'speech_stim_timing.txt', 'stim_timing.txt', 'speech_prod_timing.txt',
              and 'cues_timing.txt'. If these files exist, their data is read and
              added to the `task_table` DataFrame.
            - If a "task_timing.csv" file exists in `label_dir`, it is used to extract
              task timing information.
            - If `label_dir` does not exist, the method returns None for all outputs.
        """
        task_table = pd.DataFrame(columns= ['onset', 'offset', 'label', 'type', 'speech', 'task'])
        if label_dir is None:
            label_dir = self.preproc_root / self.recid / "labels"
        if not label_dir.exists():
            return None, None, None, None
        tasks = [d for d in label_dir.iterdir() if d.is_dir()]
        task_names = [task.name for task in tasks]
        for task in tasks:
            task_name = task.name
            speech_stim_timing = task / 'speech_stim_timing.txt'
            stim_timing = task / 'stim_timing.txt'
            speech_prod_timing = task / 'speech_prod_timing.txt'
            cues_timing = task / 'cues_timing.txt' 

            stim_df = pd.DataFrame(columns=task_table.columns)
            
            if speech_stim_timing.exists():
                speech_stim_df = pd.DataFrame(columns=task_table.columns)
                speech_stim_df = pd.read_csv(speech_stim_timing, sep="\t", header=None)
                speech_stim_df.columns = ['onset', 'offset', 'label']
                speech_stim_df['type'] = 'stim'
                speech_stim_df['task'] = task_name
                speech_stim_df['speech'] = True

                if stim_timing.exists():
                    stim_df = pd.read_csv(stim_timing, sep="\t", header=None)
                    stim_df.columns = ['onset', 'offset', 'stimID']
                    stim_df = stim_df.merge(speech_stim_df, how='left', on=['onset', 'offset'])
                else:
                    stim_df = speech_stim_df.copy()
            elif stim_timing.exists():
                stim_df = pd.read_csv(stim_timing, sep="\t", header=None)
                stim_df.columns = ['onset', 'offset', 'stimLabel']
                stim_df['stimID'] = stim_df['stimLabel']
                stim_df['type'] = 'stim'
                stim_df['task'] = task_name
                stim_df['speech'] = False

            # Merge stim_df with speech_stim_df on onset
            prod_df = pd.DataFrame(columns=task_table.columns)
            if speech_prod_timing.exists():
                prod_df = pd.read_csv(speech_prod_timing, sep="\t", header=None)
                prod_df.columns = ['onset', 'offset', 'label']
                prod_df['type'] = 'prod'
                prod_df['task'] = task_name
                prod_df['speech'] = True

            cues_df = pd.DataFrame(columns=task_table.columns)
            if cues_timing.exists():
                cues_df = pd.read_csv(cues_timing, sep="\t", header=None)
                cues_df.columns = ['onset', 'offset', 'label']
                cues_df['type'] = 'cues'
                cues_df['task'] = task_name
                cues_df['speech'] = False

            task_table = pd.concat([task_table, stim_df, prod_df, cues_df], ignore_index=True)
        task_table = task_table.sort_values(by=['onset']).reset_index(drop=True)
        task_table['label'] = task_table['label'].astype(str).str.strip()
        task_timing = label_dir / "task_timing.csv"
        if task_timing.exists():
            task_timing = pd.read_csv(task_timing)
            task_timing = task_timing[['Task name', 'Task start', 'Task end']]
            task_timing = task_timing.values.tolist()
        else:
            task_timing = None
        return task_table, task_names, task_timing, label_dir              
    
    def _init_audio(self, audio_file: Optional[Path]) -> Optional[Path]:
        """
        Initializes and validates the audio file path.

        This method determines the audio file path based on the provided `audio_file`
        argument or constructs a default path if `audio_file` is None. It also ensures
        that the resolved path exists, issuing a warning if it does not.

        Args:
            audio_file (Optional[Path]): The path to the audio file. If None, a default
                path is constructed based on `self.preproc_root` and `self.recid`.

        Returns:
            Optional[Path]: The validated audio file path if it exists, or None if the
                file does not exist.
        """
        if audio_file is None:
            audio_file = self.preproc_root / self.recid / "audio_files" / f"{self.recid}_mic_denoised.wav"
        if isinstance(audio_file, str):
            audio_file = Path(audio_file)
        if not audio_file.exists():
            print(f"Warning: Audio file {audio_file} does not exist.")
            audio_file = None
        return audio_file

    def _init_text_grid(self, text_grid_dir: Optional[Path]) -> Optional[Path]:
        """
        Initializes the TextGrid directory path.

        This method determines the path to the TextGrid directory based on the provided
        `text_grid_dir` argument. If `text_grid_dir` is not provided, it defaults to a 
        specific subdirectory under the `preproc_root` attribute. If the provided or 
        default path does not exist, a warning is printed, and the method returns `None`.

        Args:
            text_grid_dir (Optional[Path]): The directory path to the TextGrid files. 
                If None, a default path is constructed.

        Returns:
            Optional[Path]: The resolved TextGrid directory path if it exists, otherwise None.
        """
        if text_grid_dir is None:
            text_grid_dir = self.preproc_root / self.recid / "phones" / "results" / "all"
        if isinstance(text_grid_dir, str):
            text_grid_dir = Path(text_grid_dir)
        if not text_grid_dir.exists():
            print(f"Warning: TextGrid directory {text_grid_dir} does not exist.")
            text_grid_dir = None
        return text_grid_dir
    
    def _init_artics_new(self, artics_new_dir: Optional[Path]) -> Optional[Path]:
        """
        Initializes the articulatory directory path.

        This method determines the path to the articulatory directory based on the
        provided `artics_new_dir` parameter. If the parameter is `None`, it constructs
        the path using the `preproc_root`, `recid`, and a default subdirectory name 
        "artics_new". If the provided path is a string, it converts it to a `Path` object.
        If the directory does not exist, a warning is printed, and the method returns `None`.

        Args:
            artics_new_dir (Optional[Path]): The path to the articulatory directory. 
                If `None`, a default path is constructed.

        Returns:
            Optional[Path]: The validated and resolved path to the articulatory directory,
            or `None` if the directory does not exist.
        """
        if artics_new_dir is None:
            artics_new_dir = self.preproc_root / self.recid / "artics_new"
        if isinstance(artics_new_dir, str):
            artics_new_dir = Path(artics_new_dir)
        if not artics_new_dir.exists():
            print(f"Warning: Articulatory directory {artics_new_dir} does not exist.")
            artics_new_dir = None
        return artics_new_dir
    
    def _init_artics(self, artics_old_dir: Optional[Path]) -> Optional[Path]:
        """
        Initializes the articulatory directory path.

        This method determines the path to the articulatory directory based on the 
        provided `artics_old_dir` parameter. If `artics_old_dir` is not provided, it defaults 
        to a subdirectory named "artics" within the preprocessed root directory for 
        the current record ID. If the specified or default directory does not exist, 
        a warning is printed, and `None` is returned.

        Args:
            artics_old_dir (Optional[Path]): The path to the articulatory directory. 
                If None, a default path is constructed.

        Returns:
            Optional[Path]: The resolved articulatory directory path, or None if 
            the directory does not exist.
        """
        if artics_old_dir is None:
            artics_old_dir = self.preproc_root / self.recid / "artics" / "F01_indep_Haskins_loss_90_filter_fix_bn_False_0_setting2"
        if isinstance(artics_old_dir, str):
            artics_old_dir = Path(artics_old_dir)
        if not artics_old_dir.exists():
            print(f"Warning: Articulatory directory {artics_old_dir} does not exist.")
            artics_old_dir = None
        return artics_old_dir

    def _init_mpl_params(self):
        import matplotlib

        # Allow for embedding fonts in PDF and PS files
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['savefig.transparent'] = True
        matplotlib.rcParams['savefig.dpi'] = 600
    
    def get_artics_df(self, velocity = False, acceleration = False, which_artics = 'merged', verbose = False) -> pd.DataFrame:
        """
        Retrieve a DataFrame containing articulatory data with optional filtering for velocity and acceleration.
        Parameters:
        -----------
        velocity : bool, optional
            If False (default), columns ending with '_vel' (velocity data) will be dropped from the DataFrame.
        acceleration : bool, optional
            If False (default), columns ending with '_accel' (acceleration data) will be dropped from the DataFrame.
        which_artics : str, optional
            Specifies which articulatory dataset to retrieve. Options are:
            - 'old': Retrieve the old articulatory dataset.
            - 'new': Retrieve the new articulatory dataset.
            - 'merged': Retrieve the merged articulatory dataset (default).
            Raises a ValueError if an invalid option is provided.
        verbose : bool, optional
            If True (default), prints annotations for the articulatory columns based on their prefixes.
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the requested articulatory data, optionally filtered for velocity and acceleration.
        Raises:
        -------
        ValueError
            If `which_artics` is not one of 'old', 'new', or 'merged'.
        Notes:
        ------
        - The method initializes the requested articulatory dataset if it has not been initialized already.
        - Column annotations are printed in verbose mode, providing descriptions for recognized prefixes.
        Annotations:
        ------------
        The following prefixes in column names are annotated:
        - 'tt': tongue tip
        - 'td': tongue dorsum
        - 'tb': tongue body
        - 'li': lower incisor
        - 'ul': upper lip
        - 'll': lower lip
        - 'la': lip aperture
        - 'pro': lip protrusion
        - 'ja': jaw aperture
        - 'v': velum
        - 'ttcc': tongue tip constriction location
        - 'tbcc': tongue body constriction location
        - 'tdcc': tongue dorsum constriction cosine
        - 'loud': loudness
        - 'f0': pitch
        """
        if not hasattr(self, '_artic_verbose'):
            self._artic_verbose = True
        if which_artics.lower() == 'old':
            if not hasattr(self, 'artics_old_df') or self.artics_old_df is None:
                artics.init_artics_old_df(self)
            ouput = self.artics_old_df
        elif which_artics.lower() == 'new':
            if not hasattr(self, 'artics_new_df') or self.artics_new_df is None:
                artics.init_artics_new_df(self)
            ouput = self.artics_new_df
        elif which_artics.lower() == 'merged':
            if not hasattr(self, 'artics_merged_df') or self.artics_merged_df is None:
                artics.init_artics_merged_df(self)
            ouput = self.artics_merged_df
        else:
            raise ValueError("Invalid value for which_artics. Choose 'old', 'new', or 'merged'.")

        if not velocity:
            ouput = ouput.drop(columns=[col for col in ouput.columns if col.startswith('d_')])
        if not acceleration:
            ouput = ouput.drop(columns=[col for col in ouput.columns if col.startswith('dd_')])
        
        annotations = {
            'tt': 'tongue tip',
            'td': 'tongue dorsum',
            'tb': 'tongue body',
            'li': 'lower incisor',
            'ul': 'upper lip',
            'll': 'lower lip',
            'la': 'lip aperture',
            'pro': 'lip protrusion',
            'ja': 'jaw aperture',
            'v': 'velum',
            'ttcc': 'tongue tip constriction location',
            'tbcc': 'tongue body constriction location',
            'tdcc': 'tongue dorsum constriction cosine',
            'loud': 'loudness',
            'f0': 'pitch'
        }

        if verbose or self._artic_verbose:
            for k, v in annotations.items():
                print(f"{k}: {v}")
            self._artic_verbose = False

        return ouput            
    
    def _process_tg(self, tg_file, onset, offset, label_type, trial_index, label, tier_name):
            if pd.isna(onset) or pd.isna(offset) or pd.isna(tg_file) or pd.isna(label):
                return [[np.nan, np.nan, np.nan, label_type, trial_index, label]]
            if not Path(self.text_grid_dir / tg_file).exists():
                return [[np.nan, np.nan, np.nan, label_type, trial_index, label]]
            tg = textgrid.TextGrid.fromFile(self.text_grid_dir / tg_file)
            entries = []
            for tier in tg.tiers:
                if isinstance(tier, textgrid.IntervalTier) and tier.name == tier_name:
                    for interval in tier.intervals:
                        entries.append([
                            interval.minTime + onset,
                            interval.maxTime + onset,
                            interval.mark,
                            label_type,
                            trial_index,
                            label
                        ])
            return entries or [[np.nan, np.nan, np.nan, label_type, trial_index, label]]

    def get_word_df(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a DataFrame containing word-level annotations derived from TextGrid files.
        This method processes a DataFrame of speech labels and extracts word-level
        information from corresponding TextGrid files. It aligns the word-level
        annotations with the provided speech labels and returns a DataFrame with
        detailed word information.
        Args:
            labels_df (pd.DataFrame): A DataFrame containing speech labels with the following columns:
            - 'onset': The start time of the speech segment.
            - 'offset': The end time of the speech segment.
            - 'speech': A boolean indicating whether the segment contains speech.
            - 'type': The type of the speech segment.
        Returns:
            pd.DataFrame: A DataFrame containing word-level annotations with the following columns:
            - 'onset': The start time of the word relative to the audio.
            - 'offset': The end time of the word relative to the audio.
            - 'label': The word label (text).
            - 'type': The type of the speech segment.
            - 'trialIndex': The index of the corresponding speech label in the input DataFrame.
            - 'trialLabel': The label of the corresponding speech segment in the input DataFrame.
            - 'wordIndex': The index of the word within its corresponding speech label.
        Raises:
            FileNotFoundError: If the TextGrid directory is not set or cannot be found.
        Notes:
            - The method assumes that TextGrid files are named in the format:
              "stim_<onset>_<offset>.TextGrid", where <onset> and <offset> are the
              start and end times of the speech segment.
            - Rows with empty or whitespace-only word labels are removed from the output.
        """
        if self.text_grid_dir is None:
            raise FileNotFoundError("TextGrid directory not found.")
        
        word_df = pd.DataFrame(columns=['onset', 'offset', 'label', 'type', 'trialIndex', 'trialLabel'])
        labels_df = labels_df.copy()
        # If the column stimSpeech is not present, create it as all False
        if 'stimSpeech' not in labels_df.columns:
            labels_df['stimSpeech'] = False
            labels_df['stimOnset'] = np.nan
            labels_df['stimOffset'] = np.nan
            labels_df['stimLabel'] = np.nan

        labels_df = labels_df[labels_df['speech'] | labels_df['stimSpeech']]

        labels_df['tgname'] = labels_df.apply(
            lambda row: f"stim_{row['onset']:.3f}_{row['offset']:.3f}.TextGrid", axis=1
        )
        labels_df['stimTgName'] = labels_df.apply(
            lambda row: f"stim_{row['stimOnset']:.3f}_{row['stimOffset']:.3f}.TextGrid", axis=1
        )

        for index, row in labels_df.iterrows():
            if 'stimOnset' not in labels_df.columns:
                entries = self._process_tg(row['tgname'], row['onset'], row['offset'], row['type'], row['trialIndex'], row['label'], 'words')
                word_df = pd.concat([word_df, pd.DataFrame(entries, columns=word_df.columns)], ignore_index=True)
            else:
                stim_entries = self._process_tg(row['stimTgName'], row['stimOnset'], row['stimOffset'], 'stim', row['trialIndex'], row['stimLabel'], 'words')
                entries = self._process_tg(row['tgname'], row['onset'], row['offset'], row['type'], row['trialIndex'], row['label'], 'words')
                word_df = pd.concat([word_df, pd.DataFrame(stim_entries, columns=word_df.columns), pd.DataFrame(entries, columns=word_df.columns)], ignore_index=True)

        # Remove any rows with empty word labels
        word_df = word_df[word_df['label'].isin(['', ' ']) == False]
        word_df = word_df[pd.notna(word_df['label'])]
        word_df = word_df.drop_duplicates(subset=['onset', 'offset', 'label']).reset_index(drop=True)

        word_df['wordIndex'] = (
            word_df
            .groupby(['trialIndex', 'type'])
            .cumcount()
        )
        word_df['wordIndex'] = word_df['wordIndex'].astype(int)
        return word_df
    
    def get_phoneme_df(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts phoneme-level information from TextGrid files and enriches it with additional features.
        Args:
            labels_df (pd.DataFrame): A DataFrame containing speech segment information with the following columns:
                - 'onset': The start time of the speech segment.
                - 'offset': The end time of the speech segment.
                - 'speech': A boolean indicating whether the segment contains speech.
                - 'type': The type of the segment (e.g., speech type).
        Returns:
            pd.DataFrame: A DataFrame containing phoneme-level information with the following columns:
                - 'onset': The start time of the phoneme (adjusted for the speech segment onset).
                - 'offset': The end time of the phoneme (adjusted for the speech segment onset).
                - 'label': The original phoneme label from the TextGrid.
                - 'type': The type of the speech segment.
                - 'trialIndex': The index of the speech segment in the input DataFrame.
                - 'trialLabel': The label of the speech segment in the input DataFrame.
                - 'stress': The stress level extracted from the phoneme label (if applicable).
                - 'phoneme': The phoneme label without stress markers.
                - 'vowel': A boolean indicating whether the phoneme is a vowel.
                - 'consonant': A boolean indicating whether the phoneme is a consonant.
                - Additional columns from the enriched vowel and consonant features.
        Raises:
            FileNotFoundError: If the TextGrid directory is not set or does not exist.
        Notes:
            - The method assumes that TextGrid files are named in the format:
              "stim_<onset>_<offset>.TextGrid".
            - Phoneme labels are enriched with additional features using the `enrich_features.get_phoneme_reference()` method.
            - Rows with empty or whitespace-only phoneme labels are removed.
        """
        if self.text_grid_dir is None:
            raise FileNotFoundError("TextGrid directory not found.")
        
        phoneme_df = pd.DataFrame(columns=['onset', 'offset', 'label', 'type', 'trialIndex', 'trialLabel'])
        
        labels_df = labels_df.copy()

        if 'stimSpeech' not in labels_df.columns:
            labels_df['stimSpeech'] = False
            labels_df['stimOnset'] = np.nan
            labels_df['stimOffset'] = np.nan
            labels_df['stimLabel'] = np.nan

        labels_df = labels_df[labels_df['speech'] | labels_df['stimSpeech']]

        labels_df['tgname'] = labels_df.apply(
            lambda row: f"stim_{row['onset']:.3f}_{row['offset']:.3f}.TextGrid", axis=1
        )
        labels_df['stimTgName'] = labels_df.apply(
            lambda row: f"stim_{row['stimOnset']:.3f}_{row['stimOffset']:.3f}.TextGrid", axis=1
        )

        for index, row in labels_df.iterrows():
            if 'stimOnset' not in labels_df.columns:
                entries = self._process_tg(row['tgname'], row['onset'], row['offset'], row['type'], row['trialIndex'], row['label'], 'phones')
                phoneme_df = pd.concat([phoneme_df, pd.DataFrame(entries, columns=phoneme_df.columns)], ignore_index=True)
            else:
                stim_entries = self._process_tg(row['stimTgName'], row['stimOnset'], row['stimOffset'], 'stim', row['trialIndex'], row['stimLabel'], 'phones')
                entries = self._process_tg(row['tgname'], row['onset'], row['offset'], row['type'], row['trialIndex'], row['label'], 'phones')
                phoneme_df = pd.concat([phoneme_df, pd.DataFrame(stim_entries, columns=phoneme_df.columns), pd.DataFrame(entries, columns=phoneme_df.columns)], ignore_index=True)

        # Remove any rows with empty phoneme labels
        phoneme_df = phoneme_df[phoneme_df['label'].isin(['', ' ']) == False]
        phoneme_df = phoneme_df[pd.notna(phoneme_df['label'])]
        phoneme_df = phoneme_df.drop_duplicates(subset=['onset', 'offset', 'label']).reset_index(drop=True)

        phoneme_df['stress'] = phoneme_df['label'].apply(lambda x: ''.join(filter(str.isdigit, x)) if any(char.isdigit() for char in x) else np.nan)
        phoneme_df['phoneme'] = phoneme_df['label'].apply(lambda x: ''.join(filter(str.isalpha, x)))
        phoneme_df['vowel'] = phoneme_df['phoneme'].apply(lambda x: x in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'])
        phoneme_df['consonant'] = phoneme_df['phoneme'].apply(lambda x: x in ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'])

        vowel_df, consonant_df = enrich_features.get_phoneme_reference()
        vowel_df['phoneme'] = vowel_df.index
        consonant_df['phoneme'] = consonant_df.index

        phoneme_df = phoneme_df.merge(vowel_df, how='left', left_on='phoneme', right_on='phoneme')
        phoneme_df = phoneme_df.merge(consonant_df, how='left', left_on='phoneme', right_on='phoneme')
        return phoneme_df

    def get_extended_phoneme_df(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate an extended phoneme DataFrame by enriching phoneme-level information 
        with word and syllable-level details.
        This method processes phoneme and word data to assign additional attributes 
        such as word indices, syllable indices, vowel counts, and phoneme positions 
        within words and syllables. It also performs syllabification for each word 
        to determine syllable boundaries.
        Args:
            labels_df (pd.DataFrame): A DataFrame containing label information, 
                including phoneme, word, and sentence-level details.
        Returns:
            pd.DataFrame: An enriched DataFrame containing phoneme-level information 
                with additional columns:
                - 'phonemeDFIndex': Index of the phoneme in the original DataFrame.
                - 'syllableDFIndex': Index of the syllable in the DataFrame.
                - 'wordDFIndex': Index of the word in the DataFrame.
                - 'word': The word associated with the phoneme.
                - 'syllable': The syllable associated with the phoneme.
                - 'wordIndex': Index of the word within the sentence.
                - 'syllableIndex': Index of the syllable within the word.
                - 'vowelCount': Cumulative count of vowels within the word.
                - 'phonemeSyllableIndex': Index of the phoneme within the syllable.
                - 'phonemeWordIndex': Index of the phoneme within the word.
        Notes:
            - The method assumes that the input DataFrame contains onset and offset 
              times for both phonemes and words.
            - Syllabification is performed using the `syllabify.syllabifyARPA` 
              function. If syllabification fails, the phonemes are treated as a 
              single syllable.
            - The method uses numpy and pandas operations for efficient processing.
        Raises:
            ValueError: If the input DataFrame does not contain required columns.
        """
        phoneme_df = self.get_phoneme_df(labels_df)
        word_df = self.get_word_df(labels_df)
        phoneme_df['phonemeDFIndex'] = phoneme_df.index

        phoneme_df = phoneme_df.reset_index(drop=False).rename(columns={phoneme_df.index.name or 'index': 'phoneme_index'})
        word_df = word_df.reset_index(drop=False).rename(columns={word_df.index.name or 'index': 'word_index'})

        phoneme_df['_key'] = 1
        word_df['_key'] = 1
        merged = phoneme_df.merge(
            word_df,
            on=['_key', 'trialIndex', 'type'],
            suffixes=('_phon', '_word')
        )
        match = merged[
            (merged['onset_word'] <= merged['onset_phon']) &
            (merged['offset_word'] >= merged['offset_phon'])
        ]
        index_map = match.groupby('phoneme_index')['word_index'].first()
        phoneme_df['wordDFIndex'] = phoneme_df['phoneme_index'].map(index_map).fillna(-1).astype(int)
        phoneme_df = phoneme_df.drop(columns=['_key', 'phoneme_index'])
        word_df = word_df.drop(columns=['_key', 'word_index'])

        phoneme_df['word'] = phoneme_df['wordDFIndex'].map(word_df['label'])
        phoneme_df['wordIndex'] = phoneme_df['wordDFIndex'].map(word_df['wordIndex'])
        
        # Count vowels per word
        vowel_mask = phoneme_df['vowel'] == True
        phoneme_df['vowelCount'] = vowel_mask.groupby(phoneme_df['wordDFIndex']).cumsum() 
        phoneme_df['vowelCount'] = phoneme_df['vowelCount'] - vowel_mask     
        
        # Group by word index
        for idx, group in phoneme_df.groupby('wordDFIndex'):
            phones = list(group['label'].values)
            trialIndex = group['trialIndex'].values[0]
            word = group['word'].values[0]
            sentence = labels_df.loc[trialIndex, 'label']

            try:
                syllables = syllabify.syllabifyARPA(phones)
            except:
                print(f'WARNING: Syllabification failed for {phones} in word {word}')
                print(f'Full sentence: {sentence}')
                print(f"Using {phones} as a singular syllable")
                syllables = phones

            phoneme_word_indices = group.index.values
            length_syllables = [len(syll.split(' ')) for syll in syllables]

            phoneme_df.loc[phoneme_word_indices, 'syllable'] = np.repeat(syllables, length_syllables)
            phoneme_df.loc[phoneme_word_indices, 'syllableIndex'] = np.concatenate([np.repeat(idx, length) for idx, length in enumerate(length_syllables)]).astype(int)
            phoneme_df.loc[phoneme_word_indices, 'phonemeIndexInSyllable'] = np.concatenate([np.arange(length) for length in length_syllables]).astype(int)
            phoneme_df.loc[phoneme_word_indices, 'phonemeIndexInWord'] = np.arange(len(group.index.values))

        # Everytime syllable changes, syllableDFIndex increments
        phoneme_df['syllableDFIndex'] = phoneme_df.groupby(['trialIndex', 'syllable']).ngroup()

        phoneme_df['syllableDFIndex'] = phoneme_df['syllableDFIndex'].astype(int)
        phoneme_df['syllableIndex'] = phoneme_df['syllableIndex'].astype(int)
        phoneme_df['phonemeIndexInSyllable'] = phoneme_df['phonemeIndexInSyllable'].astype(int)
        phoneme_df['phonemeIndexInWord'] = phoneme_df['phonemeIndexInWord'].astype(int)

        return phoneme_df

    def get_syllable_df(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame containing phoneme labels to generate a syllable-level DataFrame.

        Args:
            labels_df (pd.DataFrame): A DataFrame containing phoneme-level information, 
                which includes columns required for syllable aggregation.

        Returns:
            pd.DataFrame: A DataFrame aggregated at the syllable level with the following columns:
                - 'onset': The minimum onset time of the syllable.
                - 'offset': The maximum offset time of the syllable.
                - 'label': The syllable label (renamed from 'syllable').
                - 'type': The type of the syllable (e.g., stressed, unstressed).
                - 'trialIndex': The index of the label in the original DataFrame.
                - 'trialLabel': The label of the trial in the original DataFrame.
                - 'syllableIndex': The index of the syllable in the sequence.

        Notes:
            - The function first extends the phoneme DataFrame using `get_extended_phoneme_df`.
            - It groups the data by 'syllableDFIndex' to aggregate syllable-level information.
            - The index of the resulting DataFrame is reset and then reassigned to match the original grouping.
        """
        extended_phoneme_df = self.get_extended_phoneme_df(labels_df).copy()
        syllable_df = extended_phoneme_df[['onset', 'offset', 'syllable', 'type', 'trialIndex', 'trialLabel', 'syllableDFIndex', 'syllableIndex']]
        syllable_df = syllable_df.groupby('syllableDFIndex').agg({
            'onset': 'min',
            'offset': 'max',
            'syllable': 'first',
            'type': 'first',
            'trialIndex': 'first',
            'trialLabel': 'first',
            'syllableIndex': 'first'
        })
        syllable_df.rename(columns={'syllable': 'label'}, inplace=True)
        index_values = syllable_df.index.values
        syllable_df.reset_index(drop=True, inplace=True)
        syllable_df.index = index_values
        return syllable_df
    
    def get_consonantOnset(self, labels_df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the minimum onset times of consonant phonemes from the provided labels DataFrame.

        Args:
            labels_df (pd.DataFrame): A DataFrame containing phoneme labels and their associated metadata.

        Returns:
            np.ndarray: An array of minimum onset times for consonant phonemes, grouped by label index.
        """
        phoneme_df = self.get_phoneme_df(labels_df)
        return phoneme_df[phoneme_df['consonant']].groupby('trialIndex')['onset'].min().values
    
    def get_vowelOnset(self, labels_df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the minimum onset times of vowel phonemes from the provided labels DataFrame.

        Args:
            labels_df (pd.DataFrame): A DataFrame containing phoneme labels and their associated metadata.

        Returns:
            np.ndarray: An array of minimum onset times for vowel phonemes, grouped by label index.
        """
        phoneme_df = self.get_phoneme_df(labels_df)
        return phoneme_df[phoneme_df['vowel']].groupby('trialIndex')['onset'].min().values

    def make_stim_cue_cols(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates stimulus and cue columns in the DataFrame and removes them as rows.

        Args:
            labels_df (pd.DataFrame): The input DataFrame containing task-related data.

        Returns:
            pd.DataFrame: The modified DataFrame with stimulus and cue columns added.
        """
        task_df = labels_df.copy().sort_values(by='onset').reset_index(drop=True)
        # Get all stim and prod event indices
        # Get all the tasks and do this for each task
        tasks = task_df['task'].unique()
        results = []

        for task in tasks:
            df = task_df[task_df['task'] == task].copy()
            stim_indices = df[df['type'] == 'stim'].index.tolist()
            prod_indices = df[df['type'] == 'prod'].index.tolist()
            
            if len(stim_indices) != 0:
                for i, stim_idx in enumerate(stim_indices):
                    stim_row = df.loc[stim_idx]
                    stim_onset = stim_row['onset']
                    stim_offset = stim_row['offset']

                    # Determine the range of this stim block
                    if i < len(stim_indices) - 1:
                        next_stim_onset = df.loc[stim_indices[i + 1], 'onset']
                        block_df = df[(df['onset'] > stim_onset) & (df['onset'] < next_stim_onset)]
                    else:
                        block_df = df[df['onset'] > stim_onset]
                    
                    cues = block_df[block_df['type'] == 'cues']
                    prods = block_df[block_df['type'] == 'prod']
                    
                    cue_annotations = {}
                    for _, cue_row in cues.iterrows():
                        cue_label = cue_row['label']
                        cue_annotations[f'{cue_label}Onset'] = cue_row['onset']
                        cue_annotations[f'{cue_label}Offset'] = cue_row['offset']

                    if not prods.empty:
                        for _, prod_row in prods.iterrows():
                            annotated = prod_row.copy()
                            annotated['stimOnset'] = stim_onset
                            annotated['stimOffset'] = stim_offset
                            annotated['stimLabel'] = stim_row['label']
                            annotated['stimSpeech'] = stim_row['speech']
                            annotated['stimID'] = stim_row['stimID']
                            for key, val in cue_annotations.items():
                                annotated[key] = val
                            results.append(annotated)
                    else:
                        # Create a nan for onset, offset, label, but use 'prod' as type and keep the task name
                        annotated = stim_row.copy()
                        annotated['onset'] = np.nan
                        annotated['offset'] = np.nan
                        annotated['label'] = np.nan
                        annotated['type'] = 'prod'
                        annotated['stimOnset'] = stim_onset
                        annotated['stimOffset'] = stim_offset
                        annotated['stimLabel'] = stim_row['label']
                        annotated['stimSpeech'] = stim_row['speech']
                        annotated['stimID'] = stim_row['stimID']
                        for key, val in cue_annotations.items():
                            annotated[key] = val
                        results.append(annotated)

            elif len(prod_indices) != 0:
                for i, prod_idx in enumerate(prod_indices):
                    prod_row = df.loc[prod_idx]
                    prod_onset = prod_row['onset']
                    prod_offset = prod_row['offset']

                    # Determine the range of this stim block
                    if i < len(prod_indices) - 1:
                        next_prod_onset = df.loc[prod_indices[i + 1], 'onset']
                        block_df = df[(df['onset'] > prod_onset) & (df['onset'] < next_prod_onset)]
                    else:
                        block_df = df[df['onset'] > prod_onset]
                    
                    cues = block_df[block_df['type'] == 'cues']
                    
                    cue_annotations = {}
                    for _, cue_row in cues.iterrows():
                        cue_label = cue_row['label']
                        cue_annotations[f'{cue_label}Onset'] = cue_row['onset']
                        cue_annotations[f'{cue_label}Offset'] = cue_row['offset']

                    annotated = prod_row.copy()
                    annotated['stimOnset'] = np.nan
                    annotated['stimOffset'] = np.nan
                    annotated['stimLabel'] = np.nan
                    annotated['stimSpeech'] = False
                    annotated['stimID'] = np.nan
                    for key, val in cue_annotations.items():
                        annotated[key] = val
                    results.append(annotated)
            else:
                raise ValueError("No stim or prod events found in the DataFrame.")

        output = pd.DataFrame(results)
        stim_cols = ['stimOnset', 'stimOffset', 'stimLabel', 'stimSpeech', 'stimID'][::-1]
        # move the stim columns to the front
        for col in stim_cols:
            if col in output.columns:
                output.insert(0, col, output.pop(col))

        return output

    def get_task_df(self, task_name: str = None, enrich: bool = True, make_stim_cue_cols: bool = True) -> pd.DataFrame:
        """
        Retrieves the task table as a pandas DataFrame, optionally filtered by a specific task name 
        and enriched with additional information.
        Args:
            task_name (str, optional): The name of the task to filter the task table. 
                                       If None, the entire task table is returned. Defaults to None.
            enrich (bool, optional): Whether to enrich the resulting DataFrame with additional 
                                     information. Defaults to True.
            make_stim_cue_cols (bool, optional): Whether to create stimulus and cue columns
                                                  in the DataFrame and remove them as rows. Defaults to True.
        Returns:
            pd.DataFrame: A DataFrame containing the task table, optionally filtered and enriched.
        Raises:
            ValueError: If the specified task_name is not found in the task names.
        
        """
        if task_name is None:
            out = self.task_table
        else:
            if task_name not in self.task_names:
                raise ValueError(f"Task name {task_name} not found.")
            out = self.task_table[self.task_table['task'] == task_name]
        if make_stim_cue_cols:
            out = self.make_stim_cue_cols(out)
        
        # Step 1: Create a key only for rows where both stimOnset and stimOffset are not NaN
        mask = (~out['stimOnset'].isna()) & (~out['stimOffset'].isna())
        stim_keys = pd.Series(np.where(mask, 
            out['stimOnset'].astype(str) + "_" + out['stimOffset'].astype(str),
            np.nan
        ))

        trial_indices = []
        key_to_index = {}
        next_index = 0

        # Step 3: Iterate over rows to preserve order and assign trialIndex
        for key in stim_keys:
            if pd.isna(key):
                # assign a new unique trialIndex for NaN
                trial_indices.append(next_index)
                next_index += 1
            else:
                if key not in key_to_index:
                    key_to_index[key] = next_index
                    next_index += 1
                trial_indices.append(key_to_index[key])

        # Step 4: Assign to DataFrame
        out['trialIndex'] = trial_indices
        out['trialLabel'] = out['label']
        
        if enrich:
            out = self.enrich_task_df(out, task_name)
        out.reset_index(drop=True, inplace=True)
        return out
    
    def enrich_task_df(self, table: pd.DataFrame, task_name: str) -> pd.DataFrame:
        """
        Enriches the given DataFrame with additional task-specific information.

        Parameters:
        -----------
        table : pd.DataFrame
            The input DataFrame to be enriched.
        task_name : str
            The name of the task for which the DataFrame should be enriched. 
            If None, the input DataFrame is returned as is.

        Returns:
        --------
        pd.DataFrame
            The enriched DataFrame with additional task-specific information.

        Raises:
        -------
        ValueError
            If the provided task_name is not found in the list of task names.

        Notes:
        ------
        - For tasks 'custom_cv' and 'cv', the method merges the input DataFrame 
          with a CV reference table and attempts to compute the 'vowelOnset' column.
        - If the 'vowelOnset' computation fails due to a missing file, a warning 
          is printed, and the 'vowelOnset' column is filled with NaN values.
        """
        if task_name is None:
            return table
        elif task_name not in self.task_names:
            raise ValueError(f"Task name {task_name} not found.")
        if task_name in ['custom_cv', 'cv']:
            cv_reference = enrich_features.get_cv_reference()
            table = table.merge(cv_reference, left_on='label', right_index=True, how='left')
            try:
                table['vowelOnset'] = self.get_vowelOnset(table)
            except FileNotFoundError:
                print(f"WARNING: Vowel onset not found for {task_name}.")
                table['vowelOnset'] = np.nan
        return table
    
    def add_audio_features(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Adds audio features to the given DataFrame.

        This method processes audio data based on the 'onset' and 'offset' columns 
        in the input DataFrame and computes various audio features such as mel 
        spectrograms, pitch, and loudness. The computed features are added as new 
        columns to the original DataFrame.

        Args:
            table (pd.DataFrame): A DataFrame containing at least the following columns:
                - 'speech': A boolean column indicating rows with speech data.
                - 'onset': The start times of the audio segments.
                - 'offset': The end times of the audio segments.

        Returns:
            tuple:
                - pd.DataFrame: The input DataFrame with additional columns:
                    - 'audio': The extracted audio segments.
                    - 'mel_spectrogram': The mel spectrograms of the audio segments.
                    - 'pitch': The pitch values of the audio segments.
                    - 'loudness': The loudness values of the audio segments.
                - int: The sample rate of the audio data.
        """
        new_table = table.copy()
        new_table = new_table[new_table['speech']]
        t0 = new_table['onset'].values
        t1 = new_table['offset'].values

        audio_segments, sr = self.get_audio_clips(t0, t1)
        mel_spectrograms = self.get_mel_spectrogram(audio_segments, sr)
        pitches = self.get_pitch(audio_segments, sr)
        loudness = self.get_loudness(audio_segments, sr)

        new_table['audio'] = audio_segments
        new_table['mel_spectrogram'] = mel_spectrograms
        new_table['pitch'] = pitches
        new_table['loudness'] = loudness
        # Merge this new_table with the original table on the index
        table = table.merge(new_table[['audio', 'mel_spectrogram', 'pitch', 'loudness']], left_index=True, right_index=True, how='left')
        return table, sr

    def calculate_waveforms(self, ks_dir: Path, n_spikes: int = None, n_samples: int = 80, n_channels: int = 50, n_jobs: int = 8, use_memmap: bool = True) -> None:
        """
        In the ks_dir, calculate the waveforms for each of the clusters, and create an array of shape (len(spike_clusters), n_samples, n_channels)

        """
        spike_indices = np.load(ks_dir / "spike_times.npy").squeeze()
        spike_clusters = np.load(ks_dir / "spike_clusters.npy").squeeze()
        templates = np.load(ks_dir / "templates.npy")
        channel_map = np.load(ks_dir / "channel_map.npy").squeeze()
        channel_positions = np.load(ks_dir / "channel_positions.npy")

        temp_wh = extract_waveform.get_temp_wh(ks_dir, use_memmap=use_memmap)
        t0 = extract_waveform.get_tmin(ks_dir)
        sample_rate = extract_waveform.get_sample_rate(ks_dir)

        shared_args = {
            'ks_folder': ks_dir,
            'n_samples': n_samples,
            'n_channels': n_channels,
            'spike_times': spike_indices,
            'spike_clusters': spike_clusters,
            'templates': templates,
            'temp_wh': temp_wh,
            'fs': sample_rate,
            't0': t0,
            'channel_positions': channel_positions,
            'channel_map': channel_map,
            'n_spikes': n_spikes
        }

        unique_clusters = np.unique(spike_clusters)

        if n_jobs > 0:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(extract_waveform.read_whitened_waveforms)(clus_id=clus_id, **shared_args)
                for clus_id in tqdm(unique_clusters)
            )
        else:
            results = [
                extract_waveform.read_whitened_waveforms(clus_id=clus_id, **shared_args)
                for clus_id in tqdm(unique_clusters)
            ]
        
        # clear temp_wh from memory
        del temp_wh

        cluster_indices = [np.where(spike_clusters == clus_id) for clus_id in unique_clusters]

        waveforms = np.empty((len(spike_clusters), n_samples, n_channels), dtype=np.float32)
        for i, result in enumerate(results):
            if result is not None:
                waveforms[cluster_indices[i], :, :] = result
            else:
                # If no waveform was found for this cluster, fill with NaN
                waveforms[cluster_indices[i], :, :] = np.nan

        # Save the waveforms to a .npy file called 'waveforms.npy' in the ks_dir
        np.save(ks_dir / "waveforms.npy", waveforms)

    def get_neuron_df(self, recalculate: bool = False, bin_size: float = 0.010, add_cluster_metadata = False, calculate_waveforms: bool = False, 
                      n_spikes: int = None, n_samples: int = 80, n_channels: int = 10, n_jobs: int = 8) -> pd.DataFrame:
        """
        Generate and return a DataFrame containing neuron-related data.

        This method processes spike data from multiple directories, aggregates it,
        and computes additional information such as per-neuron PSTHs (Peri-Stimulus
        Time Histograms) and spike times.

        Args:
            recalculate (bool): If True, forces recalculation of the neuron DataFrame.
                                If False, returns the cached DataFrame if available.
            bin_size (float): The size of the time bins for PSTH computation (in seconds).
                                Default is 0.010 (10 ms).
            calculate_waveforms (bool): If True, computes the waveforms for each neuron.
            n_samples (int): Number of samples to extract for each waveform.
            n_channels (int): Number of channels to consider for waveform extraction.

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - 'neuron': Unique identifier for each neuron.
                - 'probe': Mean probe number associated with the neuron.
                - 'depth': Median depth of the neuron.
                - 'spike_time': Sorted array of spike times for the neuron.
                - 'psth': Peri-Stimulus Time Histogram (PSTH) for the neuron.
                - 'time': Time bins corresponding to the PSTH.

        Notes:
            - The method assumes the presence of specific files in each directory
              (e.g., "spike_times.npy", "spike_clusters.npy", "spike_positions.npy",
              "amplitudes.npy", and optionally "ops.npy").
            - If "ops.npy" is not found, a default sample rate of 30,000 Hz is used.
            - The neuron identifier is constructed using a prefix derived from
              `self.recid`, the probe number, and the spike cluster ID.
            - PSTHs are computed using a Gaussian kernel with a sigma of 1.

        Raises:
            AttributeError: If `self.ks_dirs` or `self.recid` is not properly set.
            FileNotFoundError: If required files are missing in the directories.
        """
        if hasattr(self, 'neuron_df') and self.neuron_df is not None and not recalculate:
            return self.neuron_df.copy()
        
        df = pd.DataFrame(columns=['spike_time', 'spike_cluster', 'spike_amplitude', 'depth', 'probe'])
        for ks_dir in self.ks_dirs:
            # if ops.npy exists, find the sample rate from there
            sample_rate = extract_waveform.get_fs(ks_dir)

            spike_indices = np.load(ks_dir / "spike_times.npy").squeeze()
            spike_times = spike_indices / sample_rate  # Convert spike indices to seconds
            spike_clusters = np.load(ks_dir / "spike_clusters.npy").squeeze()
            templates = np.load(ks_dir / "templates.npy")
            channel_map = np.load(ks_dir / "channel_map.npy").squeeze()
            channel_positions = np.load(ks_dir / "channel_positions.npy")
            
            # Try loading spike positions, if it fails, try loading spike_centroids.npy
            if (ks_dir / "spike_positions.npy").exists():
                depths = np.load(ks_dir / "spike_positions.npy")
            elif (ks_dir / "spike_centroids.npy").exists():
                depths = np.load(ks_dir / "spike_centroids.npy")
            elif (ks_dir / "rez.mat").exists():
                print("Warning: No depth information calculated. Using estimated depths from channel positions.")
                rez = extract_waveform.read_rezfile(ks_dir)
                iTemp = np.load(ks_dir / "spike_templates.npy").squeeze() # Load spike templates
                iChan = (rez['iNeighPC'][iTemp, 15] - 1).astype(int)  # Get the channel index for the 16th neighbor
                xcoords = rez['xcoords'][iChan]
                ycoords = rez['ycoords'][iChan] # Calculate depths from the ycoords of the channels
                depths = np.column_stack((xcoords, ycoords))
            else:
                raise FileNotFoundError(f"Neither spike_positions.npy nor spike_centroids.npy found in {ks_dir}.")

            depths = depths[:, 1] # Assuming depth is in the second column
            amplitudes = np.load(ks_dir / "amplitudes.npy").flatten()
            probe = int(re.search(r"imec(\d+)", str(ks_dir)).group(1))

            # Create spike-level DataFrame
            spike_df = pd.DataFrame({
                'spike_time': spike_times,
                'spike_cluster': spike_clusters,
                'spike_amplitude': amplitudes,
                'depth': depths,
                'probe': probe
            })

            if add_cluster_metadata:
                cluster_group = pd.read_csv(ks_dir / "cluster_group.tsv", sep='\t')
                cluster_contam = pd.read_csv(ks_dir / "cluster_ContamPct.tsv", sep='\t')
                cluster_amplitude = pd.read_csv(ks_dir / "cluster_Amplitude.tsv", sep='\t')

                # Merge all cluster-level metadata on 'cluster_id'
                cluster_metadata = cluster_group.merge(cluster_contam, on='cluster_id') \
                                                .merge(cluster_amplitude, on='cluster_id')

                

                spike_df = spike_df.merge(cluster_metadata, left_on='spike_cluster', right_on='cluster_id', how='left')
                # Optional: drop 'cluster_id' if it's redundant
                spike_df = spike_df.drop(columns='cluster_id')

            if calculate_waveforms:
                unique_clusters = np.unique(spike_clusters)
                if ks_dir / "waveforms.npy" in ks_dir.iterdir():
                    waveforms = np.load(ks_dir / "waveforms.npy")
                    results = [waveforms[spike_clusters == clus_id] for clus_id in unique_clusters]
                    print("Using precomputed waveforms from waveforms.npy")
                else:
                    temp_wh = extract_waveform.get_temp_wh(ks_dir)
                    t0 = extract_waveform.get_tmin(ks_dir)
                    shared_args = {
                        'ks_folder': ks_dir,
                        'n_samples': n_samples,
                        'n_channels': n_channels,
                        'spike_times': spike_indices,
                        'spike_clusters': spike_clusters,
                        'templates': templates,
                        'temp_wh': temp_wh,
                        'fs': sample_rate,
                        't0': t0,
                        'channel_positions': channel_positions,
                        'channel_map': channel_map,
                        'n_spikes': n_spikes
                    }
                    if n_jobs > 0:
                        from joblib import Parallel, delayed

                        results = Parallel(n_jobs=n_jobs)(
                            delayed(extract_waveform.read_whitened_waveforms)(clus_id=clus_id, **shared_args)
                            for clus_id in tqdm(unique_clusters)
                        )
                    else:
                        results = [
                            extract_waveform.read_whitened_waveforms(clus_id=clus_id, **shared_args)
                            for clus_id in tqdm(unique_clusters)
                        ]
                        
                for i, clus_id in enumerate(unique_clusters):
                    idx = spike_df.index[spike_df['spike_cluster'] == clus_id][0]
                    spike_df.at[idx, 'waveformMed'] = np.nanmedian(results[i], axis=0)
                    spike_df.at[idx, 'waveformStd'] = np.nanstd(results[i], axis=0)

            # Append to main df
            df = pd.concat([df, spike_df], ignore_index=True)

        unit_prefix = f"u{self.recid.split('_')[0][2:]}{int(self.recid.split('_')[1][1:]):02}"
        df['neuron'] = unit_prefix + df['probe'].astype(str) + df['spike_cluster'].astype(int).apply(lambda x: f"{x:04d}")
        df['spike_time'] = df['spike_time'].astype(float)
        agg_dict = {}

        for col in df.columns:
            if col in ['neuron', 'spike_time']:
                continue
            if col == 'depth':
                agg_dict[col] = 'median'
            # Check if the column is numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'

        neuron_df = df.groupby('neuron').agg(
            agg_dict
        ).join(
            df.groupby('neuron')['spike_time'].apply(lambda x: np.sort(x)).rename('spike_time')
        )

        time_start = 0
        time_end = df['spike_time'].max() + 1
        bins = np.arange(time_start, time_end + bin_size, bin_size)
        time = (bins[:-1] + bins[1:]) / 2  # Center of each bin

        # Compute a PSTH for each neuron
        smoothed_psths = []
        for spike_array in neuron_df['spike_time'].tolist():
            smoothed_psths.append(compute_psth(spike_array, bins))
        neuron_df['psth'] = smoothed_psths
        
        neuron_df['time'] = [time for _ in range(len(neuron_df))]
        neuron_df['num_spikes'] = neuron_df['spike_time'].apply(len)

        # # Upsample the PSTH to 1000 Hz
        # new_bin_size = 0.001  # 1 ms in seconds
        # new_bins = np.arange(time_start, time_end + new_bin_size, new_bin_size)
        # upsampled_time = (new_bins[:-1] + new_bins[1:]) / 2  # Center of each bin
        
        # psth_array = np.stack(neuron_df['psth'].values)  # shape: (n_neurons, len(time))
        # interpolator = interp1d(time, psth_array, axis=1, kind='linear', fill_value='extrapolate')
        # psth_upsampled = interpolator(upsampled_time)  # shape: (n_neurons, len(new_bins))
        # psth_upsampled = np.clip(psth_upsampled, 0, np.max(psth_upsampled))
        # neuron_df['psth'] = list(psth_upsampled)

        # neuron_df['time'] = [upsampled_time for _ in range(len(neuron_df))]

        neuron_df.reset_index(inplace=True)
        self.neuron_df = neuron_df

        return neuron_df.copy()
    
    def get_trial_indices(self, labels_df: pd.DataFrame, input_value: Optional[Union[int, List, np.ndarray, str]] = None) -> np.ndarray:
        """
        Retrieve trial indices based on the input value.

        This method returns trial indices from a DataFrame based on the type of the input value.
        The input can be a single integer, a list/array of integers, or a string representing
        a stimulus label.

        Args:
            
            labels_df (pd.DataFrame): A pandas DataFrame containing a 'label' column. This is used
                to match the stimulus text when the input is a string.
            input_value (Optional[Union[int, List, np.ndarray, str]]): The input used to retrieve trial indices.
                - If an integer, it is treated as a single index.
                - If a list or numpy array, it is treated as a collection of indices.
                - If a string, it is treated as a stimulus label to match in the DataFrame.
                - If None, all trial indices are returned.

        Returns:
            np.ndarray: An array of trial indices corresponding to the input value.

        Raises:
            ValueError: If the input value is not an integer, a list of integers, or a string.
        """
        if isinstance(input_value, int):  # Single index
            return np.atleast_1d(input_value)
        elif isinstance(input_value, (list, np.ndarray)):  # List of indices
            return np.array(input_value, dtype=int)
        elif isinstance(input_value, str):  # Stimulus text
            return np.array(labels_df[labels_df['label'] == input_value].index.tolist(), dtype=int)
        elif input_value is None:  # No input
            return np.array(labels_df.index.tolist(), dtype=int)
        else:
            raise ValueError("Input must be an integer, a list of integers, or a string.")
    
    def get_trial_psth(self, labels_df: pd.DataFrame, 
                       input_value: Union[int, List[int], np.ndarray, str] = None, nanpad: float = 0.5,
                       n_trials_back: int = 1, align_col: str = 'onset', neuron: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate peri-stimulus time histograms (PSTHs) for specified trials.
        This function computes PSTHs for neural data aligned to specific events 
        in the trial data. It also computes pre- and post-event PSTHs, with 
        optional padding and alignment.
        Args:
            
            labels_df (pd.DataFrame): 
                DataFrame containing trial metadata, including onset, offset, and 
                alignment columns.
            input_value (Optional[Union[int, List[int], np.ndarray, str]]): 
                Specifies the trials to include. Can be an integer, list of integers, 
                numpy array, a string identifier, or None for all trials.
            nanpad (float, optional): 
                Padding value (in seconds) to extend the intervals before the first 
                trial and after the last trial. Defaults to 0.5.
            n_trials_back (int, optional): 
                Number of trials to look back or forward for determining pre- and 
                post-event intervals. Defaults to 1.
            align_col (str, optional): 
                Column name in `labels_df` to use for alignment. Defaults to 'onset'.
            neuron (str, optional):
                Neuron identifier to filter the neuron DataFrame. If None, all neurons are used.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - psths: 3D array of PSTHs for the main event interval.
                - pre_psths: 3D array of PSTHs for the pre-event interval.
                - post_psths: 3D array of PSTHs for the post-event interval.
                - psth_timings: 1D array of time bins used for the PSTHs.
        Notes:
            - The function assumes that the `neuron_df` DataFrame contains a 'time' 
                column with time bins and a 'psth' column with neural activity data.
            - Interpolation is used to align the neural data to the specified time bins.
            - NaN padding is applied to ensure consistent array shapes across trials.
        """
        neuron_df = self.get_neuron_df().copy()
        if neuron is not None:
            neuron_df = neuron_df[neuron_df['neuron'] == neuron]
            if len(neuron_df) == 0:
                raise ValueError(f"Neuron {neuron} not found in the neuron DataFrame.")
        trial_indices = self.get_trial_indices(labels_df, input_value)
        time_intervals = np.stack(labels_df.loc[trial_indices][['onset', 'offset']].values)
        align_values = np.array(labels_df.loc[trial_indices][align_col].values, dtype=np.float32)
        numerical_indices = np.array(labels_df.loc[trial_indices].index.tolist(), dtype=int)
        prevEventOffs = np.array([labels_df.iloc[i - n_trials_back]['offset'] if i > n_trials_back - 1 and (n_trials_back >= 1) else labels_df.iloc[0]['onset'] - nanpad for i in numerical_indices])
        nextEventOns = np.array([labels_df.iloc[i + n_trials_back]['onset'] if i < len(labels_df) - n_trials_back and (n_trials_back >= 1) else labels_df.iloc[-1]['offset'] + nanpad for i in numerical_indices])
        prevEventOffs[np.isnan(prevEventOffs)] = time_intervals[np.isnan(prevEventOffs), 0]
        nextEventOns[np.isnan(nextEventOns)] = time_intervals[np.isnan(nextEventOns), 1]

        psth_intervals = np.array(list(zip(prevEventOffs, nextEventOns)))
        time = np.array(neuron_df['time'].values[0], dtype=np.float32)  
        bin_size = time[1] - time[0]

        min_time = np.nanmin(psth_intervals[:, 0] - align_values)
        max_time = np.nanmax(psth_intervals[:, 1] - align_values)
        # Round to the nearest bin
        min_time = np.floor(min_time / bin_size) * bin_size
        max_time = np.ceil(max_time / bin_size) * bin_size
        total_bins = int((max_time - min_time) / bin_size) + 1
        psth_timings = np.arange(min_time, max_time + bin_size, bin_size)

        psths = []
        pre_psths = []
        post_psths = []
        for i in range(len(psth_intervals)):
            prodOn, prodOff = time_intervals[i]
            align = align_values[i]
            t0, t1 = psth_intervals[i]

            # Indices in raw time
            pre_start_idx = np.searchsorted(time, t0, side='left')
            prod_start_idx = np.searchsorted(time, prodOn, side='left')
            prod_end_idx = np.searchsorted(time, prodOff, side='right')
            post_end_idx = np.searchsorted(time, t1, side='right')

            # Actual time values for these segments
            pre_time = time[pre_start_idx:prod_start_idx] - align
            interval_time = time[pre_start_idx:post_end_idx] - align
            post_time = time[prod_end_idx:post_end_idx] - align

            # Raw data arrays
            pre = np.stack(neuron_df['psth'].apply(lambda x: x[pre_start_idx:prod_start_idx]).values)
            interval = np.stack(neuron_df['psth'].apply(lambda x: x[pre_start_idx:post_end_idx]).values)
            post = np.stack(neuron_df['psth'].apply(lambda x: x[prod_end_idx:post_end_idx]).values)

            # Full arrays (nan-padded)
            pre_full = np.full((pre.shape[0], total_bins), np.nan)
            interval_full = np.full((interval.shape[0], total_bins), np.nan)
            post_full = np.full((post.shape[0], total_bins), np.nan)

            # Target time slices within the full psth_timings
            pre_mask = (psth_timings >= pre_time[0]) & (psth_timings < pre_time[-1])
            interval_mask = (psth_timings >= interval_time[0]) & (psth_timings < interval_time[-1])
            post_mask = (psth_timings >= post_time[0]) & (psth_timings < post_time[-1])

            pre_full[:, pre_mask] = interp1d(pre_time, pre, axis=1, fill_value="extrapolate")(psth_timings[pre_mask])
            interval_full[:, interval_mask] = interp1d(interval_time, interval, axis=1, fill_value="extrapolate")(psth_timings[interval_mask])
            post_full[:, post_mask] = interp1d(post_time, post, axis=1, fill_value="extrapolate")(psth_timings[post_mask])

            pre_psths.append(pre_full)
            psths.append(interval_full)
            post_psths.append(post_full)

        psths = np.stack(psths)
        pre_psths = np.stack(pre_psths)
        post_psths = np.stack(post_psths)

        return psths, pre_psths, post_psths, psth_timings

    def get_average_psth(self, labels_df: pd.DataFrame, 
                         input_value: Optional[Union[int, List[int], np.ndarray, str]] = None, nanpad: float = 0.5, 
                         n_trials_back: int = 1, align_col: str = 'onset', neuron: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the average Peri-Stimulus Time Histogram (PSTH) along with its 
        standard error of the mean (SEM), timing information, and baseline values.
        Parameters:
            labels_df (pd.DataFrame): DataFrame containing trial labels and timing 
                information. Must include the column specified by `align_col`.
            input_value (Optional[Union[int, List[int], np.ndarray, str]]): Identifier(s) or 
                condition(s) for which the PSTH is to be computed. Can be a single 
                value, a list of values, a numpy array, a string, or None for all trials.
            nanpad (float, optional): Padding value for handling NaN values in the 
                PSTH computation. Default is 0.5.
            n_trials_back (int, optional): Number of trials to consider for 
                computing the PSTH. Default is 1.
            align_col (str, optional): Column name in `labels_df` used for aligning 
                the PSTH. Default is 'onset'.
            neuron (str, optional): Neuron identifier to filter the neuron DataFrame.
                If None, all neurons are used.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - avg_psth: The average PSTH across trials.
                - sem_psth: The standard error of the mean of the PSTH.
                - psth_timings: Timing information corresponding to the PSTH.
                - baselines_pre: Baseline values computed before the alignment point.
                - baselines_post: Baseline values computed after the alignment point.
            Notes:
                - If you include all trials, the average PSTH will be computed across all trials, not each condition.
        """
        # Create an empty array to hold the average PSTH
        # Find the maximum length of the time intervals and add padding
        
        psths, pre_psths, post_psths, psth_timings = self.get_trial_psth(labels_df, input_value, nanpad, n_trials_back, align_col, neuron)
        if len(psths) > 8:
            print(f"WARNING: {len(psths)} trials found for {input_value}. Average will be computed across all {len(psths)} trials.")
        avg_psth = np.nanmean(psths, axis=0)
        sem_psth = np.nanstd(psths, axis=0) / np.sqrt(psths.shape[0])

        baselines_pre = np.nanmean(pre_psths, axis=0)
        baselines_post = np.nanmean(post_psths, axis=0)

        return avg_psth, sem_psth, psth_timings, baselines_pre, baselines_post

    def get_trial_spikes(self, labels_df: pd.DataFrame, 
                         input_value: Optional[Union[int, List[int], np.ndarray, str]] = None, nanpad: float = 0.5,
                         n_trials_back: int = 1, align_col: str = 'onset', neuron: str = None,
                         sample_rate: float = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spike time arrays for specified trials. 
        This function computes spike times aligned to specific events in the trial data.
        It also computes pre- and post-event spike times, with optional padding and alignment.
        Args:
            labels_df (pd.DataFrame): 
                DataFrame containing trial metadata, including onset, offset, and 
                alignment columns.
            input_value (Optional[Union[int, List[int], np.ndarray, str]]): 
                Specifies the trials to include. Can be an integer, list of integers, 
                numpy array, a string identifier, or None for all trials.
            nanpad (float, optional): 
                Padding value (in seconds) to extend the intervals before the first 
                trial and after the last trial. Defaults to 0.5.
            n_trials_back (int, optional): 
                Number of trials to look back or forward for determining pre- and 
                post-event intervals. Defaults to 1.
            align_col (str, optional): 
                Column name in `labels_df` to use for alignment. Defaults to 'onset'.
            neuron (str, optional):
                Neuron identifier to filter the neuron DataFrame. If None, all neurons are used.
            sample_rate (float, optional):
                Sample rate for the spike times. Default is 1000 Hz.
        Returns:
            np.ndarray: A 3D array of spike times for the specified trials, 
                aligned to the specified event.
            np.ndarray: A 1D array of time bins used for the spike times.
        """        
        neuron_df = self.get_neuron_df().copy()
        if neuron is not None:
            neuron_df = neuron_df[neuron_df['neuron'] == neuron]
            if len(neuron_df) == 0:
                raise ValueError(f"Neuron {neuron} not found in the neuron DataFrame.")
        trial_indices = self.get_trial_indices(labels_df, input_value)
        align_values = np.array(labels_df.loc[trial_indices][align_col].values, dtype=np.float32)
        time_intervals = np.stack(labels_df.loc[trial_indices][['onset', 'offset']].values)
        numerical_indices = np.array(labels_df.loc[trial_indices].index.tolist(), dtype=int)
        prevEventOffs = np.array([labels_df.iloc[i - n_trials_back]['offset'] if i > n_trials_back - 1 and (n_trials_back >= 1) else labels_df.iloc[0]['onset'] - nanpad for i in numerical_indices])
        nextEventOns = np.array([labels_df.iloc[i + n_trials_back]['onset'] if i < len(labels_df) - n_trials_back and (n_trials_back >= 1) else labels_df.iloc[-1]['offset'] + nanpad for i in numerical_indices])
        prevEventOffs[np.isnan(prevEventOffs)] = time_intervals[np.isnan(prevEventOffs), 0]
        nextEventOns[np.isnan(nextEventOns)] = time_intervals[np.isnan(nextEventOns), 1]

        spike_intervals = np.array(list(zip(prevEventOffs, nextEventOns)))
        bin_size = 1 / sample_rate  # Convert to seconds

        all_rel_times = []
        for i in range(len(spike_intervals)):
            t0, t1 = spike_intervals[i]
            align = align_values[i]
            all_rel_times.append([t0 - align, t1 - align])

        min_time = np.nanmin([t[0] for t in all_rel_times])
        max_time = np.nanmax([t[-1] for t in all_rel_times])
        min_time = np.floor(min_time / bin_size) * bin_size
        max_time = np.ceil(max_time / bin_size) * bin_size
        common_time = np.arange(min_time, max_time + bin_size, bin_size)


        spike_binary = np.zeros((len(spike_intervals), len(neuron_df), len(common_time)), dtype=int)

        for neuron_idx, spikes in enumerate(neuron_df['spike_time']):
            for trial_idx, ((t0, t1), align) in enumerate(zip(spike_intervals, align_values)):
                trial_spikes = spikes[(spikes >= t0) & (spikes <= t1)] - align
                indices = np.searchsorted(common_time, trial_spikes)
                indices = indices[(indices >= 0) & (indices < len(common_time))]
                spike_binary[trial_idx, neuron_idx, indices] = 1

        return spike_binary, common_time

    def get_segmented_times(self, labels_df: pd.DataFrame, segment_df: pd.DataFrame = None,
                            input_value: Optional[Union[int, List[int], np.ndarray, str]] = None, 
                            align_col: str = 'onset') -> pd.DataFrame:
        """
        Processes and aligns time segments from a labels DataFrame and optionally a segment DataFrame.
        Args:
            labels_df (pd.DataFrame): A DataFrame containing label information with 'onset' and 'offset' columns.
            segment_df (pd.DataFrame, optional): 
                A DataFrame containing segment information with columns such as 'segmentOnset', 
                'segmentOffset', and 'segmentTrialIndex'. Defaults to None.
            input_value (Optional[Union[int, List[int], np.ndarray, str]]): 
                A value or list of values used to filter trials. Can be an integer, list of integers, 
                numpy array, or string. Defaults to None.
            align_col (str): The column name in `labels_df` used for alignment. Defaults to 'onset'.
        Returns:
            pd.DataFrame: A DataFrame with updated and aligned time segments. If `segment_df` is provided, 
            the resulting DataFrame includes merged segment information.
        Notes:
            - The function computes additional columns in `labels_df` such as 'prevMaxOffset' and 
                'nextMinOnset' for contextual information.
            - If `segment_df` is provided, its columns are renamed to prepend "segment" and aligned 
                based on the `align_col` value in `labels_df`.
            - The function filters trials based on `input_value` and ensures alignment of time segments 
                across both DataFrames.
        """
        valid_column = lambda x: any(k in x.lower() for k in ['onset', 'offset'])
        
        labels_copy = labels_df.copy()
        labels_copy['prevMaxOffset'] = labels_copy['offset'].shift(1)
        labels_copy['nextMinOnset'] = labels_copy['onset'].shift(-1)
        valid_cols = [col for col in labels_copy.columns if valid_column(col)]
        labels_copy.update(labels_copy[valid_cols].subtract(labels_copy[align_col], axis=0))
        
        trials = self.get_trial_indices(labels_df, input_value)
        labels_copy = labels_copy[labels_copy.index.isin(trials)]

        if segment_df is not None:
            segment_df = segment_df.copy()
            # Rename the columns to prepend segment{column.capitalize()}
            segment_df.rename(columns={col: f'segment{col[0].upper()}{col[1:]}' for col in segment_df.columns}, inplace=True)
            segment_df = segment_df[segment_df['segmentTrialIndex'].isin(trials)]
            for trial in trials:
                align = labels_df.loc[trial][align_col]
                mask = segment_df['segmentTrialIndex'] == trial
                segment_df.loc[mask, 'segmentOnset'] -= align
                segment_df.loc[mask, 'segmentOffset'] -= align

            segment_df.sort_values(by=['segmentOnset'], inplace=True)
            segment_df = segment_df.groupby('segmentTrialIndex').agg(list).reset_index()
            labels_copy = labels_copy.merge(segment_df, left_index=True, right_on='segmentTrialIndex', how='left')

        return labels_copy        

    def assign_color_map(self, labels_df: pd.DataFrame, color_dict: Dict[str, str] = None, cmap_column: str = 'label') -> pd.DataFrame:
        """
        Assign colors to labels in the DataFrame based on a provided color dictionary.

        Parameters:
        labels_df (DataFrame): DataFrame containing labels to be colored.
        color_dict (dict): Dictionary mapping labels to colors.
        cmap_column (str): Column name in labels_df containing the labels.

        Returns:
        DataFrame: DataFrame with an additional 'color' column for each label.
        """

        if color_dict is None:
            # If the input is a string, treat it as a single column to color
            if isinstance(labels_df[cmap_column].values[0], str):
                unique_labels = labels_df[cmap_column].unique()
            # If the input is a list of strings, handle multiple label columns
            elif isinstance(labels_df[cmap_column].values[0], (list, np.ndarray)):
                color_dict = {}
                unique_labels = sum(labels_df[cmap_column].values, [])
                unique_labels = [label for label in unique_labels if not pd.isna(label) and label.lower() != 'nan']
                unique_labels = np.unique(unique_labels)
            color_dict = {label: plt.cm.tab20(i % 20) for i, label in enumerate(unique_labels)}

        # Map the colors based on the created color dictionary
        if isinstance(labels_df[cmap_column].values[0], str):
            labels_df['color'] = labels_df[cmap_column].apply(lambda x: color_dict.get(x, 'black'))
            labels_df['color'] = labels_df['color'].apply(lambda x: to_hex(x))
        elif isinstance(labels_df[cmap_column].values[0], (list, np.ndarray)):
            labels_df['color'] = labels_df[cmap_column].apply(lambda x: [color_dict.get(label, 'black') for label in x])
            labels_df['color'] = labels_df['color'].apply(lambda x: [to_hex(color) for color in x])
        return labels_df
    
    def get_event_df(self, task_name: str, enrich: bool = True, make_stim_cue_cols: bool = True,
                     segment_on: Optional[str] = None, align_col: str = 'onset', color_dict: Optional[Dict[str, str]] = None,
                     cmap_column: str = 'label'
                     ) -> pd.DataFrame:
        """
        Generate a DataFrame of events for a given task, with optional segmentation and color mapping.
        Args:
            task_name (str): The name of the task for which to generate the event DataFrame.
            enrich (bool, optional): Whether to enrich the task DataFrame with additional information. Defaults to True.
            make_stim_cue_cols (bool, optional): Whether to create stimulus and cue columns in the task DataFrame. Defaults to True.
            segment_on (Optional[str], optional): Specifies the segmentation level. Must be one of ['phoneme', 'syllable', 'word'].
                If None, no segmentation is applied. Defaults to None.
            align_col (str, optional): The column used for alignment in the segmented DataFrame. Defaults to 'onset'.
            color_dict (Optional[Dict[str, str]], optional): A dictionary mapping labels to specific colors. If None, a default
                color map is used. Defaults to None.
            cmap_column (str, optional): The column used to determine the color mapping. Defaults to 'label'.
        Returns:
            pd.DataFrame: A DataFrame containing the events, optionally segmented and color-mapped.
        Raises:
            ValueError: If `segment_on` is specified but is not one of ['phoneme', 'syllable', 'word'].
        Notes:
            - This function relies on several helper methods, such as `get_task_df`, `get_phoneme_df`, `get_syllable_df`,
                `get_word_df`, `get_segmented_times`, and `assign_color_map`.
            - Use this function as a template for generating event DataFrames with custom segmentation or color mapping.
        """
        labels_df = self.get_task_df(task_name, enrich=enrich, make_stim_cue_cols=make_stim_cue_cols)
        segment_df = None
        if isinstance(segment_on, str) and segment_on.lower() == 'phoneme':
            segment_df = self.get_phoneme_df(labels_df)
            cmap_column = f'segment{cmap_column[0].upper()}{cmap_column[1:]}'
        elif isinstance(segment_on, str) and segment_on.lower() == 'syllable':
            segment_df = self.get_syllable_df(labels_df)
            cmap_column = f'segment{cmap_column[0].upper()}{cmap_column[1:]}'
        elif isinstance(segment_on, str) and segment_on.lower() == 'word':
            segment_df = self.get_word_df(labels_df)
            cmap_column = f'segment{cmap_column[0].upper()}{cmap_column[1:]}'
        elif isinstance(segment_on, str) and segment_on not in ['phoneme', 'syllable', 'word']:
            raise ValueError("segment_on must be one of ['phoneme', 'syllable', 'word']. Use this function as a template.")
        segmented_df = self.get_segmented_times(labels_df, segment_df=segment_df, align_col=align_col)
        segmented_df = self.assign_color_map(segmented_df, color_dict=color_dict, cmap_column=cmap_column)
        return segmented_df

    def get_audio_clips(self, t0: Union[float, np.ndarray], t1: Union[float, np.ndarray]) -> Tuple[np.ndarray, int]:
        """
        Extract audio segments from an audio file based on specified time intervals.
        Parameters:
        -----------
        t0 : Union[float, np.ndarray]
            Start time(s) in seconds. Can be a single float value or a numpy array of floats.
        t1 : Union[float, np.ndarray]
            End time(s) in seconds. Can be a single float value or a numpy array of floats.
            Must have the same length as `t0`.
        Returns:
        --------
        Tuple[np.ndarray, int]
            A tuple containing:
            - A list of numpy arrays, where each array represents an audio segment
              corresponding to the specified time intervals.
            - The sample rate (int) of the audio file.
        Raises:
        -------
        ValueError
            If `t0` and `t1` have different lengths.
            If the audio file is not found or not specified.
        Notes:
        ------
        - The method reads the audio file specified by `self.audio_file`.
        - Time intervals are converted to sample indices using the sample rate of the audio file.
        - The extracted audio segments are returned as numpy arrays.
        """
        if isinstance(t0, (int, float)):
            t0 = np.array([t0])
        if isinstance(t1, (int, float)):
            t1 = np.array([t1])
        if len(t0) != len(t1):
            raise ValueError("t0 and t1 must be the same length.")
        
        if self.audio_file is None:
            raise ValueError("Audio directory not found.")
        
        audio, sr = sf.read(self.audio_file)
        time_pairs = np.array(list(zip(t0, t1)))
        time_idxs = np.array([librosa.time_to_samples(pair, sr=sr) for pair in time_pairs])
        audio_segments = []
        for t0, t1 in time_idxs:
            audio_segments.append(audio[t0:t1])
        
        return audio_segments, sr
    
    def get_mel_spectrogram(self, audio_clips: np.ndarray, sr: float) -> np.ndarray:
        """
        Computes the mel spectrograms for a given set of audio clips.

        This function takes an array of audio clips and their sampling rate, 
        computes the mel spectrogram for each audio clip, and converts it to 
        decibel (dB) scale.

        Args:
            audio_clips (np.ndarray): A numpy array containing audio clips. 
                                      Each clip should be a 1D array.  Get (audio_clips, sr) from the `get_audio_clips` function
            sr (float): The sampling rate of the audio clips.

        Returns:
            np.ndarray: A list of mel spectrograms in decibel scale, where each 
                        spectrogram corresponds to an audio clip.
        """
        mel_spectrograms = []
        for audio_segment in audio_clips.copy():
            # Ensure the audio segment is a 1D array
            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.flatten()
            mel_spect = librosa.feature.melspectrogram(y=audio_segment, 
                                            sr=sr,
                                            n_mels=128,
                                            fmax=8000)

            mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
            mel_spectrograms.append(mel_spect_db)

        return mel_spectrograms

    def get_pitch(self, audio_clips: np.ndarray, sr: float) -> np.ndarray:
        """
        Extract the pitch (fundamental frequency) from audio clips using the PYIN algorithm.

        Parameters:
        -----------
        audio_clips : np.ndarray
            A NumPy array containing audio segments. Each segment should be a 1D array.  
            Get (audio_clips, sr) from the `get_audio_clips` function
        sr : float
            The sampling rate of the audio clips.

        Returns:
        --------
        np.ndarray
            A list of NumPy arrays, where each array contains the pitch values (f0) 
            for the corresponding audio segment. Unvoiced segments will have `None` 
            values in place of pitch values.

        Notes:
        ------
        - This function uses the `librosa.pyin` method to estimate the pitch.
        - The pitch is estimated within the range of C2 (65.41 Hz) to C7 (2093 Hz).
        - If an audio segment has more than one dimension, it will be flattened 
          before processing.
        """
        pitches = []
        for audio_segment in audio_clips.copy():
            # Ensure the audio segment is a 1D array
            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.flatten()
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            pitches.append(f0)
        return pitches

    def get_loudness(self, audio_clips: np.ndarray, sr: float) -> np.ndarray:
        """
        Calculate the loudness of audio clips.
        This function computes the root mean square (RMS) energy of each audio segment 
        in the input array and converts it to decibels (dB) relative to the maximum amplitude.
        Args:
            audio_clips (np.ndarray): A 2D array where each row represents an audio segment.
                                      Get (audio_clips, sr) from the `get_audio_clips` function
            sr (float): The sampling rate of the audio clips.
        Returns:
            np.ndarray: A list of arrays, where each array contains the loudness values 
                        (in dB) for the corresponding audio segment.
        """
        
        loudness = []
        for audio_segment in audio_clips.copy():
            # Ensure the audio segment is a 1D array
            if len(audio_segment.shape) > 1:
                audio_segment = audio_segment.flatten()
            loud = librosa.feature.rms(y=audio_segment)
            loud = librosa.amplitude_to_db(loud, ref=np.max)
            loudness.append(loud)
        return loudness
    
    def plot_driftmap(self, num_points: int = 300000, plot_tasks: bool = True, plot_motion: bool = True, color_clusters: bool = False) -> Dict[int, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots drift maps for each probe directory in `self.ks_dirs`.
        This method visualizes the drift of neural recordings over time for each probe. 
        It uses spike times, spike clusters, and spike positions to create scatter plots, 
        and optionally overlays drift correction data and task timings.
        Args:
            num_points (int, optional): The maximum number of points to display in the scatter plot. 
                If the total number of spikes exceeds this value, a random subset of spikes is selected. 
                Defaults to 300,000.
            plot_tasks (bool, optional): If True, overlays task timings on the plots.
                Defaults to True.
            plot_motion (bool, optional): If True, overlays motion correction data on the plots.
                Defaults to True.
            color_clusters (bool, optional): If True, colors the spikes by their cluster IDs rather than using amplitudes.
                Defaults to False.
        Returns:
            Dict[int, Tuple[plt.Figure, plt.Axes]]: A dictionary where the keys are probe numbers 
            (extracted from the directory names) and the values are tuples containing the Matplotlib 
            figure and axes objects for the corresponding drift map.
        Notes:
            - The method assumes that each probe directory contains the following files:
                - `spike_times.npy`: Spike times in samples.
                - `spike_clusters.npy`: Cluster IDs for each spike.
                - `spike_positions.npy`: Spike positions, where the second column represents depths.
                - `amplitudes.npy`: Spike amplitudes, if available.
                - `ops.npy` (optional): Contains metadata such as sampling frequency (`fs`), 
                  drift correction data (`dshift`), and time bounds (`tmin`, `tmax`).
            - If `ops.npy` is not found, a default sampling frequency of 30,000 Hz is used.
            - If `self.task_timings` is provided, task start and end times are marked on the plots.
        """
        # From ks_dirs find the directory         
        probe_dict = {}
        for probe_dir in self.ks_dirs:
            fig, ax = plt.subplots(figsize=(10, 5))
            probe_dir = Path(probe_dir)
            probe_num = int(re.search(r"imec(\d+)", str(probe_dir)).group(1)) # Search for the imec probe number

            ops_dir = probe_dir / "ops.npy" # Kilosort4 ops file
            rez_dir = probe_dir / "rez.mat" # Kilosort2.5 rez file

            if rez_dir.exists():
                rez = extract_waveform.read_rezfile(probe_dir)
                ops, dshift = rez['ops'], rez['dshift']
                min_time, max_time, fs = ops['trange'][0], ops['trange'][1], float(rez['ops']['fs'])
                dshift = dshift.T if dshift is not None else None

                if 'st0' in rez.keys() and not color_clusters:
                    st0 = rez['st0']
                    spike_times = st0[0, :].flatten() / fs
                    spike_clusters = np.zeros_like(spike_times, dtype=int)  # No cluster info in st0
                    spike_amps = st0[2, :].flatten()
                    depths = st0[1, :].flatten()

                else: # We either want to color clusters or all spikes detected is not available
                    spike_times = np.load(probe_dir / "spike_times.npy").flatten() / fs
                    spike_clusters = np.load(probe_dir / "spike_clusters.npy").flatten()
                    spike_amps = np.load(probe_dir / "amplitudes.npy").flatten()

                    if (probe_dir / "spike_centroids.npy").exists():
                        depths = np.load(probe_dir / "spike_centroids.npy")[:, 1] # Use the calculated centroids
                    else:
                        iTemp = np.load(probe_dir / "spike_templates.npy").flatten() # Load spike templates
                        iChan = (rez['iNeighPC'][iTemp, 15] - 1).astype(int)  # Get the channel index for the 16th neighbor
                        depths = rez['ycoords'][iChan] # Calculate depths from the ycoords of the channels

                if dshift is not None:
                    dshift_sec = np.linspace(min_time, max_time, dshift.shape[0])
                    depth_bins = np.linspace(0, np.max(depths), dshift.shape[1])
                    for i in range(dshift.shape[1]):
                        dshift[:,i] = -dshift[:,i] + depth_bins[i]   

            elif ops_dir.exists():
                ops = np.load(probe_dir / "ops.npy", allow_pickle=True).item()
                min_time, max_time, fs, dshift = ops['tmin'], ops['tmax'], ops['fs'], ops['dshift']

                spike_times = np.load(probe_dir / "spike_times.npy").flatten() / fs
                spike_clusters = np.load(probe_dir / "spike_clusters.npy").flatten()
                spike_amps = np.load(probe_dir / "amplitudes.npy").flatten()
                depths = np.load(probe_dir / "spike_positions.npy")[:, 1]
                
                if dshift is not None:
                    dshift_sec = np.linspace(min_time, max_time, dshift.shape[0])
                    depth_bins = np.linspace(0, np.max(depths), dshift.shape[1])
                    for i in range(dshift.shape[1]):
                        dshift[:,i] = -dshift[:,i] + depth_bins[i]
            
            # Keep the indices of spike_amps > 8 and spike_amps < 100
            if not color_clusters:
                non_noise_indices = np.where((spike_amps > 8) & (spike_amps < 100))[0]
                spike_times = spike_times[non_noise_indices]
                depths = depths[non_noise_indices]
                spike_clusters = spike_clusters[non_noise_indices]
                spike_amps = spike_amps[non_noise_indices]

            # Reduce spike_times, depths, and spike_clusters to display at most num_points
            if len(spike_times) > num_points:
                indices = np.random.choice(len(spike_times), num_points, replace=False)
                spike_times = spike_times[indices]
                depths = depths[indices]
                spike_clusters = spike_clusters[indices]
                spike_amps = spike_amps[indices] 
            
            if color_clusters:
                ax.scatter(spike_times, depths,
                        c=spike_clusters % 20,
                        cmap='tab20',
                        alpha=0.5,
                        s=2, rasterized=True)
            else:
                spike_amps = np.maximum(0, 1 - spike_amps / 40)
                sorted_indices = np.argsort(spike_amps)[::-1]
                spike_times = spike_times[sorted_indices]
                depths = depths[sorted_indices]
                spike_amps = spike_amps[sorted_indices]

                color_rgb = np.column_stack([spike_amps] * 3)
                ax.scatter(spike_times, depths,
                           c=color_rgb, # Normalize amplitudes to [0, 1] and use black to white colormap
                           alpha=0.5,
                           s=2, rasterized=True)
            
            if plot_motion and dshift is not None:
                ax.plot(dshift_sec, dshift, color='black', linewidth=1)
            if self.task_timings is not None and plot_tasks:
                for task, start, end in self.task_timings:
                    ax.axvline(x=start, color='black', linestyle='--', linewidth=1)
                    ax.axvline(x=end, color='black', linestyle='--', linewidth=1)
                    ax.text((start + end) / 2, ax.get_ylim()[1]*0.96, task, 
                             horizontalalignment='center', 
                             verticalalignment='center', 
                             fontsize=12, 
                             color='black')
                    
            ax.set_title(f"Probe {probe_num} (Yield: {len(np.unique(np.load(probe_dir / 'spike_clusters.npy').flatten()))} units)")
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Distance from probe tip (μm)')
            probe_dict[probe_num] = (fig, ax)
        return probe_dict
    
    def get_ap_bin(self):
        ap_bins = []
        for probe_dir in self.ks_dirs:
            probe_dir = Path(probe_dir)
            probe_dir = str(probe_dir).replace('/kilosort/', '/sglx/')
            probe_dir = probe_dir.replace(f'{self.ks_suffix}', "")
            probe_dir = Path(probe_dir)
            filename = str(probe_dir.name).replace('_imec', '_tcat.imec') + ".ap.bin"
            probe_dir = probe_dir / filename
            if probe_dir.exists():
                ap_bins.append(probe_dir)
            else:
                raise ValueError(f"AP bin file not found: {probe_dir}")
        return ap_bins

    
        
