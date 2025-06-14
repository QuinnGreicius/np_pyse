import numpy as np
from pathlib import Path
import os

def read_pyfile(filepath, ignored_chars=[" ", "'", "\"", "\n", "\r"]):
    from ast import literal_eval as ale
    '''
    Reads .py file and returns contents as dictionnary.

    Assumes that file only has "variable=value" pairs, no fucntions etc

    - filepath: str, path to file
    - ignored_chars: list of characters to remove (only trailing and leading)
    '''
    filepath = Path(filepath)
    assert filepath.exists(), f'{filepath} not found!'

    params={}
    with open(filepath) as f:
        for ln in f.readlines():
            assert '=' in ln, 'WARNING read_pyfile only works for list of variable=value lines!'
            tmp = ln.split('=')
            for i, string in enumerate(tmp):
                string=string.strip("".join(ignored_chars))
                tmp[i]=string
            k, val = tmp[0], tmp[1]
            try: val = ale(val)
            except: pass
            params[k]=val

    return params

def read_rezfile(ks_folder):
    import h5py

    def h5_to_dict(h5obj):
        if isinstance(h5obj, h5py.Group):  # Group is like a dictionary
            return {key: h5_to_dict(h5obj[key]) for key in h5obj}
        elif isinstance(h5obj, h5py.Dataset):  # Dataset is like an array
            return np.squeeze(np.array(h5obj))
        else:
            return h5obj
    
    rez_path = ks_folder / 'rez.mat'
    with h5py.File(rez_path, 'r') as f:
        rez = h5_to_dict(f['rez'])
    return rez

def get_temp_wh(ks_folder):
    ks_folder = Path(ks_folder)
    params = read_pyfile(ks_folder / 'params.py')
    dat_path = ks_folder / 'temp_wh.dat'
    dtype = np.dtype(params['dtype'])
    n_channels_dat = params['n_channels_dat']
    offset = params['offset']

    # Read temp_wh as a memmap file
    filesize_bytes = os.path.getsize(dat_path)
    n_samples_total = filesize_bytes // np.dtype('int16').itemsize
    n_timepoints = n_samples_total // int(n_channels_dat)
    temp_wh = np.memmap(dat_path, dtype=dtype, mode='r', offset=offset, shape=(n_timepoints, n_channels_dat))
    return temp_wh

def get_fs(ks_folder):
    params = read_pyfile(ks_folder / 'params.py')
    return params['sample_rate']

def get_tmin(ks_folder):
    # Check if ops.npy exists
    ops_path = ks_folder / 'ops.npy'
    rez_path = ks_folder / 'rez.mat'
    if ops_path.exists() and t0 is None:
        ops = np.load(ops_path, allow_pickle=True).item()
        return ops['tmin']
    elif rez_path.exists() and t0 is None:
        rez = read_rezfile(ks_folder)
        return rez['ops']['trange'][0]
    else:
        raise FileNotFoundError("Neither ops.npy nor rez.mat found in the specified folder.")

def read_whitened_waveforms(ks_folder, clus_id=0, n_samples=40, n_channels=10,
                            spike_times=None, spike_clusters=None, templates=None,
                            temp_wh=None, fs=None, t0=None):
    ks_folder = Path(ks_folder)
    
    if temp_wh is None:
        temp_wh = get_temp_wh(ks_folder)
    if fs is None:
        fs = get_fs(ks_folder)
    if t0 is None:
        t0 = get_tmin(ks_folder)
    if spike_times is None:
        spike_times = np.load(ks_folder / 'spike_times.npy').squeeze()
    if spike_clusters is None:
        spike_clusters = np.load(ks_folder / 'spike_clusters.npy').squeeze()
    if templates is None:
        templates = np.load(ks_folder / 'templates.npy')
    
    spk_mask = spike_clusters == clus_id
    spk_idx = spike_times[spk_mask].astype(int)
    chan_best = (templates[clus_id]**2).sum(axis=0).argmax()

    # Define spike-centered time and channel windows
    t_center = spk_idx - int(t0 * fs)
    time_window = np.arange(-n_samples, n_samples + 1)  # (2 * n_samples + 1,)
    channel_window = np.arange(-n_channels // 2, n_channels - n_channels // 2)  # length n_channels

    # Compute 2D index arrays for time and channels
    time_idx = t_center[:, None] + time_window[None, :]  # (n_spikes, n_window)
    time_idx_exp = time_idx[:, :, None]  # (n_spikes, n_window, 1)
    valid_time = (time_idx_exp >= 0) & (time_idx_exp < temp_wh.shape[0])
    time_idx_exp = np.clip(time_idx_exp, 0, temp_wh.shape[0] - 1) 

    space_idx = chan_best + channel_window  # (n_spikes, n_channels)
    space_idx_exp = space_idx[None, None, :]  # (1, 1, n_channels)
    valid_space = (space_idx_exp >= 0) & (space_idx_exp < temp_wh.shape[1])
    space_idx_exp = np.clip(space_idx_exp, 0, temp_wh.shape[1] - 1)  # Ensure indices are within bounds
    
    time_idx_broadcasted = np.broadcast_to(time_idx_exp, (time_idx.shape[0], time_idx.shape[1], space_idx.shape[0]))
    space_idx_broadcasted = np.broadcast_to(space_idx_exp, (time_idx.shape[0], time_idx.shape[1], space_idx.shape[0]))
    flat_time_idx = time_idx_broadcasted.reshape(-1)
    flat_space_idx = space_idx_broadcasted.reshape(-1)

    waveforms_flat = temp_wh[flat_time_idx, flat_space_idx]  # shape: (126*81*10,)
    waveforms = waveforms_flat.reshape(time_idx.shape[0], time_idx.shape[1], space_idx.shape[0])  # (126, 81, 10)
    waveforms = waveforms.astype(np.float32)
    waveforms[~valid_time | ~valid_space] = np.nan
    return waveforms

# https://github.com/m-beau/NeuroPyxels?tab=readme-ov-file#load-waveforms-from-unit-u
def get_raw_waveforms(dp, u, n_waveforms=100, t_waveforms=82, selection='regular', periods='all',
                  spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
                  whiten=0, med_sub=0, hpfilt=0, hpfiltf=300,
                  nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=True, verbose=False,
                  med_sub_in_time=True, return_corrupt_mask=False, again=False,
                  cache_results=True, cache_path=None):

    raise NotImplementedError("This function is a work in progress. Please use the whitened_waveforms for now.")
    # Extract and process metadata
    dp             = Path(dp)
    meta           = read_metadata(dp)
    dat_path       = get_binary_file_path(dp, 'ap')

    dp_source      = get_source_dp_u(dp, u)[0]
    meta           = read_metadata(dp_source)
    dtype          = np.dtype(meta['highpass']['datatype'])
    n_channels_dat = meta['highpass']['n_channels_binaryfile']
    n_channels_rec = n_channels_dat-1 if meta['acquisition_software']=='SpikeGLX' else n_channels_dat
    sample_rate    = meta['highpass']['sampling_rate']
    item_size      = dtype.itemsize
    fileSizeBytes  = meta['highpass']['binary_byte_size']
    assert not isinstance(fileSizeBytes, str), f"It seems like there isn't any binary file at {dp}."
    if meta['acquisition_software']=='SpikeGLX':
        if meta['highpass']['fileSizeBytes'] != fileSizeBytes:
            print((f"\033[91;1mMismatch between ap.meta and ap.bin file size"
            "(assumed encoding is {str(dtype)} and Nchannels is {n_channels_dat})!! "
            f"Probably wrong meta file - just edit fileSizeBytes in the .ap.meta file at {dp} "
            f"(replace {int(meta['highpass']['fileSizeBytes'])} with {fileSizeBytes}) "
            "and be aware that something went wrong in your data management...\033[0m"))

    # Select subset of spikes
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    if spike_ids is None:
        spike_ids_subset = get_ids_subset(dp, u,
                                          n_waveforms, wvf_batch_size, selection, periods,
                                          ignore_nwvf, verbose,
                                          again, cache_results=cache_results, cache_path=cache_path)
    else:
        assert isinstance(spike_ids, Iterable), "WARNING spike_ids must be a list/array of ids!"
        spike_ids_subset = np.array(spike_ids)
    n_spikes = len(spike_ids_subset)

    # Get waveforms times in bytes
    # and check that, for this waveform width,
    # they no not go beyond file limits
    waveforms_t  = spike_samples[spike_ids_subset].astype(np.int64)
    waveforms_t1 = (waveforms_t-t_waveforms//2)*n_channels_dat*item_size
    waveforms_t2 = (waveforms_t+t_waveforms//2)*n_channels_dat*item_size
    wcheck_m=(0<=waveforms_t1)&(waveforms_t2<fileSizeBytes)
    if not np.all(wcheck_m):
        print(f"Invalid times: {waveforms_t[~wcheck_m]}")
        waveforms_t1 = waveforms_t1[wcheck_m]
        waveforms_t2 = waveforms_t2[wcheck_m]

    # Iterate over waveforms
    waveforms = np.zeros((n_spikes, t_waveforms, n_channels_rec), dtype=np.float32)
    if verbose: print(f'Loading waveforms of unit {u} ({n_spikes})...')
    with open(dat_path, "rb") as f:
        for i,t1 in enumerate(waveforms_t1):
            if n_spikes>10:
                if i%(n_spikes//10)==0 and verbose: print(f'{round((i/n_spikes)*100)}%...', end=' ')
            f.seek(t1, 0) # 0 for absolute file positioning
            try:
                wave = f.read(n_channels_dat*t_waveforms*item_size)
                wave = np.frombuffer(wave, dtype=dtype).reshape((t_waveforms,n_channels_dat))
                # get rid of sync channel
                waveforms[i,:,:] = wave[:,:-1] if meta['acquisition_software']=='SpikeGLX' else wave
            except:
                print(f"WARNING it seems the binary file at {dp} is corrupted. Waveform {i} (at byte {t1}, {t1/n_channels_dat/item_size/sample_rate}s) could not be loaded.")
                waveforms[i,:,:] = np.nan
    corrupt_mask = np.isnan(waveforms[:,0,0])
    waveforms = waveforms[~corrupt_mask,:,:]
    n_spikes -= np.sum(corrupt_mask)
    if med_sub_in_time:
        medians = np.median(waveforms, axis = 1)
        waveforms = waveforms - medians[:,np.newaxis,:]
    if verbose: print('\n')

    # Preprocess waveforms
    if hpfilt|med_sub|whiten:
        waveforms     = waveforms.reshape((n_spikes*t_waveforms, n_channels_rec))
        if hpfilt:
            waveforms = apply_filter(waveforms, bandpass_filter(rate=sample_rate, low=None, high=hpfiltf, order=3), axis=0)
        if med_sub:
            waveforms = med_substract(waveforms, axis=1, nRange=nRangeMedSub)
        if whiten:
            waveforms = whitening(waveforms.T, nRange=nRangeWhiten).T # whitens across channels so gotta transpose
        waveforms     = waveforms.reshape((n_spikes,t_waveforms, n_channels_rec))

    # Filter channels ignored by kilosort if necesssary
    if not ignore_ks_chanfilt:
        channel_ids_ks = np.load(Path(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
        channel_ids_ks = channel_ids_ks[channel_ids_ks!=384]
        waveforms      = waveforms[:,:,channel_ids_ks] # only AFTER processing, filter out channels

    # Correct voltage scaling
    waveforms *= meta['bit_uV_conv_factor']

    if return_corrupt_mask:
        return waveforms, corrupt_mask
    
    return waveforms.astype(np.float32)