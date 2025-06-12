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

def read_whitened_waveforms(ks_folder, clus_id=0, n_samples=40, n_channels=10):
    ks_folder = Path(ks_folder)
    params = read_pyfile(ks_folder / 'params.py')
    dat_path = ks_folder / 'temp_wh.dat'
    dtype = np.dtype(params['dtype'])
    n_channels_dat = params['n_channels_dat']
    fs = params['sample_rate']
    offset = params['offset']

    # Read temp_wh as a memmap file
    filesize_bytes = os.path.getsize(dat_path)
    n_samples_total = filesize_bytes // np.dtype('int16').itemsize
    n_timepoints = n_samples_total // int(n_channels_dat)
    temp_wh = np.memmap(dat_path, dtype=dtype, mode='r', offset=offset, shape=(n_timepoints, n_channels_dat))

    spike_times = np.load(ks_folder / 'spike_times.npy').squeeze()
    spike_clusters = np.load(ks_folder / 'spike_clusters.npy').squeeze()
    templates = np.load(ks_folder / 'templates.npy')

    # Check if ops.npy exists
    ops_path = ks_folder / 'ops.npy'
    rez_path = ks_folder / 'rez.mat'
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        t0 = ops['tmin']
    elif rez_path.exists():
        rez = read_rezfile(ks_folder)
        t0 = rez['ops']['trange'][0]
    else:
        raise FileNotFoundError("Neither ops.npy nor rez.mat found in the specified folder.")

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
    valid_time = (time_idx_exp >= 0) & (time_idx_exp < n_timepoints)
    time_idx_exp = np.clip(time_idx_exp, 0, n_timepoints - 1) 

    space_idx = chan_best + channel_window  # (n_spikes, n_channels)
    space_idx_exp = space_idx[None, None, :]  # (1, 1, n_channels)
    valid_space = (space_idx_exp >= 0) & (space_idx_exp < n_channels_dat)
    space_idx_exp = np.clip(space_idx_exp, 0, n_channels_dat - 1)  # Ensure indices are within bounds
    
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
    
def metadata(dp):
    '''
    Read spikeGLX (.ap/lf.meta) or openEphys (.oebin) metadata files
    and returns their contents as dictionnaries.

    The 'highpass' or 'lowpass' nested dicts correspond to Neuropixels 1.0 high or low pass filtered metadata.
    2.0 recordings only have a 'highpass' key, as they are acquired as a single file matched with a .ap.meta file.
        for spikeGLX, corresponds to metadata of .ap.meta and .lf.meta files.
        for OpenEphys, .oebin metadata relating to the first and second dictionnaries in 'continuous' of the .oebin file
                       which match the /continuous/Neuropix-PXI-100.0 or .1 folders respectively.

    Arguments:
        - dp: str, datapath to spike sorted dataset

    Returns:
        - meta: dictionnary containing contents of meta file.
        the structure of meta is as follow:
        {
        'probe_version': either of '3A', '1.0_staggered', '2.0_1shank', '2.0_4shanks', 'ultra_high_density';
        'highpass':
            {
            'binary_relative_path':relative path to binary file from dp,
            'sampling_rate':int, # sampling rate
            'n_channels_binaryfile':int, # n channels saved on file, typically 385 for .bin and 384 for .dat
            'n_channels_analysed':int, # n channels used for spikesorting. Will set the shape of temp_wh.daat for kilosort.
            'datatype':str, # datatype of binary encoding, typically int16
            'binary_relative_path':relative path to binary file from dp,
            'key1...': all other keys present in meta file, that you must be familiar with!
                       e.g. 'fileSizeBytes' for spikeGLX or 'channels' for OpenEphys...
            },
        'lowpass': {...}, # same as high for low pass filtered data (not existing in 2.0 recordings)
        'events': {...}, # only for openephys recordings, contents of oebin file
        'spikes': {...} # only for openephys recordings, contents of oebin file
        }
    '''
    dp = Path(dp)
    assert dp.exists(), "Provided path does not exist!"
    assert dp.is_dir(), f"Provided path {dp} is a filename!"

    probe_versions = {
        'glx':{3.0:  '3A', # option 3
               0.0:  '1.0',
               1.0:  '1.0', # precise type unknown

               1020:  '1.0', # precise type unknown
               1100:  '1.0', # precise type unknown
               1200:  '1.0', # precise type unknown
               1300:  '1.0', # precise type unknown

               1110:  '1.0', # precise type unknown
               1120:  '1.0', # precise type unknown
               1121:  '1.0', # precise type unknown
               1122:  '1.0', # precise type unknown
               1123:  'ultra_high_density',

               1030: 'NHP_1.0',

               21:   '2.0_singleshank',
               2003: '2.0_singleshank',
               2004: '2.0_singleshank',

               24:   '2.0_fourshanks',
               2013: '2.0_fourshanks',
               2014: '2.0_fourshanks', # assumed type
               2020: '2.0_fourshanks', # assuned type
               },
        'oe':{"Neuropix-3a":'3A', # source_processor_name keys
                "Neuropix-PXI":'1.0',
                '?1':'2.0_singleshank', # do not know yet
                '?2':'2.0_fourshanks'}, # do not know yet
        'int':{'3A':1,
               '1.0':1,
               'NHP_1.0':1,
               '2.0_singleshank':2,
               '2.0_fourshanks':2,
               'ultra_high_density':3}
        }

    # import params.py data
    params_f = dp/'params.py'
    if params_f.exists():
        params=read_pyfile(dp/'params.py')

    # find meta file
    glx_ap_files = list_files(dp, "ap.meta", True)
    glx_lf_files = list_files(dp, "lf.meta", True)
    oe_files = list_files(dp, "oebin", True)
    glx_found = np.any(glx_ap_files) or np.any(glx_lf_files)
    oe_found = np.any(oe_files)
    assert glx_found or oe_found, \
        f'WARNING no .ap/lf.meta (spikeGLX) or .oebin (OpenEphys) file found at {dp}.'
    assert not (glx_found and oe_found),\
        'WARNING dataset seems to contain both an open ephys and spikeGLX metafile - fix this!'
    assert len(glx_ap_files)==1 or len(glx_lf_files)==1 or len(oe_files)==1,\
        'WARNING more than 1 .ap.meta or 1 .oebin files found!'

    # Formatting of openephys meta file
    meta = {}
    meta['path'] = os.path.realpath(dp)
    if oe_found:
        meta['acquisition_software']='OpenEphys'
        # Load OpenEphys metadata
        metafile=Path(oe_files[0])
        with open(metafile) as f:
            meta_oe = json.load(f)

        # find probe version
        for i,processor in enumerate(meta_oe['continuous']):
            if 'Neuropix-PXI' in processor["source_processor_name"]:
                probe_index = i
                break
        oe_probe_version = meta_oe["continuous"][probe_index]["source_processor_name"]
        assert oe_probe_version in probe_versions['oe'].keys(),\
            f'WARNING only probe version {oe_probe_version} not handled with openEphys - post an issue at www.github.com/m-beau/NeuroPyxels'
        meta['probe_version']=probe_versions['oe'][oe_probe_version]
        meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]

        # Find conversion factor
        # should be 0.19499999284744262695
        meta['bit_uV_conv_factor']=meta_oe["continuous"][probe_index]["channels"][0]["bit_volts"]

        # index for highpass and lowpass
        filt_index = {'highpass': [], 'lowpass': []}
        for i,processor in enumerate(meta_oe['continuous']):
            if 'AP' in processor['folder_name']:
                filt_index['highpass'] = i
            if 'LFP' in processor['folder_name']:
                filt_index['lowpass'] = i


        # find everything else
        for filt_key in ['highpass','lowpass']:
            meta[filt_key]={}
            filt_key_i=filt_index[filt_key]
            meta[filt_key]['sampling_rate']=float(meta_oe["continuous"][filt_key_i]['sample_rate'])
            meta[filt_key]['n_channels_binaryfile']=int(meta_oe["continuous"][filt_key_i]['num_channels'])
            if params_f.exists():
                meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
                meta[filt_key]['datatype']=params['dtype']
            else:
                meta[filt_key]['n_channels_analysed']=meta[filt_key]['n_channels_binaryfile']
                meta[filt_key]['datatype']='int16'
            binary_folder = './continuous/'+meta_oe["continuous"][filt_key_i]['folder_name']
            binary_file = list_files(dp/binary_folder, "dat", False)
            if any(binary_file):
                binary_rel_path = binary_folder+binary_file[0]
                meta[filt_key]['binary_relative_path']=binary_rel_path
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
                if filt_key=='highpass' and params_f.exists() and params['dat_path']!=binary_rel_path:
                    print((f'\033[34;1mWARNING edit dat_path in params.py '
                    f'so that it matches relative location of high pass filtered binary file: {binary_rel_path}'))
            else:
                meta[filt_key]['binary_relative_path']='not_found'
                meta[filt_key]['binary_byte_size']='unknown'
                print(f"\033[91;1mWARNING {filt_key} binary file not found at {dp}\033[0m")
            meta[filt_key]={**meta[filt_key], **meta_oe["continuous"][filt_key_i]}
        meta["events"]=meta_oe["events"]
        meta["spikes"]=meta_oe["spikes"]


    # Formatting of SpikeGLX meta file
    elif glx_found:
        meta['acquisition_software']='SpikeGLX'
        # Load SpikeGLX metadata
        meta_glx = {}
        for metafile in glx_ap_files+glx_lf_files:
            if metafile in glx_ap_files: filtkey='highpass'
            elif metafile in glx_lf_files: filtkey='lowpass'
            metafile=Path(metafile)
            meta_glx[filtkey]={}
            with open(metafile, 'r') as f:
                for ln in f.readlines():
                    tmp = ln.split('=')
                    k, val = tmp[0], ''.join(tmp[1:])
                    k = k.strip()
                    val = val.strip('\r\n')
                    if '~' in k:
                        meta_glx[filtkey][k] = val.strip('(').strip(')').split(')(')
                    else:
                        try:  # is it numeric?
                            meta_glx[filtkey][k] = float(val)
                        except:
                            meta_glx[filtkey][k] = val

        # find probe version
        if 'imProbeOpt' in meta_glx["highpass"]: # 3A
            glx_probe_version = meta_glx["highpass"]["imProbeOpt"]
        elif 'imDatPrb_type' in meta_glx["highpass"]: # 1.0 and beyond
            glx_probe_version = meta_glx["highpass"]["imDatPrb_type"]
        else:
             glx_probe_version = 'N/A'

        if glx_probe_version in probe_versions['glx']:
            meta['probe_version'] = probe_versions['glx'][glx_probe_version]
            meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]
        else:
            print(f'WARNING probe version {glx_probe_version} not handled - post an issue at www.github.com/m-beau/NeuroPyxels and provide your .ap.meta file.')
            meta['probe_version'] = glx_probe_version
            meta['probe_version_int'] = 0
            

        # Based on probe version,
        # Find the voltage range, gain, encoding
        # and deduce the conversion from units/bit to uV
        Vrange=(meta_glx["highpass"]['imAiRangeMax']-meta_glx["highpass"]['imAiRangeMin'])*1e6
        if meta['probe_version'] in ['3A', '1.0', 'ultra_high_density', 'NHP_1.0']:
            if Vrange!=1.2e6: print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (1.2*10^6). Might be normal!')
            bits_encoding=10
            ampFactor=ale(meta_glx["highpass"]['~imroTbl'][1].split(' ')[3]) # typically 500
            #if ampFactor!=500: print(f'\u001b[31mHeads-up, the voltage amplification factor seems to be {ampFactor}, which is not the default (500). Might be normal!')
        elif meta['probe_version'] in ['2.0_singleshank', '2.0_fourshanks']:
            if Vrange!=1e6:
                print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (10^6). Might be normal!')
            bits_encoding=14
            ampFactor=80 # hardcoded
        else:
            raise ValueError(f"Probe version unhandled - bits_encoding unknown.")
        meta['bit_uV_conv_factor']=(Vrange/2**bits_encoding/ampFactor)


        # find everything else
        for filt_key in ['highpass','lowpass']:
            if filt_key not in meta_glx.keys(): continue
            meta[filt_key]={}

            # binary file
            filt_suffix={'highpass':'ap','lowpass':'lf'}[filt_key]
            binary_rel_path = get_binary_file_path(dp, filt_suffix, False)
            if binary_rel_path!='not_found':
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
                meta[filt_key]['binary_relative_path']='./'+binary_rel_path
            else:
                meta[filt_key]['binary_byte_size']='unknown'
                meta[filt_key]['binary_relative_path']=binary_rel_path
                #print(f"\033[91;1mWARNING binary file .{filt_suffix}.bin not found at {dp}\033[0m")

            # sampling rate
            if meta_glx[filt_key]['typeThis'] == 'imec':
                meta[filt_key]['sampling_rate']=float(meta_glx[filt_key]['imSampRate'])
            else:
                meta[filt_key]['sampling_rate']=float(meta_glx[meta_glx['typeThis'][:2]+'SampRate'])

            meta[filt_key]['n_channels_binaryfile']=int(meta_glx[filt_key]['nSavedChans'])
            if params_f.exists():
                meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
                meta[filt_key]['datatype']=params['dtype']
            else:
                meta[filt_key]['n_channels_analysed']=meta[filt_key]['n_channels_binaryfile']
                meta[filt_key]['datatype']='int16'
            meta[filt_key]={**meta[filt_key], **meta_glx[filt_key]}

    # Calculate length of recording
    high_fs = meta['highpass']['sampling_rate']

    if meta['highpass']['binary_byte_size']=='unknown':
        if (dp/'spike_times.npy').exists():
            t_end=np.load(dp/'spike_times.npy').ravel()[-1]
            meta['recording_length_seconds']=t_end/high_fs
        else:
            meta['recording_length_seconds'] = 'unkown'
    else:
        file_size = meta['highpass']['binary_byte_size']
        item_size = np.dtype(meta['highpass']['datatype']).itemsize
        nChans = meta['highpass']['n_channels_binaryfile']
        meta['recording_length_seconds'] = file_size/item_size/nChans/high_fs

    return meta

def get_waveforms(dp, u, n_waveforms=100, t_waveforms=82, selection='regular', periods='all',
                  spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
                  whiten=0, med_sub=0, hpfilt=0, hpfiltf=300,
                  nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=True, verbose=False,
                  med_sub_in_time=True, return_corrupt_mask=False, again=False,
                  cache_results=True, cache_path=None):

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