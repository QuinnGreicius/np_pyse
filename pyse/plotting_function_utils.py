import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from matplotlib import gridspec
from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange

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

def plot_psth(grouping, event_dict, segment_colors, segment_labels=None, ax=None, title=None, bin_size=10, window_size=50, default_color='black', min_time=None, max_time=None, pre_pad=500, 
              collapsed=False, color_preceding=False, color_proceeding=False, non_signficiant_alpha=0.2, use_signficance_mask=False, y_label_title=True, color_outside_bounds=False, **kwargs):
    

    event_data = event_dict['event_data']
    group_boundaries = event_dict['group_boundaries']
    ylabels = event_dict['y_tick_labels']
    group_indices = [0] + group_boundaries

    def process_segment_times(segment_times, group_boundaries):
        if segment_times is not None:
            if isinstance(segment_times[0], (np.ndarray, list)):
                return find_median_by_group(group_boundaries, segment_times)
            else:
                segment_times = find_median_by_group(group_boundaries, np.array(segment_times, dtype="float32")[:, None])
                return segment_times.squeeze()
        return None
    
    processed_seg_times = {}
    for key in event_dict.keys():
        if 'onset' in key.lower() or 'offset' in key.lower():
            processed_seg_times[key] = process_segment_times(event_dict[key], group_boundaries)
    
    if isinstance(prodOn_segment_times[0], (np.ndarray, list)):
        prodOn_segment_times = [np.append(on, off[-1]) for on, off in zip(prodOn_segment_times, prodOff_segment_times)]
    else:
        prodOn_segment_times = np.array(list(zip(prodOn_segment_times, prodOff_segment_times)))
    
    if prevProdOff_segment_times is not None and nextProdOn_segment_times is not None:
        if isinstance(prevProdOff_segment_times[0], (np.ndarray, list)):
            prevProdOff_segment_times = [np.append(on, off[-1]) for on, off in zip(prevProdOff_segment_times, nextProdOn_segment_times)]
        else:
            prevProdOff_segment_times = np.array(list(zip(prevProdOff_segment_times, nextProdOn_segment_times)))
    
    # Get the min prodOn_segment_times and max prodOff_segment_times
    if prodOn_segment_times is not None:
        psth_min_time = np.min([np.min(on) for on in prodOn_segment_times])
    if prodOff_segment_times is not None:
        psth_max_time = np.max([np.max(off) for off in prodOff_segment_times])
    if prevProdOff_segment_times is not None:
        plot_min_time = min(psth_min_time, np.min([np.min(on) for on in prevProdOff_segment_times]))
        plot_min = min(psth_min_time, np.max([np.min(off) for off in prevProdOff_segment_times]))
    if nextProdOn_segment_times is not None:
        plot_max_time = max(psth_max_time, np.max([np.max(off) for off in nextProdOn_segment_times]))
        plot_max = max(psth_max_time, np.median([np.max(on) for on in nextProdOn_segment_times]))

    psth_max_time = (psth_max_time // pre_pad + 1) * pre_pad
    psth_min_time = (psth_min_time // pre_pad - 1) * pre_pad

    plot_min_time = (plot_min_time // pre_pad - 1) * pre_pad if plot_min_time is not None else psth_min_time
    plot_max_time = (plot_max_time // pre_pad + 1) * pre_pad if plot_max_time is not None else psth_max_time
    
    plot_min = (plot_min // pre_pad - 1) * pre_pad if plot_min is not None else plot_min_time
    plot_max = (plot_max // pre_pad + 1) * pre_pad if plot_max is not None else plot_max_time

    if ax is None:
        if collapsed:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig, ax = plt.subplots(len(group_boundaries), 1, figsize=(12, 18), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.4)  # Add spacing between subplots
    else:
        fig = ax.get_figure()
        if not collapsed:
            ss = ax.get_subplotspec()
            ax.remove()
            inner_gs = gridspec.GridSpecFromSubplotSpec(len(group_boundaries), 1, subplot_spec=ss, hspace=0.4)  # Add spacing between subplots
            ax = [fig.add_subplot(inner_gs[i, 0]) for i in range(len(group_boundaries))]
            [plt.setp(ax[i].get_xticklabels(), visible=False) for i in range(len(group_boundaries)-1)]

    #Get segment labels as empty strings with the same dimensions as segment_colors
    if segment_labels is None:
        if isinstance(segment_colors[0], (list, np.ndarray)):
            segment_labels = [['' for _ in range(len(colors))] for colors in segment_colors]
        else:
            segment_labels = ['' for _ in range(len(segment_colors))]
    
    psths = []
    sems = []

    bins = np.arange(plot_min_time, plot_max_time, bin_size)
    time = (bins[:-1] + bins[1:]) / 2    

    for start, end in zip(group_indices[:-1], group_indices[1:]):
        spike_counts = np.array([np.histogram(spike_array, bins=bins)[0] for spike_array in event_data[start:end]])
        spike_rates = spike_counts / (bin_size / 1000)  # Convert to Hz
        smoothed_psth = gaussian_filter1d(spike_rates, sigma=window_size / bin_size, axis=1)
        psths.append(np.mean(smoothed_psth, axis=0))
        sems.append(np.std(smoothed_psth, axis=0) / np.sqrt(spike_counts.shape[0]))

    if use_signficance_mask:
        # Concatenate all PSTHs and get mean/std
        all_psths = np.concatenate(psths)
        mean_psth = np.mean(all_psths)
        std_psth = np.std(all_psths)
        
        # Create significance masks for each group's PSTH
        significance_mask = []
        for psth in psths:
            # Points are significant if they deviate more than 2 std from mean
            mask = np.abs(psth - mean_psth) > 2 * std_psth
            significance_mask.append(mask)

    psth_max = 1.1 * (np.max(np.array(psths)) + np.max(np.array(sems)))

    for i in range(len(group_indices)-1):
        start, end = group_indices[i], group_indices[i+1]
        colors = segment_colors[i].copy()
        labels = np.array(segment_labels[i].copy())
        psth = psths[i]
        sem = sems[i]
        axes = ax[i] if not collapsed else ax

        if prodOn_segment_times is not None:
            if isinstance(prodOn_segment_times[0], (np.ndarray, list)):
                group_segment_times = np.median(prodOn_segment_times[start:end], axis=0)
            else:
                group_segment_times = prodOn_segment_times[start:end]
            prev_idx = 0
            
            # Compute significant_psth and significant_sem once for the current group
            if use_signficance_mask:
                significant_psth = np.where(significance_mask[i], psth, np.nan)
                significant_sem = np.where(significance_mask[i], sem, np.nan)

            proceeding_to_be_colored = False
            preceeding_to_be_colored = False
            preceeding_text_x = None
            proceeding_text_x = None
            for j, seg_time in enumerate(group_segment_times):
                if seg_time < np.min(time) or seg_time > np.max(time):
                    continue
                seg_idx = np.argmin(np.abs(time - seg_time))
                if color_outside_bounds:
                    if j == 0:
                        c = colors[j]
                    elif j-1 > len(colors):
                        c = colors[-1]
                    else:
                        c = colors[j-1]
                else:
                    c = default_color if (j==0) or (j-1 > len(colors)) else colors[j-1] 
                text = '' if (j==0) or (j-1 > len(labels)) or collapsed else labels[j-1]
                next_color = colors[j] if j < len(colors) else default_color
                text_x = np.mean([time[prev_idx], time[seg_idx]])
                following_idx = np.argmin(np.abs(time - group_segment_times[j+1])) if j+1 < len(group_segment_times) and group_segment_times[j+1] < np.max(time) else None

                # Plot with low alpha for non-significant data
                axes.plot(time[prev_idx:seg_idx], psth[prev_idx:seg_idx], color=c, linewidth=2, alpha=0.1)  # Low alpha for the initial plot
                axes.fill_between(time[prev_idx:seg_idx], psth[prev_idx:seg_idx] - sem[prev_idx:seg_idx], psth[prev_idx:seg_idx] + sem[prev_idx:seg_idx], color=c, alpha=0.1)  # Low alpha for shaded region
                
                if use_signficance_mask:
                    min_time_idx = np.argmin(np.abs(time - psth_min_time))
                    max_time_idx = np.argmin(np.abs(time - psth_max_time))
                    start_idx = np.max([prev_idx, min_time_idx])
                    end_idx = np.min([seg_idx, max_time_idx])
                    if np.sum(significance_mask[i][start_idx:end_idx]) >= 0.05 * (end_idx - start_idx):
                        text_alpha = 1
                        vline_alpha = 1
                        proceeding_to_be_colored = color_proceeding
                        preceeding_to_be_colored = color_preceding
                    else:
                        text_alpha = non_signficiant_alpha
                        vline_alpha = non_signficiant_alpha
                        proceeding_to_be_colored = False
                        preceeding_to_be_colored = False

                    if preceeding_to_be_colored:
                        preceeding_text = '' if (j==0) or (j-2 > len(labels)) or collapsed else labels[j-2]
                        previous_color = default_color if (j==0) or (j-2 > len(colors)) else colors[j-2]
                        axes.text(preceeding_text_x, psth_max, preceeding_text, color=previous_color, ha='center', va='center', alpha=text_alpha)
                    if proceeding_to_be_colored and following_idx is not None:
                        proceeding_text_x = np.mean([time[seg_idx], time[following_idx]])
                        next_text = '' if (j > len(labels)) or collapsed else labels[j]
                        next_color = default_color if (j==0) or (j > len(colors)) or collapsed else colors[j]
                        axes.text(proceeding_text_x, psth_max, next_text, color=next_color, ha='center', va='center', alpha=text_alpha)
                    axes.text(text_x, psth_max, text, color=c, ha='center', va='center', alpha=text_alpha)
                    if not collapsed:
                        if prev_idx != 0:
                            axes.axvline(time[prev_idx], color=c, linestyle='--', linewidth=1.2, alpha=vline_alpha)
                        axes.axvline(time[seg_idx], color=next_color, linestyle='--', linewidth=1.2, alpha=vline_alpha)
                    axes.plot(time[start_idx:end_idx], significant_psth[start_idx:end_idx], color=c, linewidth=2, alpha=1.0)  # Higher alpha for significant plot
                    axes.fill_between(time[start_idx:end_idx], significant_psth[start_idx:end_idx] - significant_sem[start_idx:end_idx], 
                                       significant_psth[start_idx:end_idx] + significant_sem[start_idx:end_idx], color=c, alpha=0.4)  # Higher alpha for shaded region
    
                    # Plot black line segment under significant range
                    significant_range = ~np.isnan(significant_psth[start_idx:end_idx])
                    if np.sum(significant_range) >= 0.05 * (end_idx - start_idx):
                        axes.plot(time[start_idx:end_idx][significant_range], [1]*np.sum(significant_range), 
                                 color='black', linewidth=2, alpha=1)
                else:
                    if not collapsed:
                        axes.text(text_x, psth_max, text, color=c, ha='center', va='center', alpha=1.0)
                        axes.axvline(time[seg_idx], color=next_color, linestyle='--', linewidth=1.2, alpha=1.0)
                    axes.plot(time[prev_idx:seg_idx], psth[prev_idx:seg_idx], color=c, linewidth=2, alpha=1.0)
                    axes.fill_between(time[prev_idx:seg_idx], psth[prev_idx:seg_idx] - sem[prev_idx:seg_idx], psth[prev_idx:seg_idx] + sem[prev_idx:seg_idx], color=c, alpha=0.4)
                proceeding_text_x = text_x
                prev_idx = seg_idx
            c = colors[-1] if color_outside_bounds else default_color
            axes.plot(time[prev_idx:], psth[prev_idx:], color=c, linewidth=2, alpha=non_signficiant_alpha)
            axes.fill_between(time[prev_idx:], psth[prev_idx:] - sem[prev_idx:], psth[prev_idx:] + sem[prev_idx:], color=c, alpha=non_signficiant_alpha)
        else:
            axes.plot(time, psth, color=default_color, linewidth=2, alpha=1)

        # Plot the previous and next production onsets and color everything before and after grey
        if prevProdOff_segment_times is not None:
            if isinstance(prevProdOff_segment_times[0], (np.ndarray, list)):
                group_segment_times = np.median(prevProdOff_segment_times[start:end], axis=0)
            else:
                group_segment_times = prevProdOff_segment_times[start:end]

            for j, seg_time in enumerate(group_segment_times):
                if seg_time < np.min(time) or seg_time > np.max(time):
                    continue
                seg_idx = np.argmin(np.abs(time - seg_time))
                if not collapsed:
                    axes.axvline(time[seg_idx], color=default_color, linestyle='--', linewidth=1.2, alpha=non_signficiant_alpha)
                    # Create a grey shaded area before the line to infinity
                    if j == 0:
                        axes.fill_between(time[:seg_idx], 0, 1, transform=axes.get_xaxis_transform(), color='grey', alpha=0.5)
                    elif j == len(group_segment_times) - 1:
                        axes.fill_between(time[seg_idx:], 0, 1, transform=axes.get_xaxis_transform(), color='grey', alpha=0.5)
        for prodEvent in prod_segment_times:
            if isinstance(prodEvent[0], (np.ndarray, list)):
                group_segment_times = np.median(prodEvent[start:end], axis=0)
            else:
                group_segment_times = prodEvent[start:end]
            for j, seg_time in enumerate(group_segment_times):
                if seg_time < np.min(time) or seg_time > np.max(time):
                    continue
                seg_idx = np.argmin(np.abs(time - seg_time))
                if not collapsed:
                    axes.axvline(time[seg_idx], color=default_color, linestyle='--', linewidth=1.2, alpha=non_signficiant_alpha)

        if not collapsed:
            if i == 0:
                ax[i].spines[['left', 'bottom']].set_visible(False)
                ax[i].set_xticks([])   
                ax[i].tick_params(axis='x', which='both', bottom=False, top=False)
            elif i < len(group_indices)-2:
                ax[i].spines[['top', 'left', 'bottom']].set_visible(False)
                ax[i].set_xticks([])    
                ax[i].tick_params(axis='x', which='both', bottom=False, top=False)
            else:
                ax[i].spines[['left', 'top']].set_visible(False)
            if y_label_title:
                ax[i].set_ylabel(ylabels[i], rotation=0, labelpad=0, ha='right', va='center')
            else:
                # Format it such that long sentences are split up half way and the justification is left and right
                sent_split = ylabels[i].split(' ') # Split the sentence into two lines if nwords > 5
                nwords = len(sent_split)
                if nwords > 5:
                    split_idx = nwords // 2
                    sent_split = ' '.join(sent_split[:split_idx]) + '\n' + ' '.join(sent_split[split_idx:])
                else:
                    sent_split = ylabels[i]
                ax[i].set_ylabel(sent_split, fontsize=plt.rcParams['font.size'], rotation=0, labelpad=0, ha='right', va='center')
            if i == (len(group_indices)-1)//2:
                ax_right = ax[i].twinx()
                ax_right.set_yticks([])
                ax_right.set_xticks([])
                ax_right.spines[['left', 'top', 'bottom', 'right']].set_visible(False)
                ax_right.set_ylabel('Firing Rate (Hz)', labelpad=30)
                ax_right.yaxis.set_label_position('right')
            ax[i].yaxis.set_ticks_position('right')
        else:
            axes.set_ylabel('Firing Rate (Hz)', labelpad=30)

    if collapsed:
        med_prodOns = np.median(process_segment_times(prodOns, [len(prodOns)]).flatten())
        med_prodOffs = np.median(process_segment_times(prodOffs, [len(prodOffs)]).flatten())
        med_prevProdOffs = np.median(process_segment_times(prevProdOffs, [len(prevProdOffs)]).flatten())
        med_nextProdOns = np.median(process_segment_times(nextProdOns, [len(nextProdOns)]).flatten())
        med_prodEvents = [np.median(process_segment_times(event_dict[key], [len(event_dict[key])]).flatten()) for key in prod_keys]

        med_prodOns_idx = np.argmin(np.abs(time - med_prodOns))
        med_prodOffs_idx = np.argmin(np.abs(time - med_prodOffs))
        med_prevProdOffs_idx = np.argmin(np.abs(time - med_prevProdOffs))
        med_nextProdOns_idx = np.argmin(np.abs(time - med_nextProdOns))
        med_prodEvents_idx = [np.argmin(np.abs(time - med_event)) for med_event in med_prodEvents]

        ax.axvline(x=time[med_prodOns_idx], color='black', linestyle='--', linewidth=1.2, alpha=1)
        ax.axvline(x=time[med_prodOffs_idx], color='black', linestyle='--', linewidth=1.2, alpha=1)
        ax.axvline(x=time[med_prevProdOffs_idx], color='black', linestyle='--', linewidth=1.2, alpha=1)
        ax.axvline(x=time[med_nextProdOns_idx], color='black', linestyle='--', linewidth=1.2, alpha=1)
        for med_event_idx in med_prodEvents_idx:
            ax.axvline(x=time[med_event_idx], color='black', linestyle='--', linewidth=1.2, alpha=1)
        ax.fill_between(time[:med_prevProdOffs_idx], 0, 1, transform=ax.get_xaxis_transform(), color='grey', alpha=0.5)
        ax.fill_between(time[med_nextProdOns_idx:], 0, 1, transform=ax.get_xaxis_transform(), color='grey', alpha=0.5)
    
    xlim_lower = plot_min if min_time is None else min_time
    xlim_upper = plot_max if max_time is None else max_time

    xlabels = np.arange(xlim_lower, xlim_upper, pre_pad)
    axes = ax if collapsed else ax[-1]
    axes.set_xlabel('Time (ms)')
    axes.set_xticks(xlabels)
    axes.set_xticklabels([int(x) for x in xlabels])
    axes.set_xlim(xlim_lower, xlim_upper) # Set the x-axis limits
    axes = ax if collapsed else ax[0]
    axes.set_title(title if title else grouping)
    
    # Get the max psth value and set it for all axes
    if collapsed:
        max_psth = psth_max / 1.1 * 1.5
        ax.set_ylim(0, max_psth)
        ax.set_xlim(xlim_lower, xlim_upper)
    else:
        max_psth = max([np.max(ax[i].get_ylim()) for i in range(len(group_boundaries))]) * 1.5
        [ax[i].set_ylim(0, max_psth) for i in range(len(group_boundaries))]
        [ax[i].set_xlim(xlim_lower, xlim_upper) for i in range(len(group_boundaries))]

    return fig, ax


        


def add_legend(ax, groups, color_dicts, ncols, titles, location, sorting_key=None):
    if not isinstance(color_dicts, list):
        color_dicts = [color_dicts]
        groups = [groups]
        ncols = [ncols]
        titles = [titles]

    if location not in ['right', 'top', 'bottom', 'left', 'center']:
        raise ValueError("location must be one of 'right', 'top', 'bottom', 'left', or 'center'")

    legends = []
    fig = ax.figure  # Get the figure object to adjust layout

    # Adjust figure size to create space for legends
    if location == 'right':
        fig.subplots_adjust(right=0.75)  # Shrink plot area
        base_x, base_y, offset = 1.1, 0.5, 0.2  # Move vertically
    elif location == 'left':
        fig.subplots_adjust(left=0.3)
        base_x, base_y, offset = -0.4, 0.5, 0.2  # Move vertically
    elif location == 'top':
        fig.subplots_adjust(top=0.8)
        base_x, base_y, offset = 0.5, 1.2, 0.3  # Move horizontally
    elif location == 'bottom':
        fig.subplots_adjust(bottom=0.3)
        base_x, base_y, offset = 0.5, -0.2, 0.3  # Move horizontally
    elif location == 'center':
        base_x, base_y, offset = 0.5, 0.5, 0.5  # Move vertically
        ax.axis('off')  # Hide the axis for center legend

    for i, (group, cd, ncol, title) in enumerate(zip(groups, color_dicts, ncols, titles)):
        handles = [mpatches.Rectangle((0, 0), 1, 1, color=cd[g], label=g) 
           for g in sorted(group, key=sorting_key)]
        labels = [g for g in sorted(group, key=sorting_key)]
        legend_kwargs = {}
        if location in ['right', 'left']:
            legend_y = base_y + (i - (len(groups) - 1) / 2) * offset  # Spread vertically
            legend_x = base_x
        elif location in ['top', 'bottom']:
            legend_x = base_x + (i - (len(groups) - 1) / 2) * offset  # Spread horizontally
            legend_y = base_y
        elif location == 'center':
            legend_x = base_x
            legend_y = base_y + (i - (len(groups) - 1) / 2) * offset
            # Add kwargs to increase the size of the legend text to title size
            # Get title font size from rcparams
            legend_kwargs = {'prop': {'size': 'large'}}

        loc_map = {'right': 'center left', 'left': 'center right', 'top': 'lower center', 'bottom': 'upper center', 'center': 'center'}
        legend = ax.legend(handles, labels, bbox_to_anchor=(legend_x, legend_y), loc=loc_map[location], ncol=ncol, title=title, **legend_kwargs)
        ax.add_artist(legend)
        legends.append(legend)

    return legends[0] if len(legends) == 1 else legends


