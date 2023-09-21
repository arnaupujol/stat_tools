#This module contains methods to produce tables

import pandas as pd
import numpy as np

def count_time_label(dataframe, time_label, label, time_bins = 5, \
                    mean_dates = False, show_total = False, ignore_nulls = True):
    """
    This method creates a pandas DataFrame with the value counts
    of a given label in different time bins.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        Dataframe for which we get the counts
    time_label: str
        Name of variable involving time values
    label: str
        Name of label to be selected
    time_bins: int or list
        Number of bins to cut the date times (int) or time bin edges (list)
    mean_dates: bool
        If True, the mean date per time bin is given instead of
        the time range
    show_total: bool
        If True, it shows the cumulated counts per label
    ignore_nulls: bool
        If True, null cases for any of the two labels are ignored

    Returns:
    --------
    table_sizes: pd.DataFrame
        The table with the counts for each label and time bin
    """
    if ignore_nulls:
        dataframe = dataframe[dataframe[label].notnull()]
    if type(time_bins) is int:
        out, time_edges = pd.cut(dataframe[time_label], time_bins, retbins=True)
    elif len(time_bins) > 1:
        time_edges = time_bins
        time_bins = len(time_edges) - 1
    else:
        if verboe:
            print("Error: incorrect assignment of time_bins (int or list): " + str(time_bins))
    table_sizes = {}
    for i in range(time_bins):
        mask = (dataframe[time_label] >= time_edges[i])&(dataframe[time_label] < time_edges[i + 1])
        if mean_dates:
            string = str(dataframe[time_label][mask].mean())[:10]
        else:
            string = str(time_edges[i])[:10] + ' - ' + str(time_edges[i+1])[:10]
        table_sizes[string] = dataframe[mask][label].value_counts().to_dict()
        if show_total:
            counts = dataframe[mask][label].value_counts()
            table_sizes[string]['Total'] = np.sum(counts[counts>0])
    if show_total:
        table_sizes['Total'] = dataframe[label].value_counts().to_dict()
        table_sizes['Total']['Total'] = dataframe[label].value_counts().sum()
    table_sizes = pd.DataFrame(table_sizes, dtype = int).T
    return table_sizes

def labels_2d(dataframe, label_1, label_2, show_total = False, ignore_nulls = True):
    """
    This method creates a pandas DataFrame with the value counts
    of two selected labels.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        Dataframe for which we get the counts
    label_1: str
        Name of first label to be selected
    label_2: str
        Name of second label to be selected
    show_total: bool
        If True, it shows the cumulated counts per label
    ignore_nulls: bool
        If True, null cases for any of the two labels are ignored

    Returns:
    --------
    table_sizes: pd.DataFrame
        The table with the counts for both labels
    """
    if ignore_nulls:
        mask = dataframe[label_1].notnull()&dataframe[label_2].notnull()
        dataframe = dataframe[mask]
    table_sizes = {}
    for i in dataframe[label_1].unique():
        mask = dataframe[label_1] == i
        table_sizes[i] = dataframe[mask][label_2].value_counts().to_dict()
        if show_total:
            counts = dataframe[mask][label_2].value_counts()
            table_sizes[i]['Total'] = np.sum(counts[counts>0])
    if show_total:
        table_sizes['Total'] = dataframe[label_2].value_counts().to_dict()
        table_sizes['Total']['Total'] = dataframe[label_2].value_counts().sum()
    table_sizes = pd.DataFrame(table_sizes).T
    table_sizes[table_sizes.isnull()] = 0
    return table_sizes

def labels_nd(dataframe, label_1, label_list, show_total = False, ignore_nulls = True):
    """
    This method creates a pandas DataFrame with the value counts
    of one label versus other selected labels.

    Parameters:
    -----------
    dataframe: pd.DataFrame
        Dataframe for which we get the counts
    label_1: str
        Name of first label to be selected
    label_list: list
        List of labels to be combined with label_1
    show_total: bool
        If True, it shows the cumulated counts per label
    ignore_nulls: bool
        If True, null cases for any of the two labels are ignored

    Returns:
    --------
    table_sizes: pd.DataFrame
        The table with the counts for all labels
    """
    for i, l in enumerate(label_list):
        if i == 0:
            table_sizes = labels_2d(dataframe, label_1, l, ignore_nulls=ignore_nulls, show_total=show_total)
            if show_total:
                table_sizes = table_sizes.rename(columns={'Total' : 'Total ' + l})
        else:
            table_sizes = pd.merge(table_sizes, labels_2d(dataframe, label_1, l, ignore_nulls=ignore_nulls, \
                                                             show_total=show_total), left_index = True, \
                               right_index = True, how = 'outer')
            if show_total:
                table_sizes = table_sizes.rename(columns={'Total' : 'Total ' + l})
    return table_sizes
