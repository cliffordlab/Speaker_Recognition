from scipy.stats import chi2_contingency
from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations
import numpy as np
import pandas as pd

# Uploaded by Moran Neuhof, 2018
# https://github.com/neuhofmo/chisq_test_wrapper/blob/master/chisq_test_wrapper.py
# https://neuhofmo.github.io/chi-square-and-post-hoc-in-python/

def get_asterisks_for_pval(p_val, alpha=0.05):
    """Receives the p-value and returns asterisks string."""
    if p_val > alpha:  # bigger than alpha
        p_text = "ns"
    # following the standards in biological publications
    elif p_val < 1e-4:  
        p_text = '****'
    elif p_val < 1e-3:
        p_text = '***'
    elif p_val < 1e-2:
        p_text = '**'
    else:
        p_text = '*'
    
    return p_text  # string of asterisks


def run_chisq_on_combination(df, combinations_tuple):
    """Receives a dataframe and a combinations tuple and returns p-value after performing chisq test."""
    assert len(combinations_tuple) == 2, "Combinations tuple is too long! Should be of size 2."
    new_df = df[(df.index == combinations_tuple[0]) | (df.index == combinations_tuple[1])]
    # import pdb
    # pdb.set_trace()
    chi2, p, dof, ex = chi2_contingency(new_df*100, correction=True)
    return p


def chisq_and_posthoc_corrected(df, correction_method='fdr_bh', alpha=0.05):
    """Receives a dataframe and performs chi2 test and then post hoc.
    Prints the p-values and corrected p-values (after FDR correction).
    alpha: optional threshold for rejection (default: 0.05)
    correction_method: method used for mutiple comparisons correction. (default: 'fdr_bh').
    See statsmodels.sandbox.stats.multicomp.multipletests for elaboration."""

    # start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(df, correction=True)
    print("Chi2 result of the contingency table: {}, p-value: {}\n".format(chi2, p))
    
    # post-hoc test
    all_combinations = list(combinations(df.index, 2))  # gathering all combinations for post-hoc chi2
    print("Post-hoc chi2 tests results:")
    p_vals = [run_chisq_on_combination(df, comb) for comb in all_combinations]  # a list of all p-values
    # the list is in the same order of all_combinations

    # correction for multiple testing
    reject_list, corrected_p_vals = multipletests(p_vals, method=correction_method, alpha=alpha)[:2]
    for p_val, corr_p_val, reject, comb in zip(p_vals, corrected_p_vals, reject_list, all_combinations):
        print("{}: p_value: {:5f}; corrected: {:5f} ({}) reject: {}".format(comb, p_val, corr_p_val, get_asterisks_for_pval(p_val, alpha), reject))


if __name__ == '__main__':
    acc_1 = np.load('../notebooks/saved_data_prev_2/total_acc_list_500_prev_1.npy')
    acc_2 = np.load('../notebooks/saved_data_prev_2/total_acc_list_500_prev_2.npy')
    acc_3 = np.load('../notebooks/saved_data_prev_2/total_acc_list_500_prev_3.npy')
    acc_4 = np.load('../notebooks/saved_data_prev_2/total_acc_list_500_prev_4.npy')
    acc_5 = np.load('../notebooks/saved_data_prev_2/total_acc_list_500_prev_5.npy')

    index = ["1 sec", "2 sec", "3 sec", "4 sec", "5 sec"]
    dfs=[pd.DataFrame()]*5
    for i in range(5):
        dfs[i]=pd.DataFrame(data=[acc_1[i], acc_2[i], acc_3[i], acc_4[i],  acc_5[i]], index = index)
    
        chisq_and_posthoc_corrected(dfs[i])
        