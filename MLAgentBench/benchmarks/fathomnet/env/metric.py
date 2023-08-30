import gc
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass

    
def apk(actual, predicted, k=20):
    """
    Compute the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=20):
    """
    Compute the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def prepare(df, out_of_sample_column_name):
    df['categories'] = df['categories'].str.split(' ')
    df[out_of_sample_column_name] = df[out_of_sample_column_name].astype(float)
    return df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, out_of_sample_column_name: str='osd', k: int=20) -> float:
    """Metric for the FathomNet 2023 FGVC competition (46149).

    Computes the average of a MAP@k and a normalized AUC on an "out-of-sample" indicator.

    Parameters
    ----------
    solution : DataFrame with columns having for each instance:
        - categories: a list of integer categories
        - osd: a binary out-of-sample indicator
    submission : array-like of float, shape = (n_samples, n_classes + 1)
    out_of_sample_column_name: str, the name of the out-of-sample indicator
    k: the maximum number of predicted categories
    """
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError('Submission file missing expected column ' + row_id_column_name)
    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError('Solution file missing expected column ' + row_id_column_name)
    solution = solution.sort_values(by=[row_id_column_name])
    submission = submission.sort_values(by=[row_id_column_name])
    if not (solution[row_id_column_name].values == submission[row_id_column_name].values).all():
        raise ParticipantVisibleError('The solution and submission row IDs are not identical')
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    gc.collect()

    if out_of_sample_column_name is None:
        raise ParticipantVisibleError('out_of_sample_column_name cannot be None')
    missing_cols = solution.columns.difference(submission.columns)
    if len(missing_cols) > 0:
        raise ParticipantVisibleError('Submission file missing expected columns ' + ', '.join(missing_cols))

    solution, submission = prepare(solution, out_of_sample_column_name), prepare(submission, out_of_sample_column_name)

    oos_true = solution.pop(out_of_sample_column_name).to_numpy()
    oos_pred = submission.pop(out_of_sample_column_name).to_numpy()
    oos_score = sklearn.metrics.roc_auc_score(oos_true, oos_pred)
    normalized_oos_score = 2 * (oos_score - 0.5)  # random AUC is 0.5

    solution = solution.squeeze().to_list()
    submission = submission.squeeze().to_list()
    cat_score = mapk(solution, submission, k=k)
    results = 0.5 * (normalized_oos_score + cat_score)
    return results