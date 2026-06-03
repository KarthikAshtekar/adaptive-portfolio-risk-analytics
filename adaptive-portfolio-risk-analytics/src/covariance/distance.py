import numpy as np
import pandas as pd


def compute_distance_matrix(
    correlation_matrix: pd.DataFrame
) -> pd.DataFrame:

    distance_matrix = np.sqrt(
        (1 - correlation_matrix) / 2
    )

    return pd.DataFrame(
        distance_matrix,
        index=correlation_matrix.index,
        columns=correlation_matrix.columns,
    )