"""
Utilities for loading, preprocessing, and filtering CelebA metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_attributes(filepath: str = "../data/list_attr_celeba.csv") -> pd.DataFrame:
    """Load the 40 binary attributes and convert -1/1 to 0/1."""
    df = pd.read_csv(filepath)
    attr_cols = df.columns[1:]  # Skip image_id
    df[attr_cols] = (df[attr_cols] + 1) / 2
    return df


def load_landmarks(filepath: str = "../data/list_landmarks_align_celeba.csv") -> pd.DataFrame:
    """Load facial landmark coordinates."""
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def get_landmark_features(landmarks: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from landmarks (eye distance, head tilt, etc.).

    Args:
        landmarks: DataFrame with landmark columns (lefteye_x, lefteye_y, …)

    Returns:
        Copy of input with extra feature columns appended.
    """
    df = landmarks.copy()

    df['eye_distance'] = np.sqrt(
        (df['righteye_x'] - df['lefteye_x'])**2 +
        (df['righteye_y'] - df['lefteye_y'])**2
    )

    df['eye_center_x'] = (df['lefteye_x'] + df['righteye_x']) / 2
    df['eye_center_y'] = (df['lefteye_y'] + df['righteye_y']) / 2

    df['mouth_center_x'] = (df['leftmouth_x'] + df['rightmouth_x']) / 2
    df['mouth_center_y'] = (df['leftmouth_y'] + df['rightmouth_y']) / 2

    df['mouth_width'] = np.sqrt(
        (df['rightmouth_x'] - df['leftmouth_x'])**2 +
        (df['rightmouth_y'] - df['leftmouth_y'])**2
    )

    df['face_length'] = np.sqrt(
        (df['mouth_center_x'] - df['eye_center_x'])**2 +
        (df['mouth_center_y'] - df['eye_center_y'])**2
    )

    df['nose_to_eye_dist'] = np.sqrt(
        (df['nose_x'] - df['eye_center_x'])**2 +
        (df['nose_y'] - df['eye_center_y'])**2
    )

    df['face_aspect_ratio'] = df['eye_distance'] / df['face_length']
    df['mouth_eye_ratio'] = df['mouth_width'] / df['eye_distance']

    df['head_tilt'] = np.arctan2(
        df['righteye_y'] - df['lefteye_y'],
        df['righteye_x'] - df['lefteye_x']
    ) * 180 / np.pi

    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def compute_attribute_popularity(df: pd.DataFrame, attr_cols: List[str]) -> pd.DataFrame:
    """Compute popularity (percentage) for each attribute."""
    summary = pd.DataFrame({
        'attribute': attr_cols,
        'count': [df[col].sum() for col in attr_cols],
        'percentage': [df[col].mean() * 100 for col in attr_cols]
    })
    return summary.sort_values('percentage', ascending=False).reset_index(drop=True)


def get_unpopular_attributes(
    df: pd.DataFrame,
    attr_cols: List[str],
    threshold: float = 40.0
) -> List[str]:
    """Get list of attributes below popularity threshold."""
    popularity = compute_attribute_popularity(df, attr_cols)
    return popularity[popularity['percentage'] < threshold]['attribute'].tolist()


def filter_by_attribute_popularity(
    df: pd.DataFrame,
    attr_cols: List[str],
    threshold: float = 40.0
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove samples that have any unpopular attribute (below threshold)."""
    unpopular = get_unpopular_attributes(df, attr_cols, threshold)

    if not unpopular:
        return df.copy(), []

    mask = df[unpopular].sum(axis=1) == 0
    return df[mask].copy(), unpopular


def filter_by_head_tilt(
    df: pd.DataFrame,
    tolerance: float = 1.0,
    head_tilt_col: str = 'head_tilt'
) -> pd.DataFrame:
    """Filter for samples with head tilt near zero."""
    mask = df[head_tilt_col].abs() <= tolerance
    return df[mask].copy()


def filter_by_face_aspect_ratio(
    df: pd.DataFrame,
    min_ratio: float = 0.9,
    face_aspect_ratio_col: str = 'face_aspect_ratio'
) -> pd.DataFrame:
    """Filter for samples with face_aspect_ratio above min_ratio."""
    mask = df[face_aspect_ratio_col] >= min_ratio
    return df[mask].copy()


def filter_dataset(
    df: pd.DataFrame,
    attr_cols: List[str],
    popularity_threshold: float = 40.0,
    head_tilt_tolerance: Optional[float] = None,
    head_tilt_col: str = 'head_tilt',
    face_aspect_ratio_min: Optional[float] = None,
    face_aspect_ratio_col: str = 'face_aspect_ratio'
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply all filters to dataset.

    Returns:
        Tuple of (filtered DataFrame, stats dict)
    """
    stats = {'original_count': len(df)}

    filtered, unpopular = filter_by_attribute_popularity(
        df, attr_cols, popularity_threshold
    )
    stats['after_attr_filter'] = len(filtered)
    stats['unpopular_attributes'] = unpopular

    if head_tilt_tolerance is not None:
        filtered = filter_by_head_tilt(
            filtered, head_tilt_tolerance, head_tilt_col
        )
        stats['after_tilt_filter'] = len(filtered)
        stats['head_tilt_tolerance'] = head_tilt_tolerance

    if face_aspect_ratio_min is not None:
        filtered = filter_by_face_aspect_ratio(
            filtered, face_aspect_ratio_min, face_aspect_ratio_col
        )
        stats['after_aspect_ratio_filter'] = len(filtered)
        stats['face_aspect_ratio_min'] = face_aspect_ratio_min

    filtered = filtered.drop(columns=unpopular, errors='ignore')

    stats['final_count'] = len(filtered)
    return filtered, stats
