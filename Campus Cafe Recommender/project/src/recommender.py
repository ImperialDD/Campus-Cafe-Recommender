"""Nearest-neighbor recommendation in PCA space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Fixed category split (matches the dataset feature order).
FOOD_FEATURES: list[str] = ["Misal Pav", "Samosa Pav", "Vada Pav", "Pav Bhaji", "Sandwich"]
DRINK_FEATURES: list[str] = ["Sprite", "Diet Coke", "Coca Cola", "Thums Up", "Lassi"]


@dataclass
class Neighbor:
    name: str
    distance: float
    pc1: float
    pc2: float


@dataclass
class Recommendation:
    item: str
    score: float              # average rating from neighbors
    reason: str
    supporting_students: list[str]


@dataclass
class ComboRecommendation:
    snack: Recommendation
    drink: Recommendation
    neighbor_ids: list[str]
    neighbor_explanation: str


def calculate_distances(
    new_user_coords: np.ndarray, student_coords: np.ndarray, student_ids: list[str]
) -> list[Neighbor]:
    """Euclidean distance in (PC1, PC2) space from the new user to every student."""
    distances = np.linalg.norm(student_coords - new_user_coords, axis=1)
    neighbors = [
        Neighbor(
            name=sid,
            distance=float(d),
            pc1=float(student_coords[i, 0]),
            pc2=float(student_coords[i, 1]),
        )
        for i, (sid, d) in enumerate(zip(student_ids, distances))
    ]
    return sorted(neighbors, key=lambda n: n.distance)


def _rank_category(
    new_user_ratings: np.ndarray,
    neighbor_observed: pd.DataFrame,
    neighbors: list[Neighbor],
    feature_names: list[str],
    category_features: list[str],
    min_rating: int = 4,
) -> list[Recommendation]:
    """Rank every candidate item in a single category (food OR drink).

    Uses only observed neighbor ratings. Items the user rated below
    `min_rating` (and that neighbors like) are ranked ahead of items the
    user already rates highly; within each group items are ordered by
    descending neighbor average. This full ranking is what lets the app
    step through the 2nd-best, 3rd-best, ... combo without any randomness.
    """
    top_ids = [n.name for n in neighbors]
    cat_observed = neighbor_observed.loc[
        neighbor_observed.index.intersection(top_ids), category_features
    ]

    candidates: list[tuple[str, float, list[str], int]] = []
    fallback: list[tuple[str, float, list[str], int]] = []

    for item in category_features:
        col = cat_observed[item].dropna()
        if col.empty:
            continue
        avg = float(col.mean())
        supporters = col[col >= 4].index.tolist()
        idx = feature_names.index(item)
        user_rating = new_user_ratings[idx]
        user_low = np.isnan(user_rating) or user_rating < min_rating
        row = (item, avg, supporters, idx)
        if user_low and avg >= 3.0:
            candidates.append(row)
        else:
            fallback.append(row)

    candidates.sort(key=lambda r: r[1], reverse=True)
    fallback.sort(key=lambda r: r[1], reverse=True)
    pool = candidates + fallback

    top_ids_str = ", ".join(top_ids)
    ranked: list[Recommendation] = []
    for item, avg, supporters, _ in pool:
        n_ratings = len(cat_observed[item].dropna())
        reason = (
            f"Your PCA profile is closest to {top_ids_str}. "
            f"These similar students rate {item} highly "
            f"(average {avg:.2f}/5 from {n_ratings} ratings)."
        )
        ranked.append(
            Recommendation(item=item, score=avg, reason=reason, supporting_students=supporters)
        )
    return ranked


def _best_in_category(
    new_user_ratings: np.ndarray,
    neighbor_observed: pd.DataFrame,
    neighbors: list[Neighbor],
    feature_names: list[str],
    category_features: list[str],
    min_rating: int = 4,
) -> Recommendation | None:
    """Pick the single best recommendation from a category (food OR drink)."""
    ranked = _rank_category(
        new_user_ratings, neighbor_observed, neighbors, feature_names,
        category_features, min_rating,
    )
    return ranked[0] if ranked else None


def generate_combo_recommendation(
    new_user_ratings: np.ndarray,
    observed_df: pd.DataFrame,
    neighbors: list[Neighbor],
    feature_names: list[str],
    min_rating: int = 4,
) -> ComboRecommendation:
    """Generate exactly ONE snack + ONE drink from the nearest neighbors.

    Food and drink candidates are scored separately using only observed
    neighbor ratings, then combined into a single combo.
    """
    top_ids = [n.name for n in neighbors]
    neighbor_observed = observed_df[
        observed_df["Respondent"].isin(top_ids)
    ].set_index("Respondent")

    snack = _best_in_category(
        new_user_ratings, neighbor_observed, neighbors,
        feature_names, FOOD_FEATURES, min_rating,
    )
    drink = _best_in_category(
        new_user_ratings, neighbor_observed, neighbors,
        feature_names, DRINK_FEATURES, min_rating,
    )

    neighbor_explanation = (
        f"Your PCA taste profile is closest to {', '.join(top_ids)}."
    )

    return ComboRecommendation(
        snack=snack,
        drink=drink,
        neighbor_ids=top_ids,
        neighbor_explanation=neighbor_explanation,
    )


def generate_all_combos(
    new_user_ratings: np.ndarray,
    observed_df: pd.DataFrame,
    neighbors: list[Neighbor],
    feature_names: list[str],
    min_rating: int = 4,
) -> list[ComboRecommendation]:
    """Rank every possible (snack, drink) combo from the nearest neighbors.

    Builds on the same per-category ranking used for the single best combo,
    then pairs every ranked snack with every ranked drink and orders the
    resulting combos by combined neighbor-approval score. This gives a
    deterministic, data-driven sequence of distinct combos — combo #1 is the
    single best combo (identical to `generate_combo_recommendation`), combo
    #2 is the next-best distinct pairing, and so on. Nothing here is random
    or hard-coded: the ranking flows entirely from the PCA neighbor ratings.
    """
    top_ids = [n.name for n in neighbors]
    neighbor_observed = observed_df[
        observed_df["Respondent"].isin(top_ids)
    ].set_index("Respondent")

    ranked_snacks = _rank_category(
        new_user_ratings, neighbor_observed, neighbors,
        feature_names, FOOD_FEATURES, min_rating,
    )
    ranked_drinks = _rank_category(
        new_user_ratings, neighbor_observed, neighbors,
        feature_names, DRINK_FEATURES, min_rating,
    )

    neighbor_explanation = (
        f"Your PCA taste profile is closest to {', '.join(top_ids)}."
    )

    if not ranked_snacks or not ranked_drinks:
        # At least one full combo is still returned when one side is empty,
        # so the page can show "No snack/drink candidate found" as before.
        snack = ranked_snacks[0] if ranked_snacks else None
        drink = ranked_drinks[0] if ranked_drinks else None
        return [
            ComboRecommendation(
                snack=snack, drink=drink, neighbor_ids=top_ids,
                neighbor_explanation=neighbor_explanation,
            )
        ]

    combos: list[tuple[float, int, int]] = []
    for si, snack in enumerate(ranked_snacks):
        for di, drink in enumerate(ranked_drinks):
            combos.append((snack.score + drink.score, si, di))
    # Highest combined score first; ties broken by each item's own rank
    # position so the ordering stays fully deterministic.
    combos.sort(key=lambda c: (-c[0], c[1], c[2]))

    return [
        ComboRecommendation(
            snack=ranked_snacks[si],
            drink=ranked_drinks[di],
            neighbor_ids=top_ids,
            neighbor_explanation=neighbor_explanation,
        )
        for _, si, di in combos
    ]


def generate_recommendations(
    new_user_ratings: np.ndarray,
    observed_df: pd.DataFrame,
    neighbors: list[Neighbor],
    feature_names: list[str],
    k: int = 3,
    min_rating: int = 4,
) -> list[Recommendation]:
    """Legacy: generate a flat list of recommendations across all features.

    Kept for backward compatibility; the UI now uses generate_combo_recommendation.
    """
    k = min(k, len(neighbors))
    top = neighbors[:k]
    top_ids = [n.name for n in top]
    neighbor_observed = observed_df[observed_df["Respondent"].isin(top_ids)].set_index(
        "Respondent"
    )

    recs: list[Recommendation] = []
    for idx, item in enumerate(feature_names):
        user_rating = new_user_ratings[idx]
        col = neighbor_observed[item].dropna()
        if col.empty:
            continue
        avg = float(col.mean())
        user_low = np.isnan(user_rating) or user_rating < min_rating
        if user_low and avg >= 3.5:
            supporters = col[col >= 4].index.tolist()
            reason = (
                f"Your PCA profile is closest to {', '.join(top_ids)}. "
                f"These similar students rate {item} highly "
                f"(average {avg:.2f}/5 from {len(col)} ratings)."
            )
            recs.append(
                Recommendation(
                    item=item, score=avg, reason=reason, supporting_students=supporters,
                )
            )
    recs.sort(key=lambda r: r.score, reverse=True)
    return recs
