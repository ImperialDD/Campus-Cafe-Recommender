# Campus Cafe Drinks & Snack Recommender Engine

A **Linear Algebra / PCA** mini-project for Second Year CSE (Data Science).

It learns taste patterns from campus-cafe survey ratings and recommends
drinks & snacks to a brand-new student using **Principal Component Analysis**
and **nearest-neighbor** matching — all implemented from scratch with NumPy.

## Tech stack
- Python · Streamlit (UI)
- Pandas (data handling)
- NumPy (PCA: covariance, eigendecomposition, projection)
- Plotly (2D taste map)

## Project structure
```
app.py                      Streamlit application (7 sections + math explainer)
data/
    survey_data.csv         Cleaned survey dataset (21 students × 10 items)
src/
    data_processing.py      load_data, clean_data, create_matrix, feature_means
    pca_analysis.py         mean_center, calculate_covariance, perform_pca,
                            calculate_explained_variance, project_students,
                            project_new_user
    recommender.py          calculate_distances, generate_recommendations
    visualization.py        plot_taste_map, plot_explained_variance
requirements.txt
README.md
```

## The mathematical pipeline
1. Build a **Student × Item** rating matrix.
2. **Mean-center** each feature.
3. Compute the **covariance matrix**.
4. **Eigendecompose** it (`np.linalg.eigh`).
5. **Sort** eigenvalues descending; reorder eigenvectors.
6. Extract **PC1 & PC2**.
7. Compute **explained variance** ratios.
8. **Project** existing students onto PC1–PC2.
9. Draw the **2D taste map**.
10. A **new user** rates the same items → center with *original* means,
    project with *original* PC1/PC2 (no retraining).
11. Find **nearest neighbors** (Euclidean distance in PCA space).
12. **Recommend** items the neighbors rated highly.

## Dataset
10 cafe items: Misal Pav, Samosa Pav, Vada Pav, Pav Bhaji, Sandwich,
Sprite, Diet Coke, Coca Cola, Thums Up, Lassi.

21 student responses. Blanks = the student did not rate that item
(treated as missing, **not** as 0). Missing values are imputed with the
feature's observed mean for PCA; the original missing entries are kept
separate and never claimed to be real ratings.

## Data cleaning
Free-response names are normalized to canonical names (e.g. `Wadapav` →
`Vada Pav`, `Coca-cola` → `Coca Cola`, `Spirit` → `Sprite`). The mappings
are documented in `src/data_processing.py`. Tokens like `.`, `NA`, and
`"There's nothing else"` are treated as missing.

## Running
```bash
pip install -r requirements.txt
streamlit run app.py
```

You can also upload your own survey CSV via the sidebar — it must have a
student-ID first column followed by item columns.

## Academic transparency
The app includes a **"How the Mathematics Works"** section explaining
mean, mean-centering, covariance, eigenvectors, eigenvalues, PCA,
projection, nearest neighbors, and recommendations — suitable for a
college viva presentation.
