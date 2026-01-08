# sample_representative_wvs_final_with_color_legend.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import argparse
from pathlib import Path

WVS_ITEMS = [
    'N_REGION_WVS', 'N_TOWN', 'G_TOWNSIZE2', 'H_SETTLEMENT', 'H_URBRURAL', 'E_RESPINT',
    'Q260', 'Q261', 'Q262', 'Q263', 'Q264', 'Q265', 'Q266', 'Q267', 'Q268', 'Q269',
    'Q270', 'Q271', 'Q272', 'Q273', 'Q274', 'Q275', 'Q276', 'Q277', 'Q278', 'Q279',
    'Q280', 'Q281', 'Q282', 'Q283', 'Q284', 'Q285', 'Q286', 'Q287',
    'Q288R', 'Q289', 'Q290'
]

def clean_float(x):
    if pd.isna(x): return np.nan
    try: return float(str(x).strip())
    except: return np.nan

def main(args):
    print("Loading WVS data...")
    df = pd.read_excel(args.file_path).copy()

    required = ['B_COUNTRY', 'sacsecval', 'resemaval'] + WVS_ITEMS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df['sacsecval'] = df['sacsecval'].apply(clean_float)
    df['resemaval'] = df['resemaval'].apply(clean_float)

    df_original = df.copy()
    df_pca = df[WVS_ITEMS].copy().reset_index(drop=True)

    # Encode strings → integers (only for PCA)
    for col in WVS_ITEMS:
        if df_pca[col].dtype == 'object':
            df_pca[col] = df_pca[col].fillna('missing')
            df_pca[col] = df_pca[col].astype(str).rank(method='dense').astype(int) - 1

    # PCA
    X = SimpleImputer(strategy='median').fit_transform(df_pca)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_original = df_original.reset_index(drop=True)
    df_original['PC1'] = X_pca[:, 0]
    df_original['PC2'] = X_pca[:, 1]
    df_original['PC3'] = X_pca[:, 2]
    df_original['PC4'] = X_pca[:, 3]

    print(f"PCA explained: PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}")

    # Output dirs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_country").mkdir(exist_ok=True)

    # Color setup
    countries = sorted(df_original['B_COUNTRY'].unique())
    palette = sns.color_palette("husl", n_colors=len(countries))
    country_color = dict(zip(countries, palette))

    representatives = []

    print(f"\nSelecting top {args.n} per country...")
    for country in countries:
        group = df_original[df_original['B_COUNTRY'] == country].copy()
        n_take = min(args.n, len(group))

        if n_take == len(group):
            selected = group.copy()
            selected['selection_rank'] = range(1, len(selected)+1)
            selected['distance_to_centroid'] = np.nan
        else:
            pca_vals = group[['PC1','PC2','PC3','PC4']].values
            centroid = pca_vals.mean(axis=0)
            distances = np.linalg.norm(pca_vals - centroid, axis=1)
            idx = np.argsort(distances)[:n_take]
            selected = group.iloc[idx].reset_index(drop=True)
            selected['selection_rank'] = range(1, n_take+1)
            selected['distance_to_centroid'] = distances[idx]

        safe_name = str(country).replace(' ', '_').replace('/', '_')
        selected.to_excel(out_dir / "per_country" / f"{safe_name}_representative.xlsx", index=False)
        representatives.append(selected)

    final_sample = pd.concat(representatives, ignore_index=True)
    final_sample.to_excel(out_dir / f"all_representatives_n{args.n}.xlsx", index=False)

    # === Legend elements ===
    respondent_handle = plt.Line2D([0], [0], marker='o', color='w', label='All Respondents',
                                   markerfacecolor='gray', markersize=9, alpha=0.7)
    star_handle = plt.Line2D([0], [0], marker='*', color='w', label='Selected Representative',
                             markerfacecolor='gold', markeredgecolor='black', markeredgewidth=2, markersize=15)

    # Country color legend (max 15 shown)
    max_countries_in_legend = 15
    country_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=country_color[c], markersize=10, label=c)
        for c in countries[:max_countries_in_legend]
    ]
    if len(countries) > max_countries_in_legend:
        country_handles.append(plt.Line2D([0], [0], color='none', label='…'))

    legend_handles = [respondent_handle] + country_handles + [star_handle]

    # === Plot 1: Demographic PCA ===
    plt.figure(figsize=(13, 9))
    for country in countries:
        sub = df_original[df_original['B_COUNTRY'] == country]
        plt.scatter(sub['PC1'], sub['PC2'], color=country_color[country], alpha=0.2, s=20)

    for country in countries:
        sub = final_sample[final_sample['B_COUNTRY'] == country]
        plt.scatter(sub['PC1'], sub['PC2'], color=country_color[country], s=320, marker='*',
                    edgecolor='black', linewidth=2.5)

    plt.xlabel("Principal Component 1", fontsize=13)
    plt.ylabel("Principal Component 2", fontsize=13)
    plt.title("Global Distribution (PCA on 38 WVS Items)\nStars = Most Representative per Country", fontsize=14, pad=15)
    plt.legend(handles=legend_handles, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11, ncol=2)

    plt.tight_layout()
    plt.savefig(out_dir / "01_global_demographic_pca.png", dpi=350, bbox_inches='tight')
    plt.close()

    # === Plot 2: Inglehart-Welzel ===
    plt.figure(figsize=(13, 9))
    for country in countries:
        sub = df_original[df_original['B_COUNTRY'] == country]
        plt.scatter(sub['sacsecval'], sub['resemaval'], color=country_color[country], alpha=0.35, s=35)

    for country in countries:
        sub = final_sample[final_sample['B_COUNTRY'] == country]
        plt.scatter(sub['sacsecval'], sub['resemaval'], color=country_color[country], s=340, marker='*',
                    edgecolor='black', linewidth=2.8)

    plt.xlabel("Traditional ← → Secular-Rational Values", fontsize=13)
    plt.ylabel("Survival ← → Self-Expression Values", fontsize=13)
    plt.title("Inglehart-Welzel Cultural Map\nStars = Most Representative Individual per Country", fontsize=15, pad=20)

    plt.legend(handles=legend_handles, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11, ncol=2)

    plt.tight_layout()
    plt.savefig(out_dir / "02_inglehart_welzel_cultural_map.png", dpi=400, bbox_inches='tight')
    plt.close()

    print("\nAll done! Perfect plots with color legend in bottom-right.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default="final_representatives")
    args = parser.parse_args()
    main(args)