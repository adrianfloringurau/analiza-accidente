import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def cast_colums(df: pd.DataFrame, dtypes: dict):
    for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    if dtype == "string":
                        df[col] = df[col].astype("string")
                    elif dtype == "int64":
                        df[col] = pd.to_numeric(df[col]).astype("Int64") # nullable int type
                    elif dtype == "float64":
                        df[col] = pd.to_numeric(df[col]).astype("float64")
                except ValueError as e:
                    print(f"Warning: Could not cast column {col} to {dtype}. Error: {str(e)}")
    return df

def calcul_valori_lipsa(df: pd.DataFrame):
    res = {}
    total = 0
    for col in df.columns:
        nrvl = df[col].isna().sum()
        res[col] = nrvl
        total += nrvl
    res["Total"] = total
    res_df = pd.DataFrame.from_dict(data=res, orient="index", columns=["Missing_Values_Number"])
    return res_df
    
def fill_na(df: pd.DataFrame, option: str, groupColumn: str = None):
    if not df.isnull().values.any():
        return df

    fill_methods = ['default', 'avg', 'max', 'min']
    assert any(method in option for method in fill_methods), "Option must include one of 'default', 'avg', 'max', or 'min'."

    def fill_series(series, method, col, isGroupMethod):
        if is_numeric_dtype(series):
            if method == 'default' or method == 'avg':
                mean_value = series.mean()
                if pd.isna(mean_value):  # Check if mean is NaN or pd.NA
                    if isGroupMethod == False:
                        mean_value = 0
                    else:
                        return fill_series(df[col], method, col, False)
                return series.fillna(mean_value if pd.api.types.is_float_dtype(series) else int(mean_value))
            elif method == 'max':
                return series.fillna(series.max())
            elif method == 'min':
                return series.fillna(series.min())
        else:
            mode_values = series.mode()
            if (mode_values.empty):
                if isGroupMethod == False:
                    mode_value = "N/A"
                else:
                    return fill_series(df[col], method, col, False)
            mode_value = mode_values.iloc[0]
            freq = series.value_counts()

            if method in ['default', 'max']:
                return series.fillna(mode_value)
            elif method == 'avg':
                mean_freq = freq.mean()
                closest_string = min(freq.index, key=lambda x: abs(freq[x] - mean_freq))
                return series.fillna(closest_string)
            elif method == 'min':
                least_frequent = freq.idxmin()
                return series.fillna(least_frequent)

    method_used = next((m for m in fill_methods if m in option), 'default')

    if "global" in option:
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = fill_series(df[col], method_used, col, False)

    elif "group" in option and groupColumn:
        if df[groupColumn].isnull().any():
            df[groupColumn] = fill_series(df[groupColumn], method_used, groupColumn, False)

        for col in df.columns:
            if col == groupColumn or not df[col].isnull().any():
                continue

            group_values = df.groupby(groupColumn)[col].transform(lambda x: fill_series(x, method_used, col, True))

            df[col] = df[col].fillna(group_values)

    return df

def pairplot_numeric(df, numeric_cols, title="Pairplot pentru variabilele numerice"):
    fig = sns.pairplot(df[numeric_cols], diag_kind='kde')
    plt.suptitle(title, y=1.02)
    st.pyplot(fig)