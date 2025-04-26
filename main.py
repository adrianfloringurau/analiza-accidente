import streamlit as st
import pandas as pd
import numpy as np
import functii as f
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, roc_curve, auc, classification_report, ConfusionMatrixDisplay, silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import statsmodels.api as sm

dtype_dict = {
    "Weather": "string",
    "Road_Type": "string",
    "Time_of_Day": "string",
    "Traffic_Density": "int64",
    "Speed_Limit": "float64",
    "Number_of_Vehicles": "int64",
    "Driver_Alcohol": "int64",
    "Accident_Severity": "string",
    "Road_Condition": "string",
    "Vehicle_Type": "string",
    "Driver_Age": "float64",
    "Driver_Experience": "float64",
    "Road_Light_Condition": "string",
    "Accident": "int64"
}

df = pd.read_csv(filepath_or_buffer='./dataset_traffic_accident_prediction1.csv')

page1 = "Prezentare date"
page2 = "Analiza exploratorie a datelor"
page3 = "Utilizarea GeoPandas"
page4 = "Predicții și indicatori de performanță"

section = st.sidebar.radio("Sectiune", [page1, page3, page2, page4])

st.title("Predicție Accidente Rutiere")

if "isReadyForPredictions" not in st.session_state:
    st.session_state.isReadyForPredictions = False
    st.session_state.g_df_cleaned = None
    st.session_state.g_df_encoded = None
    st.session_state.g_le_dict = None
    st.session_state.g_df_scaled = None
    st.session_state.g_scaler = None
    st.session_state.numerical_columns = None
    st.session_state.categorical_columns = None

if section == page1:
    st.header(page1)

    st.markdown(r"""

    ```
    Proiect realizat de:
    - Gurău Adrian-Florin, grupa 1092
    - Honciu Eduard-Marian, grupa 1092
    ```            

    ### Condițiile care influențează probabilitatea accidentelor rutiere

    #### **Vremea**
    Impactul condițiilor meteorologice asupra probabilității accidentelor:
    - **Clar:** Fără condiții meteorologice adverse.
    - **Ploios:** Crește șansele de accidente.
    - **Ceață:** Reduce vizibilitatea, crescând riscul de accidente.
    - **Ninsoare:** Drumuri alunecoase și probabilitate crescută de accidente.
    - **Furtună:** Condiții periculoase pentru conducere.

    #### **Tipul Drumului**
    Tipul drumului influențează probabilitatea accidentelor:
    - **Autostradă:** Viteză mare, risc crescut de accidente severe.
    - **Drum Urban:** Drumuri din orașe, trafic mai mare și viteze mai mici.
    - **Drum Rural:** În afara orașelor, mai puține vehicule și viteze mai mici.
    - **Drum de Munte:** Curburi și diferențe de altitudine, risc crescut de accidente.

    #### **Momentul Zilei**
    Timpul din zi în care are loc accidentul:
    - **Dimineață:** Între răsărit și prânz.
    - **După-amiază:** Între prânz și seară.
    - **Seară:** Perioada dinaintea apusului.
    - **Noapte:** Vizibilitate redusă, risc mai mare.

    #### **Densitatea Traficului**
    Nivelul traficului pe drum:
    - **0:** Densitate redusă (puține vehicule).
    - **1:** Densitate moderată.
    - **2:** Densitate ridicată (multe vehicule).

    #### **Limita de Viteză**
    Viteza maximă permisă pe drum.

    #### **Numărul de Vehicule**
    Numărul vehiculelor implicate în accident.

    #### **Consum Alcool**
    Dacă șoferul a consumat alcool:
    - **0:** Fără consum de alcool.
    - **1:** Consum de alcool (crește riscul de accident).

    #### **Gravitatea Accidentului**
    Nivelul de severitate al accidentului:
    - **Scăzut:** Accident minor.
    - **Moderată:** Daune sau răni moderate.
    - **Ridicat:** Accident grav, daune semnificative sau răni serioase.

    #### **Condiția Drumului**
    Starea suprafeței drumului:
    - **Uscat:** Risc minim.
    - **Ud:** Drum umed din cauza ploii, risc crescut.
    - **Înghețat:** Gheață pe drum, risc semnificativ.
    - **În construcție:** Posibile obstacole, calitate scăzută a drumului.

    #### **Tipul Vehiculului**
    Tipul de vehicul implicat în accident:
    - **Mașină:** Autoturism obișnuit.
    - **Camion:** Vehicul mare pentru transport marfă.
    - **Motocicletă:** Vehicul pe două roți.
    - **Autobuz:** Vehicul mare pentru transport public.

    #### **Vârsta Șoferului**
    Valori între 18 și 70 de ani.

    #### **Experiența Șoferului**
    Anii de experiență ai șoferului (0 - 70 ani).

    #### **Condițiile de Iluminare**
    Condițiile de iluminare de pe drum:
    - **Zi:** Vizibilitate bună.
    - **Lumină artificială:** Drum iluminat cu felinare.
    - **Fără lumină:** Drum neiluminat, vizibilitate redusă.
                
    #### **Accident**
    Aceasta este varibila depedentă, ce va fi previzionată pentru un set de valori date variabilelor enumerate anterior.
    - **0:** Nu s-a produs accident.
    - **1:** Accident realizat.
    """, unsafe_allow_html=True)

    st.header("Datele se regăsesc în următorul tabel:")

    st.dataframe(df)
elif section == page2:
    st.header(page2)

    st.markdown(r"""
    ### **Cast variabile**
    Am schimbat tipul de dată al coloanelor din setul de date, astfel încât fiecare coloană să fie setată pe un anumit tip de dată, indicat de următorul dicționar:

    ```
    dtype_dict = {
        "Weather": "string",
        "Road_Type": "string",
        "Time_of_Day": "string",
        "Traffic_Density": "int64",
        "Speed_Limit": "float64",
        "Number_of_Vehicles": "int64",
        "Driver_Alcohol": "int64",
        "Accident_Severity": "string",
        "Road_Condition": "string",
        "Vehicle_Type": "string",
        "Driver_Age": "float64",
        "Driver_Experience": "float64",
        "Road_Light_Condition": "string",
        "Accident": "int64"
    }
    ```      
    Acest lucru va ajuta mai departe, la tratarea valorilor lipsă, astfel încât noile valori să respecte tipul de dată corect.      
    """, unsafe_allow_html=True)

    st.header("Datele formatate:")

    df = f.cast_colums(df, dtype_dict)
    st.dataframe(df)

    st.header("Număr valori lipsă:")

    dfvl = f.calcul_valori_lipsa(df)
    st.dataframe(dfvl)

    st.header("Înlocuire valori lipsă:")

    st.markdown(r"""
    ### **Opțiunile globale posibile:**

    ```
    1. global_fill_default – completează valorile lipsă cu:
    - media, pentru numerice;
    - cea mai frecventă valoare (modul), pentru string-uri.
    2. global_fill_avg – similar cu global_fill_default (medie, pentru date numerice, valoare „medie” ca frecvență, în cazul string-urilor).
    3. global_fill_max – completează cu maximul valorilor (pentru datele numerice), sau valoarea cea mai frecventă (pentru string-uri).
    4. global_fill_min – completează cu minimul valorilor (pentru datele numerice), sau valoarea cea mai rară (pentru string-uri).
    ```
                 
    ### **Opțiunile group posibile:**
    
    ```
    1. group_fill_default – completează în fiecare grup (realizat după coloana aleasă) cu:
    - media grupului pentru datele numerice;
    - modul (cea mai frecventă valoare) pentru string-uri.
    2. group_fill_avg – similar cu group_fill_default (medie, pentru date numerice, valoare „medie” ca frecvență, în cazul string-urilor).
    3. group_fill_max – completează cu maximul grupului (pentru datele numerice) sau valoarea cea mai frecventă (pentru string-uri).
    4. group_fill_min – completează cu minimul grupului (pentru datele numerice) sau cea mai rară valoare (pentru string-uri).
    ```   
    """, unsafe_allow_html=True)

    tabel = {
        "Metoda": ["Global", "Group"],
        "Descriere simplificată": [
            "Completare cu o singură valoare la nivelul întregii coloane.",
            "Completare specifică pentru fiecare grup separat."
        ],
        "Exemplu utilizare": [
            "media generală a vârstelor",
            "media vârstelor pe fiecare tip de vreme"
        ]
    }

    df_tabel = pd.DataFrame(data=tabel)
    st.dataframe(df_tabel.style.set_properties(**{'text-align': 'left'}))

    options = ["global_fill_default", "group_fill_default", "global_fill_avg", "group_fill_avg", "global_fill_max", "group_fill_max", "global_fill_min", "group_fill_min"]
    method_select = st.selectbox("Selectează metoda de completare a valorilor lipsă:", options)

    column_select = st.selectbox("Selectează coloana după care să se facă gruparea:", df.columns if "group" in method_select else ["[ALEGE O METODĂ DE GRUP PENTRU A ACCESA ACEST CÂMP]"], disabled=False if "group" in method_select else True)

    st.text("Dataframe-ul rezultat:")

    df_mask = df.isnull()

    df_cleaned = f.fill_na(df=df, option=method_select, groupColumn=column_select if "group" in method_select else None)

    st.dataframe(df_cleaned.style.apply(lambda x: df_mask[x.name].map(lambda mask: 'background-color: yellow;' if mask else ''), axis=0))

    st.header(f"Detectare Valori Extreme")

    percentile = st.slider("Alege percentila pentru definirea valorilor extreme:", 0, 10, 1)

    numeric_columns = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
    selected_column = st.selectbox("Selectează coloana numerică pentru analiza valorilor aberante și a statisticilor descriptive:", numeric_columns)

    if selected_column:
        st.subheader(f"Histplot pentru: {selected_column}")

        lower_bound = np.percentile(df_cleaned[selected_column].dropna(), percentile)
        upper_bound = np.percentile(df_cleaned[selected_column].dropna(), 100 - percentile)

        st.write(f"Limita inferioară (P{percentile}): {lower_bound:.2f}")
        st.write(f"Limita superioară (P{100 - percentile}): {upper_bound:.2f}")

        outliers = df_cleaned[
            (df_cleaned[selected_column] < lower_bound) |
            (df_cleaned[selected_column] > upper_bound)
        ][selected_column]

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(df_cleaned[selected_column].dropna(), bins=60, kde=True, ax=ax)

        ax.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label=f'P{percentile} ({percentile}%)')
        ax.axvline(upper_bound, color='red', linestyle='--', linewidth=1.5, label=f'P{100 - percentile} ({100 - percentile}%)')

        ax.set_title(f"Distribuție și valori extreme ({selected_column})")
        ax.legend()

        st.pyplot(fig)

        st.subheader(f"Violinplot pentru: {selected_column}")

        fig, ax = plt.subplots(figsize=(6, 8))

        sns.violinplot(y=df_cleaned[selected_column], ax=ax, inner="point")
        ax.axhline(lower_bound, color='red', linestyle='--', label=f'P{percentile}')
        ax.axhline(upper_bound, color='red', linestyle='--', label=f'P{100 - percentile}')

        ax.set_title(f"Distribuție și valori extreme ({selected_column})")
        ax.legend()

        st.pyplot(fig)

        st.write(f"Valori extreme în {selected_column} (X < P{percentile} sau X > P{100 - percentile}):")
        st.dataframe(outliers)

        st.subheader(f"Statistici descriptive pentru '{selected_column}'")

        desc_stats = {
            "Media": df_cleaned[selected_column].mean(),
            "Mediana": df_cleaned[selected_column].median(),
            "Deviația Standard": df_cleaned[selected_column].std(),
            "Minim": df_cleaned[selected_column].min(),
            f"Percentila {percentile} (P{percentile})": np.percentile(df_cleaned[selected_column].dropna(), percentile),
            "Quartila 1 (Q1)": df_cleaned[selected_column].quantile(0.25),
            f"Percentila {100 - percentile} (P{100 - percentile})": np.percentile(df_cleaned[selected_column], 100 - percentile),
            "Quartila 3 (Q3)": np.percentile(df_cleaned[selected_column], 75),
            "Maxim": df_cleaned[selected_column].max()
        }

        stats_df = pd.DataFrame.from_dict(stats_df := {"Statistica": desc_stats.keys(), "Valoare": desc_stats.values()})
        st.dataframe(stats_df.set_index("Statistica"))

    st.header("Statistici descriptive pentru întregul Dataframe:")
    st.dataframe(df_cleaned.describe())

    st.header("Statistici descriptive pentru Dataframe-ul grupat după o coloană:")

    col_groupby = st.selectbox("Selectează coloana pentru grupare:", df_cleaned.columns)

    def p1(x):
        return np.percentile(x, 1)

    def p25(x):
        return np.percentile(x, 25)

    def p75(x):
        return np.percentile(x, 75)

    def p99(x):
        return np.percentile(x, 99)

    if col_groupby:
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()

        df_grouped_stats = df_cleaned.groupby(col_groupby, observed=False)[numeric_cols].agg(
            ['mean', 'median', 'std', 'min', 
            p1,
            p25,
            p75,
            p99,
            'max']
        )

        df_grouped_stats.columns = [''.join(" - ").join(col).strip() for col in df_grouped_stats.columns.values]

        st.subheader(f"Statistici descriptive detaliate grupate după '{col_groupby}':")
        st.dataframe(df_grouped_stats.reset_index())

    st.subheader("Matricea corelațiilor")
    correlation_matrix = df_cleaned.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    df_segment = df_cleaned.groupby(["Weather", "Driver_Alcohol"], observed=False)["Accident"].mean().unstack()
    st.subheader("Accidente cauzate de alcool în diferite condiții meteo")
    df_segment.columns = ["Alcool = " + str(col) for col in df_segment.columns]
    st.dataframe(df_segment)

    st.subheader("Pairplot pentru variabilele numerice")

    if "show_button" not in st.session_state:
        st.session_state.show_button = True

    if st.session_state.show_button:
        if st.button("Generează pairplot"):
            st.session_state.show_button = False
            st.rerun()

    if not st.session_state.show_button:
        f.pairplot_numeric(df_cleaned, numeric_cols)

    categorical_columns = [col for col, dtype in dtype_dict.items() if dtype == "string"]

    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    df_encoded = df.drop(columns=categorical_columns)

    st.subheader("Utilizarea LabelEncoder pentru encodarea valorilor non-numerice")

    st.dataframe(df_encoded)
    
    st.text("Mapping-ul Label Encoder-ului")
    for col, mapping in le_dict.items():
        st.write(f"**{col}**:")
        for category, code in mapping.items():
            st.write(f"- {category} → {code}")

    st.subheader("Scalarea datelor")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index)
    st.dataframe(df_scaled)

    st.text("Statistici descriptive înainte de scalare:")

    st.dataframe(df_encoded.describe())

    st.text("Statistici descriptive după scalare:")

    st.dataframe(df_scaled.describe())

    st.text("Se observă că media tinde spre 0 și deviația standard tinde spre 1, ceea ce confirmă scalarea corectă a datelor.")

    st.session_state.g_df_cleaned = df_cleaned.copy(deep=True)
    st.session_state.g_df_encoded = df_encoded.copy(deep=True)
    st.session_state.g_le_dict = copy.deepcopy(le_dict)
    st.session_state.g_df_scaled = df_scaled.copy(deep=True)
    st.session_state.g_scaler = copy.deepcopy(scaler)
    st.session_state.numerical_columns = copy.deepcopy(numeric_columns)
    st.session_state.categorical_columns = copy.deepcopy(categorical_columns)
    st.session_state.isReadyForPredictions = True

elif section == page3:
    st.warning("Acestă pagină folosește un alt set de date. Toate celelalte pagini utilizează setul de date principal al acestui proiect.")

    st.header("Accidente în România după severitate")

    romania_gdf = gpd.read_file("romania.json")

    df_accidents = pd.read_excel("accidents_romania.xlsx")
    gdf_accidents = gpd.GeoDataFrame(
        df_accidents,
        geometry=gpd.points_from_xy(df_accidents.Longitude, df_accidents.Latitude),
        crs="EPSG:4326"
    )

    accidents_within_romania = gpd.sjoin(gdf_accidents, romania_gdf, predicate="within", how="inner")

    fig, ax = plt.subplots(figsize=(10, 10))
    romania_gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
    accidents_within_romania.plot(ax=ax, column='Accident_Severity', cmap='inferno', markersize=20, legend=True)
    ax.set_axis_off()
    plt.title("Accidente simulate în România după severitate")

    st.dataframe(accidents_within_romania)

    st.pyplot(fig)

    st.title("Analiza accidentelor în funcție de vreme și tipul drumului")

    weather = st.selectbox("🌤️ Alege vremea:", accidents_within_romania["Weather"].unique())
    road_type = st.selectbox("🛣️ Alege tipul drumului:", accidents_within_romania["Road_Type"].unique())

    filtered_data_romania = accidents_within_romania[
        (accidents_within_romania["Weather"] == weather) & (accidents_within_romania["Road_Type"] == road_type)]

    fig, ax = plt.subplots(figsize=(10, 10))
    romania_gdf.plot(ax=ax, color='white', edgecolor='black')
    filtered_data_romania.plot(ax=ax, color='red', markersize=30)
    ax.set_axis_off()
    plt.title(f"Accidente ({weather}, {road_type})")

    st.pyplot(fig)
elif section == page4:
    st.header(page4)

    if st.session_state.isReadyForPredictions:
        if st.session_state.g_df_cleaned is not None and\
        st.session_state.g_df_encoded is not None and\
        st.session_state.g_le_dict is not None and\
        st.session_state.g_df_scaled is not None and\
        st.session_state.g_scaler is not None and\
        st.session_state.numerical_columns is not None and\
        st.session_state.categorical_columns is not None:
            df_cleaned = pd.DataFrame(copy.deepcopy(st.session_state.g_df_cleaned))
            df_encoded = pd.DataFrame(copy.deepcopy(st.session_state.g_df_encoded))
            le_dict = copy.deepcopy(st.session_state.g_le_dict)
            df_scaled = pd.DataFrame(copy.deepcopy(st.session_state.g_df_scaled))
            scaler = copy.deepcopy(st.session_state.g_scaler)
            numerical_columns = copy.deepcopy(st.session_state.numerical_columns)
            categorical_columns = copy.deepcopy(st.session_state.categorical_columns)

            st.text("Vom realiza predicțiile, folosind setul de date principal cu valorile lipsă tratate, pe cel encodat, cât și pe cel scalat. Vom avea acces și la dicționarul de label-uri pentru decodare, cât și la scaler pentru scalarea datelor pentru predicție în aceeași manieră.")

            st.subheader("Selectare valori pentru predicție")

            predictori = df.columns[:-1]
            tinta = df.columns[-1]

            selected_values = {}
            for column_name in predictori:
                if pd.api.types.is_numeric_dtype(df_cleaned[column_name]):
                    selected_values[column_name] = st.number_input(
                        f"{column_name}", 
                        min_value=float(df_cleaned[column_name].min()), 
                        max_value=float(df_cleaned[column_name].max()), 
                        value=float(df_cleaned[column_name].median())
                    )
                else:
                    selected_values[column_name] = st.selectbox(
                        f"{column_name}", 
                        df_cleaned[column_name].unique()
                    )

            st.write("Selected Values:", selected_values)

            st.subheader("Encodarea datelor LabelEncoder-ul preutilizat")

            encoded_values = {}

            for col in numerical_columns:
                if col == "Accident":
                    continue
                encoded_values[col] = selected_values[col]

            for col in categorical_columns:
                encoded_values[col + "_encoded"] = int(le_dict[col][selected_values[col]])


            st.write("Encoded Values:", encoded_values)

            encoded_values["Accident"] = 0

            st.subheader("Scalarea datelor cu StandardScaler-ul preutilizat")

            scaled_values = copy.deepcopy(encoded_values)
            df_scaled_values = pd.DataFrame([scaled_values])

            expected_features = scaler.feature_names_in_
            df_scaled_values = df_scaled_values.reindex(columns=expected_features, fill_value=0)

            df_scaled_values = pd.DataFrame(scaler.transform(df_scaled_values), columns=expected_features)

            map_scaled_values = {}
            for col in df_scaled_values.columns:
                map_scaled_values[col] = float(df_scaled_values[col].iloc[0])

            del encoded_values["Accident"]
            del scaled_values["Accident"]
            df_scaled_values = df_scaled_values.drop(columns=["Accident"])
            del map_scaled_values["Accident"]

            st.write("Scaled Values:", map_scaled_values)

            st.subheader("Alegere model și fine-tuning", divider="red")
            modele = {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier (),
                "SVC": SVC(),
                "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
                "GaussianNB": GaussianNB()
            }
            model = st.selectbox("Model", modele.keys())
            test_dims = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            test_dim = st.selectbox("Test size", test_dims, index=int(len(test_dims)/2))

            predictori_encoded = [
                predictor + "_encoded" if predictor in categorical_columns else predictor
                for predictor in predictori
            ]

            df_scaled[tinta] = df_scaled[tinta].round().astype(int)
            X_train, X_test, Y_train, Y_test = train_test_split(df_scaled[predictori_encoded], df_scaled[tinta], test_size=test_dim, random_state=42)

            Y_test = [0 if value == -1 else 1 for value in Y_test]
            Y_train = [0 if value == -1 else 1 for value in Y_train]

            ###############################################################################################
            m = modele[model]

            m.fit(X_train, Y_train)
            y_pred = m.predict(X_test)
            acc = accuracy_score(Y_test, y_pred)
            ck = cohen_kappa_score(Y_test, y_pred)
            roc = roc_auc_score(Y_test, y_pred)
            
            indicatori = pd.DataFrame(data={
                "Acuratețe globală" : acc,
                "Cohen Kappa" : ck,
                "ROC AUC": [roc] if roc is not None else ["N/A"]
            }, index=["Valoare"])

            st.dataframe(indicatori)

            st.text("Classification Report:\n" + classification_report(Y_test, y_pred))

            ConfusionMatrixDisplay.from_predictions(Y_test, y_pred, cmap="Wistia")
            plt.title(f"Confusion Matrix - {model}")
            st.pyplot(plt)

            fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            roc_plot = plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.grid(True)
            st.pyplot(roc_plot)

            st.text("Având în vedere indicatorii obținuți pe setul de testare, predicția pentru valorile alese la început este:")

            df_scaled_values = df_scaled_values.reindex(columns=predictori_encoded)
            rezultat = m.predict(df_scaled_values)
            st.write(f"Accident: {rezultat}")

            st.subheader("Clusterizare", divider="red")

            metoda_clusterizare = st.selectbox("Alege metoda de clusterizare:", ['single', 'average', 'weighted', 'complete', 'ward'])

            ierarhie = linkage(y=df_scaled, method=metoda_clusterizare)

            ierarhie_df = pd.DataFrame(data=ierarhie, index=[str(i) for i in range(len(ierarhie))], columns=[["Cluster 1", "Cluster 2", "Distanta", "Nr Instante"]])
            ierarhie_df[["Cluster 1", "Cluster 2", "Nr Instante"]] = ierarhie_df[["Cluster 1", "Cluster 2", "Nr Instante"]].values.astype(int)
            st.text("Matricea ierarhie:")
            st.dataframe(ierarhie_df)

            distante = ierarhie[:, 2]
            diferente = np.diff(distante, 2)
            elbow = len(ierarhie) - np.argmax(diferente) + 1

            threshold = (ierarhie[np.argmax(diferente), 2] + ierarhie[np.argmax(diferente) + 1, 2]) / 2

            dendograma = plt.figure()
            plt.plot()
            plt.title(label="Dendograma pentru partiția optimală")
            dendrogram(Z=ierarhie, p=int(elbow), truncate_mode='lastp')
            plt.axhline(y=threshold)
            st.pyplot(dendograma)

            clusters = fcluster(Z=ierarhie, t=int(elbow), criterion='maxclust')
            df_cleaned_copy = df_cleaned.copy(deep=True)
            df_cleaned_copy["Cluster"] = clusters
            st.text("Afișarea setului de date grupat pe clusteri:")
            st.dataframe(df_cleaned_copy)

            silhouette = silhouette_samples(X=df_scaled, labels=clusters)
            silhouette_total = silhouette_score(X=df_scaled, labels=clusters)

            st.text(f"Scorul Silhouette pentru partiția optimală: {silhouette_total}")

            s_plot = plt.figure()
            plt.plot()
            plt.title(label="Plot Silhouette pentru partiția optimală")
            x = np.arange(1, len(silhouette) + 1, 1)
            plt.xticks(x)
            #plt.scatter(x=x, y=silly)
            plt.hist(x, bins=len(silhouette), weights=silhouette)
            #plt.stem(x, silly, linefmt="b-", markerfmt="bo", basefmt="r-")
            st.pyplot(s_plot)

            st.text("Clusterizare pe partiție-k:")

            k = st.selectbox("k =", [i for i in range(2, int(len(df_scaled) / 2))])
            m = len(ierarhie)

            def calcul(k):
                for i in range(m):
                    if m - i == k + 1:
                        return (ierarhie[i, 2] + ierarhie[i + 1, 2]) / 2
                    
            threshold = calcul(k)

            dendograma_k = plt.figure()
            plt.plot()
            plt.title(label=f"Dendograma pentru partiția k={k}")
            dendrogram(Z=ierarhie, p=k, truncate_mode='lastp')
            plt.axhline(y=threshold)
            st.pyplot(dendograma_k)

            clusters_k = fcluster(Z=ierarhie, t=k, criterion='maxclust')
            df_cleaned_copy_k = df_cleaned.copy(deep=True)
            df_cleaned_copy_k["Cluster"] = clusters_k
            st.text(f"Afișarea setului de date grupat pe clusteri, cu k = {k}:")
            st.dataframe(df_cleaned_copy_k)

            silhouette_k = silhouette_samples(X=df_scaled, labels=clusters_k)
            silhouette_total_k = silhouette_score(X=df_scaled, labels=clusters_k)

            s_plot_k = plt.figure()
            plt.plot()
            plt.title(label=f"Plot Silhouette pentru k = {k}")
            x = np.arange(1, len(silhouette_k) + 1, 1)
            plt.xticks(x)
            #plt.scatter(x=x, y=silly)
            plt.hist(x, bins=len(silhouette_k), weights=silhouette_k)
            #plt.stem(x, silly, linefmt="b-", markerfmt="bo", basefmt="r-")
            st.pyplot(s_plot_k)

            st.text(f"Scorul Silhouette pentru partiția k = {k}: {silhouette_total_k}")

            st.subheader("Analiza Regresiei Multiple (statsmodels)", divider='red')
            st.write("Această secțiune folosește datele *înainte* de scalare (dar după encodare) pentru o interpretare mai ușoară a coeficienților.")

            dependent_var_options_reg = df_encoded.select_dtypes(include=np.number).columns.tolist()
            default_dependent_reg = 'Accident' if 'Accident' in dependent_var_options_reg else (dependent_var_options_reg[0] if dependent_var_options_reg else None)

            if default_dependent_reg:
                dependent_var_reg = st.selectbox(
                    "Alege Variabila Dependentă (Y) pentru regresie:",
                    dependent_var_options_reg,
                    index=dependent_var_options_reg.index(default_dependent_reg),
                    key='reg_dependent_var_select',
                )
            else:
                st.warning("Nu s-au găsit coloane numerice pentru variabila dependentă în datele encodate.")
                dependent_var_reg = None

            if dependent_var_reg:
                independent_var_options_reg = [col for col in dependent_var_options_reg if col != dependent_var_reg]
                default_independent_reg_sugg = ['Driver_Age', 'Traffic_Density', 'Number_of_Vehicles', 'Speed_Limit']
                default_independent_reg = [col for col in default_independent_reg_sugg if col in independent_var_options_reg]

                independent_vars_reg = st.multiselect(
                    "Alege Variabilele Independente (X) pentru regresie:",
                    independent_var_options_reg,
                    default=default_independent_reg,
                    key='reg_independent_vars_multi'
                )

                if independent_vars_reg:
                    st.write(f"Se va construi un model OLS pentru a prezice '{dependent_var_reg}' folosind: {', '.join(independent_vars_reg)}")

                    try:
                        Y_reg = pd.to_numeric(df_encoded[dependent_var_reg], errors='coerce')
                        X_reg = df_encoded[independent_vars_reg].apply(pd.to_numeric, errors='coerce')

                        data_reg = pd.concat([Y_reg, X_reg], axis=1).dropna()

                        if data_reg.empty:
                            st.error("Nu există date valide pentru regresie după eliminarea valorilor NaN introduse la conversia numerică.")
                        else:
                            Y_reg_clean = data_reg[dependent_var_reg].astype(float)
                            X_reg_clean = data_reg[independent_vars_reg].astype(float)

                            X_reg_const = sm.add_constant(X_reg_clean)

                            model_ols = sm.OLS(Y_reg_clean, X_reg_const)
                            results_ols = model_ols.fit()

                            st.subheader("Sumarul Modelului de Regresie Multiplă (OLS)")
                            st.code(results_ols.summary().as_text())

                            st.subheader("Interpretare Sumar (simplificat)")
                            st.write(f"- **R-squared:** {results_ols.rsquared:.3f}")
                            st.write(f"- **Adj. R-squared:** {results_ols.rsquared_adj:.3f}")
                            st.write(f"- **P-value (F-statistic):** {results_ols.f_pvalue:.3g}")

                            st.write("**Coeficienți și Semnificația lor (P>|t|):**")
                            results_summary_df = pd.DataFrame({
                                'Coeficient': results_ols.params,
                                'Std. Error': results_ols.bse,
                                't': results_ols.tvalues,
                                'P>|t|': results_ols.pvalues,
                                '[0.025': results_ols.conf_int()[0],
                                '0.975]': results_ols.conf_int()[1]
                            })
                            st.dataframe(results_summary_df)
                            st.caption("*Valorile P>|t| mici (ex: < 0.05) sugerează că predictorul respectiv este semnificativ statistic.*")

                    except Exception as e:
                        st.error(f"Eroare la construirea modelului de regresie OLS: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                else:
                    st.warning("Selectează cel puțin o variabilă independentă pentru regresie.")
            else:
                st.warning("Selectează o variabilă dependentă pentru regresie.")
    else:
        st.warning("Mergi la pagina: '" + page2 + "' pentru a procesa setul de date principal, mai întâi.")