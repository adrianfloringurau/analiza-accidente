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

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, r2_score, roc_auc_score, roc_curve, mean_squared_error, classification_report, ConfusionMatrixDisplay

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
    Numărul vehiculelor implicate în accident (1 - 5).

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
    Anii de experiență ai șoferului (0 - 50 ani).

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

    # Identify non-numerical columns
    categorical_columns = [col for col, dtype in dtype_dict.items() if dtype == "string"]

    # Apply LabelEncoder to categorical columns
    le_dict = {}  # Store label encoders for inverse transformation if needed
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))  # Store mappings

    # Drop original categorical columns if needed
    df_encoded = df.drop(columns=categorical_columns)

    st.subheader("Utilizarea LabelEncoder pentru encodarea valorilor non-numerice")

    st.dataframe(df_encoded)
    
    # Display encoding mappings in a readable format
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
    accidents_within_romania.plot(ax=ax, column='Accident_Severity', cmap='coolwarm', markersize=20, legend=True)
    ax.set_axis_off()
    plt.title("Accidente simulate în România după severitate")

    st.dataframe(accidents_within_romania)

    st.pyplot(fig)

    st.title("Analiza accidentelor în funcție de vreme și tipul drumului")

    weather = st.selectbox("🌤️ Alege vremea:", accidents_within_romania["Weather"].unique())
    road_type = st.selectbox("🛣️ Alege tipul drumului:", accidents_within_romania["Road_Type"].unique())

    # Filtrare după selectbox
    filtered_data_romania = accidents_within_romania[
        (accidents_within_romania["Weather"] == weather) & (accidents_within_romania["Road_Type"] == road_type)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    romania_gdf.plot(ax=ax, color='white', edgecolor='black')
    filtered_data_romania.plot(ax=ax, color='red', markersize=30)
    ax.set_axis_off()  # Ascunde axele pentru un aspect mai curat
    plt.title(f"Accidente ({weather}, {road_type})")

    # Afișează în Streamlit
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
                        value=float(df_cleaned[column_name].median())  # Default to median
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

            expected_features = scaler.feature_names_in_  # Attribute from sklearn 1.0+
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

            ###############################################################################################
            m = modele[model]

            m.fit(X_train, Y_train)
            y_pred = m.predict(X_test)
            #mse = mean_squared_error(Y_test, y_pred)
            acc = accuracy_score(Y_test, y_pred)
            ck = cohen_kappa_score(Y_test, y_pred)
            #r2 = r2_score(Y_test, y_pred)
            roc = roc_auc_score(Y_test, y_pred)
            
            indicatori = pd.DataFrame(data={
                #"MSE" : mse,
                "Acuratețe globală" : acc,
                "Cohen Kappa" : ck,
                #"R2" : r2,
                "ROC AUC": [roc] if roc is not None else ["N/A"]
            }, index=["Valoare"])

            st.dataframe(indicatori)

            st.text("Classification Report:\n" + classification_report(Y_test, y_pred))

            # plt.scatter(Y_test, y_pred)
            # plt.xlabel("Actual")
            # plt.ylabel("Predicted")
            # plt.title(f"{model} Predictions")
            # plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Ideal line
            # plt.grid(True)
            # st.pyplot(plt)

            ConfusionMatrixDisplay.from_predictions(Y_test, y_pred)
            plt.title(f"Confusion Matrix - {model}")
            st.pyplot(plt)

            st.text("Având în vedere indicatorii obținuți pe setul de testare, predicția pentru valorile alese la început este:")

            df_scaled_values = df_scaled_values.reindex(columns=predictori_encoded)
            rezultat = m.predict(df_scaled_values)
            st.write(f"Accident: {rezultat}")
    else:
        st.warning("Mergi la pagina: '" + page2 + "' pentru a procesa setul de date principal, mai întâi.")