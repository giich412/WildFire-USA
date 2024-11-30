import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# st.title('Prédiction des feux de forêts aux USA 🔥')
from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn import model_selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay, average_precision_score, roc_auc_score  
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import folium
from streamlit_folium import st_folium
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import class_weight
import joblib
from itertools import cycle
import json
import gc

st.set_page_config(
  page_title="Projet Feux de Forêt",
  layout="wide",
  initial_sidebar_state="expanded"
  )

col1, col2, col3 = st.columns([0.6, 0.25, 0.15], gap="small", vertical_alignment="top")
with col1:
  st.title("Prédiction des feux de forêts aux USA 🔥")
with col2:
  st.write("""
           #### Promotion DA - Fév2024 :
           ###### Marie-Laure MAILLET / Gigi DECORMON         
           ###### Adoté Sitou BLIVI / Amilcar LOPEZ TELLEZ
           """)
with col3:
  st.markdown("""
              #### Mentor :
              ###### Antoine TARDIVON
              """)

st.subheader("", divider="gray")


#    page_title="Custom Themed App",
#    page_icon="🔥",

# Custom CSS to set the theme colors
#page_bg_img = """
#<style>
#body {
#    background-color: #333341;
#    color: #F5F5F5;
#}
#.sidebar .sidebar-content {
#    background-color: #DE8B29;
#}
#header .css-1d391kg {
#    background-color: #C70001;
#}
#</style>
#"""

#st.markdown(page_bg_img, unsafe_allow_html=True)
#st.markdown("""<style>.block-container {padding-top: 3rem;padding-bottom: 2rem;padding-left: 5rem;padding-right: 5rem;}</style>""", unsafe_allow_html=True)

# Mise en forme couleur du fond de l'application C70001 #DE8B29
page_bg_img="""<style>
[data-testid="stAppViewContainer"]{
background-color: #333341;
#color: #F5F5F5
opacity: 1;
#background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F5F5F5 10px ), repeating-linear-gradient( #DE8B29, #F5F5F5 );}
[data-testid="stHeader"]{
background-color: #333341;
#color: #F5F5F5
opacity: 1;
#background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F5F5F5 10px ), repeating-linear-gradient( #DE8B29, #F5F5F5 );}
[data-testid="stSidebarContent"]{
background-color: #DE8B29;
#color: #000000
opacity: 1;
#background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F5F5F5 10px ), repeating-linear-gradient( #DE8B29, #F5F5F5 );}
[data-testid="stAppViewContainer"] { padding-top: 0rem; }
div[class^='block-container'] { padding-side: 2rem; }
/* Main content text color */
h1, h2, h3, h4, h5, h6, p, div, span, a, li, label {
    color: #FFFFFF;
}
/* Select box background color */
.stSelectbox div[data-baseweb="select"] > div {
background-color: #333341;  
color: #A9A9A9;  
}

/* Dropdown menu background color */
.stSelectbox div[data-baseweb="select"] ul {
background-color: #333341;  
}

.stSelectbox div[data-baseweb="select"] ul li {
color: #A9A9A9;  
}

/* Button background color */
div.stButton > button {
background-color: #333341; 
color: black;  
}
</style>"""
st.markdown(page_bg_img,unsafe_allow_html=True)
st.markdown("""<style>.block-container {padding-top: 3rem;padding-bottom: 2rem;padding-left: 5rem;padding-right: 5rem;}</style>""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stAppViewContainer {
        color: black;  /* Set text font color to black */
    }
    </style>
    """,
    unsafe_allow_html=True
)


#Chargement dataframe sous alias df
@st.cache_data(ttl=1200)                          #cache_resource
def load_data():
  data=pd.read_csv('Firesclean_2.csv', index_col=0)  #'Firesclean_2.csv'
  # Scale down to 10 percent of the dataset
  data_sampled = data.sample(frac=0.1, random_state=42)
  return data_sampled
  return data
df=load_data()

st.sidebar.title("Sommaire")
pages=["Contexte et présentation", "Preprocessing", "DataVizualization", "Prédiction des causes de feux", "Prédiction des classes de feux", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

# Création contenu de la première page (page 0) avec le contexte et présentation du projet
if page == pages[0] : 
    st.write("### Contexte et présentation du projet")
    st.image("feu_foret.jpg", caption="Feu de forêt en Californie du Nord", width=500)
    st.markdown("""<div style="text-align: justify;">
              Nous sommes en réorientation professionnelle et cherchons à approfondir nos compétences en data analysis. Ce projet nous permet de mettre en pratique les méthodes et outils appris durant notre formation, de l’exploration des données à la modélisation et la data visualisation.
              </div>""", unsafe_allow_html=True)
    st.write("""
      ### Étapes du projet :
      - **Nettoyage et pré-processing des données**
      - **Storytelling avec DataViz** (Plotly, seaborn, matplotlib)
      - **Construction de modèles de Machine Learning**
      - **Restitution via Streamlit**
      ### Objectif :
      Le projet vise à prédire les incendies de forêt pour améliorer la prévention et l’intervention, ainsi que la détection humaine ou naturelle des départs de feu et l’évaluation des risques de grande ampleur, dans un contexte de préservation de l’environnement, de sécurité publique et d’impacts économiques significatifs.
      ### Données utilisées :
      Nous utilisons des données provenant du **US Forest Service**, qui centralise les informations sur les incendies de forêt aux États-Unis. Ces données incluent les causes des incendies, les surfaces touchées, et leurs localisations. Nous intégrons également des données météorologiques (vent, température, humidité) provenant du **National Interagency Fire Center** pour évaluer les risques de départ et de propagation des feux.
      ### Applications :
      - Prévention des incendies criminels et anticipation des feux dus à la foudre.
      - Évaluation des risques de grande taille.
      """)

# Création de la page 1 avec explication du préprocessing     
elif page == pages[1]:
    st.write("### Preprocessing")
    st.write("#### Explication de nettoyage des données")
    st.write("""
      Le jeu de données original contenait 38 colonnes et 1 880 465 lignes. Après une révision détaillée, 
      il a été décidé de ne conserver que 14 colonnes en raison de la présence de données incomplètes ou répétées dans les autres colonnes.
      Voici les colonnes originales et celles conservées :
      """)
 
    # Afficher les colonnes originales et celles conservées
    st.write("#### Colonnes Originales")
    if st.checkbox("Afficher jeu données ") :
      st.write("#### Jeu de données et statistiques")
      st.dataframe(df.head())
      st.write("#### Statistiques")
      st.dataframe(df.describe(), use_container_width=True)

    st.write("#### Colonnes Conservées")
    conserved_columns = [
          "FPA_ID", "NWCG_REPORTING_UNIT_NAME", "FIRE_YEAR", "DISCOVERY_DATE", 
          "DISCOVERY_DOY", "STAT_CAUSE_DESCR", "CONT_DOY", "FIRE_SIZE", "FIRE_SIZE_CLASS", 
          "LATITUDE", "LONGITUDE", "STATE", "FIPS_NAME"
      ]
    if st.checkbox("Afficher les colonnes conservées "):
      st.dataframe(df[conserved_columns].head())
  

    # Explication du nettoyage des données
    st.write("""
      Les colonnes conservées ont été sélectionnées car elles contiennent des informations pertinentes et complètes 
      pour l'analyse des incendies de forêt. Les colonnes supprimées avaient des données incomplètes 
      ou répétées qui n'apportaient pas de valeur significative à l'analyse.
      """)

    st.write("#### Colonnes Ajoutées")
    st.write("""
      En plus des colonnes conservées, nous avons ajouté les colonnes MONTH_DISCOVERY, DAY_OF_WEEK_DISCOVERY et DISCOVERY_WEEK à partir des transformations de la colonne DISCOVERY_DATE en format de date avec pd.to_datetime :
      - `MONTH_DISCOVERY` : Le mois de la découverte de l'incendie.
      - `DAY_OF_WEEK_DISCOVERY` : Le jour de la semaine de la découverte de l'incendie.
      - `DISCOVERY_WEEK` : Numéro de la semaine de la découverte de l’incendie.
      - `DURATION` : La durée de l'incendie en jours.
      """)
    st.write("""
      Nous avons aussi ajouté les colonnes avg temp et avg pcp à partir de la base de données du National Interagency Fire Center:
      - `AVG_TEMP [°C]` : Température moyenne en degrés Celsius
      - `AVG_PCP [mm]` : Précipitations moyennes, utilisé pour mesurer la quantité moyenne de précipitations (pluie ou neige) qui tombe dans une région spécifique sur une période de temps donnée.
      """)

    #if st.checkbox("Afficher la dimension") :
    #   st.write(f"La dimension : {df.shape}")
    st.write("""
      Nous avons éliminé les colonnes non pertinentes ou avec trop de valeurs manquantes, notamment celles liées aux codes d’identification des agences, car elles n’étaient pas utiles pour notre analyse:""")
    if st.checkbox("Afficher les na") :
      st.dataframe(df.isna().sum(), width=300, height=640)
      
# Création de la page 2 Datavizualisation
elif page == pages[2] : 
    st.header("DataVizualisation")

    # Subcategories for DataVisualization
    subpages = ["1 - Analyse des outliers", "2 - Répartition des feux par cause et classe", "3 - Répartition temporelle des feux", "4 - Répartition géographique des feux", "5 - Analyse corrélations entre variables"]
    subpage = st.radio("Aller vers", subpages)
    
    if subpage == subpages[0]:
        st.subheader("1 - Analyse des outliers et de la répartitions des valeurs numériques")
        col1, col2 = st.columns([0.55, 0.45], gap="small", vertical_alignment="center")
        with col1:
          @st.cache_data(ttl=1200)
          def plot_box():
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            sns.set(rc={"axes.facecolor": "#333341", "figure.facecolor": "#333341"}) #F4E4AA
            sns.set_theme(style='dark')
            sns.boxplot(ax=axes[0, 0], x=df['DURATION'])
            sns.boxplot(ax=axes[0, 1], x=df['FIRE_SIZE'])
            sns.boxplot(ax=axes[0, 2], x=df['AVG_PCP [mm]'])
            sns.boxplot(ax=axes[1, 0], x=df['LATITUDE'])
            sns.boxplot(ax=axes[1, 1], x=df['LONGITUDE'])
            sns.boxplot(ax=axes[1, 2], x=df['AVG_TEMP [°C]'])
            return fig
          fig = plot_box()
          st.pyplot(fig)

        #st.subheader("1 - Analyse des outliers et de la répartitions des valeurs numériques")
        #col1, col2 =st.columns([0.55, 0.45],gap="small",vertical_alignment="center")
        #with col1 :
        #  @st.cache_data(ttl=1200)
        #  def plot_violin():
        #    fig, axes = plt.subplots(2, 3,figsize=(12,7))
        #    #sns.set_style(style='white')
        #    sns.set(rc={"axes.facecolor": "#F4E4AA", "figure.facecolor": "#F4E4AA"})
        #    sns.set_theme()
        #    sns.violinplot(ax=axes[0, 0], x=df['DURATION'])
        #    sns.violinplot(ax=axes[0, 1],x=df['FIRE_SIZE'])
        #    sns.violinplot(ax=axes[0, 2],x=df['AVG_PCP [mm]'])
        #    sns.violinplot(ax=axes[1,0],x=df['LATITUDE'])
        #    sns.violinplot(ax=axes[1, 1],x=df['LONGITUDE'])
        #    sns.violinplot(ax=axes[1, 2],x=df['AVG_TEMP [°C]'])
        #    return fig
        #  fig=plot_violin()
        #  fig 

        with col2 :
          st.markdown("Certaines variables comme les tailles de feux et les durées présentent des valeurs particulièrement extrêmes.")  
          st.markdown("Certaines valeurs extrêmes paraissent impossibles et devront être écartées des analyses (exemple feux de plus de 1 an). A l'inverse les feux de taille extrêmes restent des valeurs possibles (méga feux) et seront partie intégrante de nos analyses")
          st.markdown("Les autres variables numériques (précipitations, températures ou coordonnées géographiques) nous indiquent certaines tendances marquées sur la répartition des feux que nous allons analyser plus en détail.")
  
    elif subpage == subpages[1]:
        st.subheader("2 - Répartition des feux par cause et classe")
        with st.container():
          col1, col2 = st.columns([0.6, 0.4],gap="small",vertical_alignment="center")
        with col1 :
          def plot_fire_cause_pie_charts(df):
            Fires_cause = df.groupby("STAT_CAUSE_DESCR").agg({"FPA_ID": "count", "FIRE_SIZE": "sum"}).reset_index()
            Fires_cause = Fires_cause.rename({"FPA_ID": "COUNT_FIRE", "FIRE_SIZE": "FIRE_SIZE_SUM"}, axis=1)
            Indic = ["≈ Hawaii + Massachusetts", "≈ Hawaii + Massachusetts", "≈ Washington + Georgia", "≈ Maine", "≈ New Jersey + Massachusetts"]
            Fires_cause["Text"] = Indic
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]])
            fig.add_trace(go.Pie(labels=Fires_cause["STAT_CAUSE_DESCR"], values=Fires_cause["COUNT_FIRE"], hole=0.6,
                                 direction="clockwise", title=dict(text="Nombre", font=dict(size=20))), row=1, col=1)
            fig.add_trace(go.Pie(labels=Fires_cause["STAT_CAUSE_DESCR"], values=Fires_cause["FIRE_SIZE_SUM"], hovertext=Fires_cause["Text"],
                                 hole=0.6, direction="clockwise", title=dict(text="Surfaces (acres)", font=dict(size=20))), row=1, col=2)
            fig.update_traces(textfont_size=15, sort=False, marker=dict(colors=['#F1C40F', '#F39C12', '#e74c3c', '#E67E22', '#d35400']))
            fig.update_layout(title_text="Répartition des feux par causes (1992 - 2015)", title_x=0.2, title_y=0.99, paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0.0, y=0.95, orientation="h", font=dict(family="Arial", size=12, color="white")),
                              margin=dict(l=10, r=10, t=2, b=0), titlefont=dict(size=15), width=900, height=350)
            return fig
          fig = plot_fire_cause_pie_charts(df)
          st.plotly_chart(fig)
          #joblib.dump(st.plotly_chart(fig),"répartition_feux_acres")
    
        #Pie Chart répartition par cause
        with col2 :
        #if st.checkbox("Afficher graphiques par cause") :   
          st.markdown(":orange[Les feux d’origine humaine] (volontaire et involontaire) représentent :orange[50% des départs].")
          st.markdown(":orange[Les causes naturelles] (foudre) représentent :orange[62,1% des surfaces brûlées].")
        with st.container():
          col1, col2 = st.columns([0.6, 0.40],gap="small",vertical_alignment="center")
        with col1 :
        #with st.container(height=400):
          def plot_fire_class_pie_charts(df):
            Fires_class = df.groupby("FIRE_SIZE_CLASS").agg({"FPA_ID":"count", "FIRE_SIZE":"sum"}).reset_index()
            Fires_class = Fires_class.rename({"FPA_ID":"COUNT_FIRE", "FIRE_SIZE":"FIRE_SIZE_SUM"}, axis = 1)
            Indic = ["≈ ", "≈ ","≈ Connecticut", "≈ New Jersey", "≈ Maryland", "≈ Virginie Occidentale + Delaware", "≈ Californie + Hawaii"]
            Fires_class["Text"] = Indic
            fig1= make_subplots(rows = 1, cols = 2, specs = [[{"type":"domain"}, {"type":"domain"}]])
            fig1.add_trace(go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],values = Fires_class["COUNT_FIRE"],
                 hole = 0.6, rotation = 0,title = dict(text = "Nombre", font=dict(size=20))),row = 1, col = 1)
            fig1.add_trace(go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],values = Fires_class["FIRE_SIZE_SUM"],
                 hovertext = Fires_class["Text"],hole = 0.6, rotation = -120,title = dict(text = "Surfaces (acres)", font=dict(size=20))), row = 1, col = 2)
            fig1.update_traces(textfont_size=15,sort=False,marker=dict(colors=['yellow','brown','#F1C40F', '#F39C12', '#e74c3c','#E67E22','#d35400']))
            fig1.update_layout(title_text="Répartition des feux suivant leur taille (1992 - 2015)", title_x = 0.2, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',legend=dict(x=0.2, y=0.95,orientation="h",font=dict(
                  family="Arial",size=12,color="white")),margin=dict(l=10, r=10, t=2, b=0),titlefont=dict(size=15),width=900,height=350)
            return fig1
          fig1 = plot_fire_class_pie_charts(df)
          st.plotly_chart(fig1)
          #joblib.dump(st.plotly_chart(fig1),"répartition_feux_nb")
        with col2 :      
          st.write(":orange[Les feux de petite taille (A et B, <9,9 acres)] représentent :orange[62 % du nombre de départs] mais seulement :orange[2% des surfaces brûlées].  :orange[78 % des surfaces brûlées sont liées aux feux de la classe G] (avec des feux allant de 5000 à 600 000 acres).")
    
    elif subpage == subpages[2]:
        st.subheader("3 - Répartition temporelle des feux")   
        st.write("Cet axe révèle assez clairement des périodes à risque sur les départs et la gravité des feux")
      #Histogrammes année
        st.write("**Variabilité annuelle**")
        st.write("Certaines années semblent clairement plus propices aux départs de feux. Cela peut s’expliquer par les conditions météorologiques. On observe notamment que les années où les surfaces brûlées sont significativement supérieures à la moyenne cela est dû à la foudre")
        df1=df.groupby(['STAT_CAUSE_DESCR', 'FIRE_YEAR']).agg({"FIRE_SIZE":"sum"}).reset_index()
        df1bis=df.groupby(['STAT_CAUSE_DESCR', 'FIRE_YEAR']).agg({"FPA_ID":"count"}).reset_index()
        col1, col2 = st.columns([0.5, 0.5],gap="small",vertical_alignment="center")  
        with col1 :
          @st.cache_data(ttl=1200)
          def graph_annee():
            fig2bis = px.area(df1, 'FIRE_YEAR' , "FIRE_SIZE", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR", color_discrete_sequence=px.colors.sequential.Reds_r)    
            fig2bis.update_layout(xaxis_title="",yaxis_title="",title_text="Répartition des feux par année et cause (en acres)", title_x = 0.2, title_y = 0.99,paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',width=700, height=350,legend=dict(x=0, y=1,title=None,orientation="v",font=dict(
                  family="Arial",size=11,color="white")),margin=dict(l=0, r=0, t=20, b=50),titlefont=dict(size=15),xaxis=dict(gridcolor='darkgrey'),yaxis=dict(gridcolor='darkgrey'))
            return fig2bis
          fig2bis=graph_annee()
          fig2bis

        with col2 :
          @st.cache_data(ttl=1200)
          def graph_annee_nombre():
            fig3bis = px.area(df1bis, 'FIRE_YEAR' , "FPA_ID", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR", color_discrete_sequence=px.colors.sequential.Reds_r)
            fig3bis.update_layout(xaxis_title="",yaxis_title="",title_text="Répartition des feux par année et cause (en nombre)", title_x = 0.2, title_y = 0.99,paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',width=700, height=350,legend=dict(title=None,x=0, y=1,orientation="v",font=dict(
                  family="Arial",size=11,color="white")),margin=dict(l=0, r=0, t=20, b=00),titlefont=dict(size=15),xaxis=dict(gridcolor='darkgrey'),yaxis=dict(gridcolor='darkgrey'))
            return fig3bis
          fig3bis=graph_annee_nombre()
          fig3bis

      #Histogrammes mois
        st.write("**Périodes à risque**")
        st.write("Les mois de juin à août sont les plus dévastateurs ce qui qui peut sous-entendre 2 facteurs : un climat plus favorable aux départs de feux, des activités humaines à risque plus élevées pendant les périodes de vacances")
        #if st.checkbox("Afficher graphiques mois") :
        @st.cache_data(ttl=1200)
        def hist_mois_acres_nb():
          fig3= make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("En surfaces brûlées (acres)","En Nombre de départs"))
          fig3.add_trace(go.Histogram(histfunc="sum",
            name="Surface brûlées (acres) ",
            x=df['MONTH_DISCOVERY'],y=df['FIRE_SIZE'], marker_color='orangered'),1,1)
          fig3.add_trace(go.Histogram(histfunc="count",
            name="Nombre de feux",
            x=df['MONTH_DISCOVERY'],marker_color='darkred'),1,2)
          fig3.update_layout(title_text="Départs de feux par mois",bargap=0.2,height=400, width=1100, coloraxis=dict(colorscale='Reds'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)', #coloraxis=dict(colorscale='Bluered_r')
          plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(gridcolor='darkgrey'),yaxis=dict(gridcolor='darkgrey'))
          return fig3
        fig3=hist_mois_acres_nb()
        fig3

      #Histogrammes jour semaine
        st.write("**Corrélation avec les feux d’origine humaine**")   
        st.write("On observe également des départs de feux significativement plus élevés le week-end. Ce qui peut être mis en corrélation avec les feux d'origine humaine déclenchés par des activités à risque plus propices en périodes de week-end (feux de camps...)")
        #if st.checkbox("Afficher graphiques jour de la semaine") :
        @st.cache_data(ttl=1200)
        def hist_jours_acres_nb():
          df['DAY_OF_WEEK_DISCOVERYName'] = pd.to_datetime(df['DISCOVERY_DATE']).dt.day_name()
          Fires2=df.loc[df['STAT_CAUSE_DESCR']!="Foudre"]
          Fires3 = Fires2.groupby('DAY_OF_WEEK_DISCOVERYName')['FPA_ID'].value_counts().groupby('DAY_OF_WEEK_DISCOVERYName').sum()
          week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
          Fires3 = Fires3.reindex(week)
          Fires4 = Fires2.groupby('DAY_OF_WEEK_DISCOVERYName')['FIRE_SIZE'].sum()
          week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
          Fires4 = Fires4.reindex(week)
          fig4 = make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("En surfaces brûlées (acres)","En Nombre de départs"))
          fig4.add_trace(go.Histogram(histfunc="sum",
            name="Surface brûlées (acres) ",
            x=Fires4.index,y=Fires4.values, marker_color='orangered'),1,1)
          fig4.add_trace(go.Histogram(histfunc="sum",
            name="Nombre de feux",
            x=Fires3.index,y=Fires3.values,marker_color='darkred'),1,2)
          fig4.update_layout(title_text="Départs de feux en fonction du jour de la semaine",bargap=0.2,height=400, width=1000, coloraxis=dict(colorscale='Reds'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(gridcolor='darkgrey'),yaxis=dict(gridcolor='darkgrey'))
          return fig4
        fig4=hist_jours_acres_nb()
        fig4

        df3=df.groupby(['STAT_CAUSE_DESCR', 'DISCOVERY_DOY']).agg({"FIRE_SIZE":"sum"}).reset_index()

        @st.cache_data(ttl=1200)
        def jour_cause(): 
          fig4bis = px.area(df3, 'DISCOVERY_DOY' , "FIRE_SIZE", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR", color_discrete_sequence=px.colors.sequential.Reds_r)  #Reds_r
          fig4bis.update_layout(title_text="Répartition des feux jours de l'année et cause (en acres)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,legend=dict(title=None,x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
                  family="Arial",size=15,color="white")),margin=dict(l=100, r=100, t=25, b=50),titlefont=dict(size=20),xaxis=dict(gridcolor='darkgrey'),yaxis=dict(gridcolor='darkgrey'))
          return fig4bis
        fig4bis=jour_cause()
        fig4bis

      # Durée moyenne
        st.write('L’analyse de la durée des feux par cause montre une certaine hétérogénéité de la durée des feux en fonction de la cause. Les feux liés à la foudre sont en moyenne deux fois plus longs à contenir que les autres types de feux')
        st.write("La Foudre : Les feux déclenchés par la foudre sont souvent situés dans des zones difficiles d’accès, ce qui complique les efforts de lutte contre les incendies. De plus, ces feux peuvent se propager rapidement en raison des conditions météorologiques associées aux orages, comme les vents forts.")
        #if st.checkbox("Afficher graphiques par durée") :
        @st.cache_data(ttl=1200)
        def durée(): 
          Fires_burn = df.dropna(subset=['CONT_DOY', 'DISCOVERY_DOY']).copy()
          Fires_burn['CONT_DOY'] = Fires_burn['CONT_DOY'].astype(int)
          Fires_burn['DISCOVERY_DOY'] = Fires_burn['DISCOVERY_DOY'].astype(int)
          Fires_burn['BURN_TIME'] = Fires_burn['CONT_DOY'] - Fires_burn['DISCOVERY_DOY']
          Fires_burn.loc[Fires_burn['BURN_TIME'] < 0, 'BURN_TIME'] += 365
          cause_avg_burn_time = Fires_burn.groupby('STAT_CAUSE_DESCR', dropna=True)['BURN_TIME'].mean().reset_index()
          cause_avg_burn_time.sort_values(by='BURN_TIME', ascending=False, inplace=True)
          fig5 = px.bar(
          cause_avg_burn_time,
            x='BURN_TIME',
            y='STAT_CAUSE_DESCR',
            labels={"STAT_CAUSE_DESCR": "Cause", "BURN_TIME": "Durée moyenne (jours)"},
            title='Durée moyenne des feux par cause',
            orientation='h',  # Horizontal orientation
            color='STAT_CAUSE_DESCR',
            color_discrete_sequence=px.colors.sequential.Reds_r)
          fig5.update_layout(xaxis=dict(tickmode='linear', dtick=0.5),title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,showlegend=False,margin=dict(l=170, r=200, t=50, b=50),titlefont=dict(size=20))
          return fig5
        fig5=durée()
        fig5

    elif subpage == subpages[3]:
        st.subheader("4 - Répartition géographique")
        st.markdown("On observe une densité plus élevée de surfaces brûlées à l’ouest des États-Unis, ce qui pourrait être attribué à divers facteurs tels que le climat, la végétation et les activités humaines.")
        st.markdown("**Facteurs Climatiques**- périodes de sécheresse prolongées")
        st.markdown("**Végétations**- type de végétation vulnérables aux feux et contribule à la propagation des feux")
        st.markdown("**Activités humaines**- l’urbanisation croissante dans les zones à risque, les pratiques agricoles, et les loisirs en plein air augmentent la probabilité de départs de feux")

        @st.cache_data(ttl=1200)
        def load_FiresClasse():
            Fires_bis = df.copy()
            modif1 = ["Campfire", "Debris Burning", "Smoking", "Fireworks", "Children"]
            modif2 = ["Powerline", "Railroad", "Structure", "Equipment Use"]
            Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace("Missing/Undefined", "Miscellaneous")
            Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace(modif1, "Origine humaine")
            Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace(modif2, "Equipnt Infrastr")
            Fires_bis["FIRE_SIZE_CLASS"] = Fires_bis["FIRE_SIZE_CLASS"].replace(["A", "B", "C"], "ABC")
            Fires_bis["FIRE_SIZE_CLASS"] = Fires_bis["FIRE_SIZE_CLASS"].replace(["D", "E", "F"], "DEF")
            FiresClasse = Fires_bis[(Fires_bis['FIRE_SIZE_CLASS'] != "ABC")]
            return FiresClasse
        FiresClasse = load_FiresClasse()

#        @st.cache_data(ttl=1200)
#        def plot_fire_geo_scatter(FiresClasse):
#          fig6 = px.scatter_geo(
#            FiresClasse,
#            lon=FiresClasse['LONGITUDE'],
#            lat=FiresClasse['LATITUDE'],
#            color="STAT_CAUSE_DESCR",
#            #facet_col="FIRE_YEAR", # Uncomment to create a graph per year
#            #facet_col_wrap, # Uncomment to define the number of graphs per row
#            #animation_frame="FIRE_YEAR", # Uncomment to create an animation over the years
#            color_discrete_sequence=["blue", "orange", "red", "grey", "purple"],
#            labels={"STAT_CAUSE_DESCR": "Cause"},
#            hover_name="STATE", # Column added to hover information
#            size=FiresClasse['FIRE_SIZE'] / 1000, # Size of markers
#            projection='albers usa',
#            locationmode='USA-states',
#            width=800,
#            height=500,
#            basemap_visible=True
#            )
#          fig6.update_geos(
#            resolution=50,
#            lataxis_showgrid=True,
#            lonaxis_showgrid=True,
#            bgcolor='rgba(0,0,0,0)',
#            framecolor='blue',
#            showframe=True,
#            showland=True,
#            landcolor='#e0efe7',
#            projection_type="albers usa"
#            )
#          fig6.update_layout(
#            title_text="Répartition géographique des feux par cause et taille",
#            title_x=0.3,
#            title_y=0.95,
#            paper_bgcolor='rgba(0,0,0,0)',
#            plot_bgcolor='rgba(0,0,0,0)',
#            width=1000,
#            height=500,
#            legend=dict(
#              x=0.5,
#              y=1.05,
#              orientation="h",
#              xanchor="center",
#              yanchor="bottom",
#              font=dict(
#                family="Arial",
#                size=15,
#                color="black"
#                )
#            ),
#            margin=dict(l=50, r=50, t=100, b=50),
#            titlefont=dict(size=20)
#            )
#          return fig6
#        fig6 = plot_fire_geo_scatter(FiresClasse)
#        st.plotly_chart(fig6)
#        gc.collect()

        #if st.checkbox("Afficher graphiques répartition géographique et année") :
#        col1, col2 = st.columns(2)
#
#        with col1:
#          @st.cache_data(persist=True)
#          def scatter_geo_global():
#            fig7 = px.scatter_geo(FiresClasse,
#              lon = FiresClasse['LONGITUDE'],
#              lat = FiresClasse['LATITUDE'],
#              color="STAT_CAUSE_DESCR",
#              #facet_col="FIRE_YEAR", #pour créer un graph par année
#              #facet_col_wrap,# pour définir le nombre de graph par ligne
#              #animation_frame="FIRE_YEAR",#pour créer une animation sur l'année
#              color_discrete_sequence=["blue","orange","red","grey","purple"],
#              labels={"STAT_CAUSE_DESCR": "Cause"},
#              hover_name="STATE", # column added to hover information
#              size=FiresClasse['FIRE_SIZE']/1000, # size of markers
#              projection='albers usa',
#              width=800,
#              height=500,
#              title="Répartition géographique des feux par cause et taille",basemap_visible=True)
#            fig7.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True,bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='#e0efe7',projection_type="albers usa")
#            fig7.update_layout(title_text="Répartition géographique des feux par cause et taille", title_x = 0.1, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
#            plot_bgcolor='rgba(0,0,0,0)',width=1000, height=700,legend=dict(title=None,x=0.5, y=0.85,orientation="h",xanchor="center",yanchor="bottom",font=dict(
#                  family="Arial",size=11,color="black")),margin=dict(l=0, r=0, t=50, b=290),titlefont=dict(size=18))   
#            return fig7
#        fig7=scatter_geo_global()
#        fig7
#        gc.collect()
#        #joblib.dump(st.plotly_chart(fig7),"répartition_géo")

#        with col2:
        @st.cache_data(ttl=1200)
        def scatter_geo():
            fig7_ = px.scatter_geo(FiresClasse,
                lon = FiresClasse['LONGITUDE'],
                lat = FiresClasse['LATITUDE'],
                color="STAT_CAUSE_DESCR",
                #facet_col="FIRE_YEAR", #pour créer un graph par année
                #facet_col_wrap,# pour définir le nombre de graph par ligne
                animation_frame="FIRE_YEAR",#pour créer une animation sur l'année
                color_discrete_sequence=["blue","orange","red","grey","purple"],
                labels={"STAT_CAUSE_DESCR": "Cause"},
                hover_name="STATE", # column added to hover information
               size=FiresClasse['FIRE_SIZE']/1000, # size of markers
                projection='albers usa',
                width=800,
                height=500,
                title="Répartition géographique des feux par cause et taille",basemap_visible=True)
            fig7_.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True, bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='rgb(100,100,100)',projection_type="albers usa") #bgcolor='rgba(0,0,0,0)', landcolor='#e0efe7'
            fig7_.update_layout(title_text="Répartition géographique des feux par cause et taille", title_x = 0.3, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',width=1500, height=800,legend=dict(title=None,x=0.5, y=0.95,orientation="h",xanchor="center",yanchor="bottom",font=dict(
                  family="Arial",size=11,color="white")),margin=dict(l=0, r=0, t=100, b=50),titlefont=dict(size=18))   
            return fig7_
        fig7_=scatter_geo()
        fig7_
        gc.collect()
          #joblib.dump(st.plotly_chart(fig7_),"répartition_géo_mois")

    elif subpage == subpages[4]:
        st.subheader("5 - Analyse corrélations entre variables")       
        # Plot heatmap - correlation matrix for all numerical columns
        #style.use('ggplot')
        #if st.checkbox("Afficher heatmap") :
        st.write('Cette matrice permet d’identifier quelles variables ont de fortes corrélations entre elles, ce qui nous aide à **sélectionner les caractéristiques les plus pertinentes** pour notre modèle de Machine Learning.')
        st.write('Elles nous permettent de **comprendre les relations entre les variables**, d’améliorer la performance et l’interprétabilité du modèle en réduisant le bruit et en se concentrant sur les variables les plus influentes.')
        
        @st.cache_data(ttl=1200)
        def heat_map():
          df_Fires_ML_num = df.select_dtypes(include=[np.number])
          plt.subplots(figsize = (7,7))
          sns.set_style(style='white')
          sns.set(rc={"axes.facecolor": "#333341", "figure.facecolor": "#333341"})
          df_Fires_ML_num = df.select_dtypes(include=[np.number])
          mask = np.zeros_like(df_Fires_ML_num.corr(), dtype='bool')
          mask[np.triu_indices_from(mask)] = True
          fig7b, ax = plt.subplots(figsize = (10,7))
          sns.heatmap(df_Fires_ML_num.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, center = 0, mask=mask, annot_kws={"size": 8}, ax=ax)
          plt.title("Heatmap of all the selected features of data set", fontsize = 13, color='white')
          ax.tick_params(colors='white', labelsize = 8)  
          ax.xaxis.label.set_color('white')  
          ax.xaxis.label.set_size(8) 
          ax.yaxis.label.set_color('white')
          ax.yaxis.label.set_size(8)
          ax.grid(False) 
          return fig7b
        fig7b=heat_map()
        fig7b 
        gc.collect()
        st.write("En analysant ces données plus en détail, on peut mieux comprendre les facteurs qui contribuent aux feux. Ces données soulignent l’importance de la prévention des feux de foret d’origine humaine et de la gestion des risques naturels pour minimiser les dégâts causés par les feux de forêt.")
  
# Modèles de prédiction des causes
if page == pages[3] : 
  st.write("## Prédiction des causes de feux")

  # Suppression des variables non utiles au ML
  Drop_col_ML = ["NWCG_REPORTING_UNIT_NAME", "FPA_ID","DISCOVERY_DATE","DISCOVERY_DOY","DISCOVERY_TIME","CONT_DOY","CONT_DATE","CONT_TIME","FIRE_SIZE","STAT_CAUSE_DESCR","COUNTY","FIPS_NAME"] 
  Fires35 = df.dropna()
  Fires_ML = Fires35.drop(Drop_col_ML, axis = 1)
  # Suppression des lignes de "STATE", "AVG_TEMP [°C]", "AVG_PCP [mm]" ayant des données manquantes 
  Fires_ML = Fires_ML.dropna(subset = ["STATE", "AVG_TEMP [°C]", "AVG_PCP [mm]"])

  st.subheader("Objectif 🎯", divider="blue") 
  st.write("""L'objectif de cette section est de :
  - définir les causes de feux
  - construire des modèles de prédictions capables de déterminer la cause d'un feu à partir des caractéristiques de celui-ci.""")
  
  # Création d'une checkbox pour afficher la distribution des causes avant et après regroupement
  # Nouvelle distribution des causes suite au regroupement des causes initiales
  Fires_ML = Fires_ML[(Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 9) & (Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 13)]

  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace([3, 4, 5, 8, 2, 6, 10, 11, 12], 20)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(7, 21)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(1, 22)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace({20: 0, 21: 1, 22: 2})
  # Sauvegarde des 200 dernières lignes du jeu pour la validation finale et l'interactivité
  Fires_test2 = Fires_ML.iloc[-200:, :]
  Fires_ML = Fires_ML.iloc[:-200, :]
  #gc.collect()

  # Check shapes and indices
  print("Fires_ML shape after preprocessing:", Fires_ML.shape)
  print("Fires_test2 shape:", Fires_test2.shape)
  print("Fires_ML indices:", Fires_ML.index)
  print("Fires_test2 indices:", Fires_test2.index)

  # Verify no NaN values
  assert not Fires_ML.isnull().values.any(), "Fires_ML contains NaN values"
  assert not Fires_test2.isnull().values.any(), "Fires_test2 contains NaN values"

  gc.collect()


  st.subheader("Regroupement des causes 🗂️", divider="blue") 
  if st.checkbox("Cliquez pour voir les causes de feux"):
    col1, col2= st.columns(spec = 2, gap = "large")
    with col1:
      st.write("#### Distribution initiale des causes de feux")
      count = Fires_ML["STAT_CAUSE_DESCR_1"].value_counts()
      color = ["blue", "orange", "red", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]
      fig, ax = plt.subplots(figsize=(12, 9), facecolor='none') 
      ax.bar(count.index, count.values, label = Fires_ML["STAT_CAUSE_DESCR_1"].unique(), color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize=20, color='white')
      ax.set_xticks(range(len(count.index)))
      ax.set_xticklabels(count.index, rotation=75, fontsize=18, color='white')
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      ax.grid(color='black')
      ax.spines['bottom'].set_color('black')
      ax.spines['top'].set_color('black')
      ax.spines['left'].set_color('black')
      ax.spines['right'].set_color('black')
      st.pyplot(fig)
      st.markdown("""<div style="text-align: justify;">
                  On observe un grand déséquilibre du jeu de données. Ce qui va rendre complexe la prédiction de l'analyse.
                  Les feux Missing/Undefined et Miscellaneous représentent environ le quart des données.
                  Compte tenu de leur caractère inerte par rapport à l'objectif de l'étude, nous les supprimerons.
                  Pour les diverses qui peuvent se ressembler, nous procéderons à leur regroupement dans une cause parente
                  </div>""", unsafe_allow_html=True)
      gc.collect()
    with col2:
      st.write("#### Distribution des causes après regroupement")
      count2 = Fires_ML["STAT_CAUSE_CODE"].value_counts()
      color = ["blue", "orange", "red"]
      fig, ax = plt.subplots(figsize=(12, 9), facecolor='none')  
      ax.bar(count2.index, count2.values, color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize = 20, color='white')
      ax.set_xticks([0, 1, 2])
      ax.set_xticklabels(["Humaine", "Criminelle", "Naturelle"], rotation = 25, fontsize = 18, color='white')
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      ax.grid(color='black')
      ax.spines['bottom'].set_color('black')
      ax.spines['top'].set_color('black')
      ax.spines['left'].set_color('black')
      ax.spines['right'].set_color('black')
      st.pyplot(fig)

      st.write("Suppresion des causses non-définies : Missing/Undefined, Miscellaneous, Others")
      st.write("""
               Regroupement des feux en 3 principales causes :
               - **:blue[Humaine] (0)** : Debris burning, Campfire, Children, Smoking, Equipment Use, Railroad, Powerline, Structure, Fireworks"
               - **:red[Criminelle] (1)** : Arson"
               - **:orange[Naturelle] (2)** : Ligthning
               """)
      gc.collect()

  ######################################################################################################################################################################
  ### Fonctions de preprocessing du jeu de données pour le ML ##########################################################################################################
  ######################################################################################################################################################################

  # Séparation des variables du target
  Fires_ML = Fires_ML.drop("STAT_CAUSE_DESCR_1", axis = 1)
  @st.cache_data(ttl=1200)
  def data_labeling(data):
    # Remplacement des jours de la semaine par 1 à 7 au lieu de 0 à 6
    data["DAY_OF_WEEK_DISCOVERY"] = data["DAY_OF_WEEK_DISCOVERY"].replace({0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7})
    # Data preparation for time-series split
    data.sort_values(["FIRE_YEAR", "MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"], inplace = True)
    # Fires_ML.set_index("FIRE_YEAR", inplace = True)
    feats, target = data.drop("STAT_CAUSE_CODE", axis = 1), data["STAT_CAUSE_CODE"]
    # OneHotEncoding des variables catégorielles avec get_dummies avec le train_test_split
    feats = pd.get_dummies(feats, dtype = "int")
    return feats, target
  feats, target = data_labeling(Fires_ML)
  #gc.collect()

  # Séparation du jeu en train et test
  @st.cache_data(ttl=1200)
  def data_split(X, y):
    # Data split of features and target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)
    # display(feats.shape, X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
  X_train, X_test, y_train, y_test = data_split(feats, target)
  #gc.collect()


  # Traitement des variables cycliques
  @st.cache_data(ttl=1200)
  def cyclic_transform(X_train, X_test):
    # Séparation des variables suivant leur type
    circular_cols_init = ["MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"]
    
    # Fit on training data
    circular_data_train = X_train[circular_cols_init].copy()
    ##circular_data = X[circular_cols_init].copy()
    # Encodage des variables temporelles cycliques
    circular_data_train["SIN_MONTH"] = circular_data_train["MONTH_DISCOVERY"].apply(lambda m: np.sin(2*np.pi*m/12))     #.loc
    circular_data_train["COS_MONTH"] = circular_data_train["MONTH_DISCOVERY"].apply(lambda m: np.cos(2*np.pi*m/12))
    circular_data_train["SIN_WEEK"] = circular_data_train["DISCOVERY_WEEK"].apply(lambda w: np.sin(2*np.pi*w/53))
    circular_data_train["COS_WEEK"] = circular_data_train["DISCOVERY_WEEK"].apply(lambda w: np.cos(2*np.pi*w/53))
    circular_data_train["SIN_DAY"] = circular_data_train["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.sin(2*np.pi*d/7))
    circular_data_train["COS_DAY"] = circular_data_train["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.cos(2*np.pi*d/7))
    # Suppression des variables cycliques sources pour éviter le doublon d'informations
    circular_data_train = circular_data_train.drop(circular_cols_init, axis = 1).reset_index(drop = True)
    # Fit on training data
    circular_data_test = X_test[circular_cols_init].copy()
    ##circular_data = X[circular_cols_init].copy()
    # Encodage des variables temporelles cycliques
    circular_data_test["SIN_MONTH"] = circular_data_test["MONTH_DISCOVERY"].apply(lambda m: np.sin(2*np.pi*m/12))     #.loc
    circular_data_test["COS_MONTH"] = circular_data_test["MONTH_DISCOVERY"].apply(lambda m: np.cos(2*np.pi*m/12))
    circular_data_test["SIN_WEEK"] = circular_data_test["DISCOVERY_WEEK"].apply(lambda w: np.sin(2*np.pi*w/53))
    circular_data_test["COS_WEEK"] = circular_data_test["DISCOVERY_WEEK"].apply(lambda w: np.cos(2*np.pi*w/53))
    circular_data_test["SIN_DAY"] = circular_data_test["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.sin(2*np.pi*d/7))
    circular_data_test["COS_DAY"] = circular_data_test["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.cos(2*np.pi*d/7))
    # Suppression des variables cycliques sources pour éviter le doublon d'informations
    circular_data_train = circular_data_train.drop(circular_cols_init, axis = 1).reset_index(drop = True)
    # Récupération des noms de colonnes des nouvelles variables
    # circular_cols = circular_data.columns
    return circular_data_train, circular_data_test
  circular_train, circular_test = cyclic_transform(X_train, X_test)#, cyclic_transform(X_test)
  gc.collect()
    
  # Traitement des variables numériques
  @st.cache_data(ttl=1200)
  def num_imputer(X_train, X_test):
      circular_cols_init = ["MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"]
      num_cols = X_train.drop(circular_cols_init, axis=1).columns
    
      # Fit on training data
      X_train_num = X_train[num_cols].copy()
      numeric_imputer = SimpleImputer(strategy="median")
      X_train_num_imputed = numeric_imputer.fit_transform(X_train_num)
      X_train_num_imputed = pd.DataFrame(X_train_num_imputed, columns=num_cols)
    
      # Apply to test data
      X_test_num = X_test[num_cols].copy()
      X_test_num_imputed = numeric_imputer.transform(X_test_num)
      X_test_num_imputed = pd.DataFrame(X_test_num_imputed, columns=num_cols)
    
      return X_train_num_imputed, X_test_num_imputed

  num_train_imputed, num_test_imputed = num_imputer(X_train, X_test)

#  # Traitement des variables numériques
#  @st.cache_data(ttl=1200)
#  def num_imputer(X):
#    circular_cols_init = ["MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"]
#    num_cols = feats.drop(circular_cols_init, axis = 1).columns
#    X_num = X[num_cols].copy()
#    # Instanciation de la méthode SimpleImputer
#    numeric_imputer = SimpleImputer(strategy = "median")
#    # Initialisation des variables
#    CLASS = ["FIRE_SIZE_CLASS_A", "FIRE_SIZE_CLASS_B", "FIRE_SIZE_CLASS_C", "FIRE_SIZE_CLASS_D", "FIRE_SIZE_CLASS_E", 
#             "FIRE_SIZE_CLASS_F", "FIRE_SIZE_CLASS_G"]
#    sub_col = ["DURATION","FIRE_SIZE_CLASS_A", "FIRE_SIZE_CLASS_B", "FIRE_SIZE_CLASS_C", "FIRE_SIZE_CLASS_D", 
#               "FIRE_SIZE_CLASS_E", "FIRE_SIZE_CLASS_F", "FIRE_SIZE_CLASS_G"]
#    sub_num_data = X_num[sub_col].copy()
#    num_data = sub_num_data.copy()
#    for fire_class in CLASS:
#        num_imputed = numeric_imputer.fit_transform(sub_num_data[sub_num_data[fire_class] == 1])
#        num_data[num_data[fire_class] == 1] = num_imputed  #.loc
#    X_num["DURATION"] = num_data["DURATION"] #.loc
#    X_num = X_num.reset_index(drop = True)
#    return X_num

#  num_train_imputed, num_test_imputed = num_imputer(X_train), num_imputer(X_test)
#  #gc.collect()
  
  # Reconstitution du jeu de données après traitement
  @st.cache_data(ttl=1200)
  def X_concat(X_train_num, X_test_num, circular_train, circular_test):
      X_train_final = pd.concat([X_train_num, circular_train], axis=1)
      X_test_final = pd.concat([X_test_num, circular_test], axis=1)
    
      X_train_final = X_train_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
      X_test_final = X_test_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
    
      overall_col = X_train_final.columns
    
      return X_train_final, X_test_final, overall_col

  X_train_final, X_test_final, overall_col = X_concat(num_train_imputed, num_test_imputed, circular_train, circular_test)

  # Verify no NaN values in the datasets
  assert not X_train_final.isnull().values.any(), "X_train_final contains NaN values"
  assert not y_train.isnull().values.any(), "y_train contains NaN values"

  print("No NaN values found in the datasets.")

  gc.collect()

  # Reconstitution du jeu de données après traitement
#  @st.cache_data(ttl=1200)
#  def X_concat(X_train_num, X_test_num, circular_train, circular_test):
#    X_train_final = pd.concat([X_train_num, circular_train], axis = 1)
#    X_test_final = pd.concat([X_test_num, circular_test], axis = 1)
#    X_train_final = X_train_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
#    X_test_final = X_test_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
#    X_total = pd.concat([X_train_final, X_test_final], axis = 0)
#   y_total = pd.concat([y_train, y_test], axis = 0)
#    overall_col = X_train_final.columns
#    return X_train_final, X_test_final, overall_col
#  X_train_final, X_test_final, overall_col = X_concat(num_train_imputed, num_test_imputed, circular_train, circular_test)
#  #gc.collect()

#  @st.cache_resource
#  def model_reduction(classifier, X_train, y_train):
#    # Check Data Shapes
#    print("X_train shape:", X_train.shape)
#    print("y_train shape:", y_train.shape)
    
    # Verify Data Integrity
#    assert not X_train.isnull().values.any(), "X_train contains NaN values"
#    assert not y_train.isnull().values.any(), "y_train contains NaN values"
    
    # Align Indices
#    X_train = X_train.reset_index(drop=True)
#    y_train = y_train.reset_index(drop=True)
    
#    if classifier == "XGBoost":
#        clf = XGBClassifier(tree_method="approx", objective="multi:softprob").fit(X_train.head(1000), y_train.head(1000))
#        feat_imp_data = pd.DataFrame(list(clf.get_booster().get_fscore().items()),
#                                     columns=["feature", "importance"]).sort_values('importance', ascending=True)
#        fea#t_imp = list(feat_imp_data["feature"][-11:])
#    elif classifier == "Regression Logistique":
#        clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
#        coefficients = clf.coef_
#        avg_importance = np.mean(np.abs(coefficients), axis=0)
#        feat_imp_data = pd.DataFrame({"feature": X_train.columns, "importance": avg_importance}).sort_values('importance', ascending=True)
#        feat_imp = list(feat_imp_data["feature"][-11:])
#    elif classifier == "Arbre de Décision":
#        clf = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X_train, y_train)
#        feat_imp_data = pd.DataFrame(clf.feature_importances_,
#                                     index=X_train.columns, columns=["importance"]).sort_values('importance', ascending=True)
#        feat_imp = list(feat_imp_data.index[-11:])
    
#    return feat_imp

#gc.collect()

# Example usage (assuming X_train_final and y_train are defined)
# feat_imp = model_reduction("XGBoost", X_train_final, y_train)




  # Réduction du modèle avec la méthode feature importances
  @st.cache_resource
  def model_reduction(classifier, X_train, y_train):
    if classifier == "XGBoost":
       clf = XGBClassifier(tree_method = "approx",
                           objective = "multi:softprob").fit(X_train, y_train)
       feat_imp_data = pd.DataFrame(list(clf.get_booster().get_fscore().items()),
                                    columns=["feature", "importance"]).sort_values('importance', ascending=True)
       feat_imp = list(feat_imp_data["feature"][-11:])
    #elif classifier == "Random Forest":
    #   clf = RandomForestClassifier().fit(X_train, y_train)
    #   feat_imp_data = pd.DataFrame(clf.feature_importances_,
    #                                index=X_train.columns, columns=["importance"]).sort_values('importance', ascending=True)
    #   feat_imp = list(feat_imp_data.index[-11:])
    elif classifier == "Regression Logistique":
       clf = LogisticRegression(random_state = 42, max_iter=1000).fit(X_train, y_train)
       coefficients = clf.coef_
       avg_importance = np.mean(np.abs(coefficients), axis = 0)
       feat_imp_data = pd.DataFrame({"feature":X_train.columns, "importance":avg_importance}).sort_values('importance', ascending=True)
       feat_imp = list(feat_imp_data["feature"][-11:])
    elif classifier == "Arbre de Décision":
       clf = DecisionTreeClassifier(criterion = "gini", random_state = 42).fit(X_train, y_train)
       feat_imp_data = pd.DataFrame(clf.feature_importances_,
                                    index=X_train.columns, columns=["importance"]).sort_values('importance', ascending=True)
       feat_imp = list(feat_imp_data.index[-11:])
    return feat_imp
    #gc.collect() 

  ######################################################################################################################################################################
  ### Fonctions de visualisation des métriques et graphes ##############################################################################################################
  ######################################################################################################################################################################  
  
  # Tracé des courbes Precision_Recall  
  @st.cache_data(ttl=1200)
  def multiclass_PR_curve(_classifier, X_test, y_test):
    y_test= label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test.shape[1]
    # Predict probabilities
    y_score = _classifier.predict_proba(X_test)
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    # n_classes = y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Plot Precision-Recall curve for each class using Plotly
    fig = go.Figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        fig.add_trace(go.Scatter(x=recall[i], y=precision[i], mode='lines', name=f'Class {i} (AP = {average_precision[i]:0.2f})', line=dict(color=color)))

    fig.update_layout(
        title={'text': 'Courbe Precision-Recall', 'font': {'color': 'white'}},
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend_title='Classes'
    )
    return fig 
    #gc.collect()
  ######################################################################################################################################################################
  ### Fonctions d'entrainement sur l'ensemble du jeu d'entrainement et de sauvegarde des meilleurs modèles #############################################################
  ######################################################################################################################################################################  

  # Enregistrement des meilleurs modèles
  # Best xgb raw model
  @st.cache_resource
  def best_xgb_raw_model(X, y):
      xgb_best_params = {"learning_rate": 0.015,
                        "max_depth": 3, 
                        "n_estimators": 10}
      clf_xgb = XGBClassifier(objective = "multi:softprob",
                             tree_method = "approx",
                             **xgb_best_params).fit(X, y)
      joblib.dump(clf_xgb, "best_xgb_raw_model.joblib")
      model = joblib.load("best_xgb_raw_model.joblib")
      return model
      #gc.collect()

  # Best xgb raw model
  #@st.cache_resource
  #def best_rf_raw_model(X, y):
  #   rf_best_params = {"n_estimators": 50,
  #                    "max_depth": 100,
  #                    "min_samples_leaf": 40,
  #                    "min_samples_split": 50,
  #                    "max_features": 'sqrt'}
  #   clf_rf = RandomForestClassifier(**rf_best_params).fit(X, y)
  #   joblib.dump(clf_rf, "best_rf_raw_model.joblib")
  #   model = joblib.load("best_rf_raw_model.joblib")
  #   return model
  #   gc.collect()

  # Best LogReg raw model
  @st.cache_resource
  def best_LogReg_raw_model(X, y):
     LogReg_best_params = {"C": 0.1,
                      "solver": "sag",
                      "max_iter":1000}
     clf_LogReg = LogisticRegression(**LogReg_best_params, random_state = 42).fit(X, y)
     joblib.dump(clf_LogReg, "best_lr_raw_model.joblib")
     model = joblib.load("best_lr_raw_model.joblib")
     return model
     gc.collect()

  # Best Decision Tree raw model
  @st.cache_resource
  def best_DecTree_raw_model(X, y):
     DecTree_best_params = {"criterion": "gini"}
     clf_DecTree = DecisionTreeClassifier(**DecTree_best_params, random_state = 42).fit(X, y)
     joblib.dump(clf_DecTree, "best_DecTree_raw_model.joblib")
     model = joblib.load("best_DecTree_raw_model.joblib")
     return model
     gc.collect()

  ######################################################################################################################################################################
  ### Fonctions de labelisation de set de données fournies en input pour une prédiction en temps réelles (en phase de production) ######################################
  ######################################################################################################################################################################  

  # Labélisation des nouvelles données de prédiction
  @st.cache_data(ttl=1200)
  def real_data_process(data):
    # Initialisation du dataframe pour le ML
    data_shape = data.shape
    X = pd.DataFrame(columns = overall_col, index=range(data_shape[0]))
    # Remplacement des jours de la semaine par 1 à 7 au lieu de 0 à 6
    data["DAY_OF_WEEK_DISCOVERY"] = data["DAY_OF_WEEK_DISCOVERY"].replace({0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7})
    # Data preparation for time-series split
    # Transformation cyclique
    data_cyclic = cyclic_transform(data)
    data[overall_col[-6:]] = data_cyclic
    # Intégration des input à la grille labellisée du ML
    for col in overall_col:
       if col in data.columns:
          X[col] = data[col]
    # OneHot Encoding manuel des FIRE_SIZE_CLASS et STATE
    FIRE_CLASS = ["FIRE_SIZE_CLASS_A", "FIRE_SIZE_CLASS_B", "FIRE_SIZE_CLASS_C", "FIRE_SIZE_CLASS_D", "FIRE_SIZE_CLASS_E", "FIRE_SIZE_CLASS_F", "FIRE_SIZE_CLASS_G"]
    for i in range(data_shape[0]):
       for fire_class in FIRE_CLASS:
          if data.loc[i, "FIRE_SIZE_CLASS"] == fire_class[-1]:
             X.loc[i, fire_class] = 1
          else:
             X.loc[i, fire_class] = 0
       for state in (overall_col[14:-6]):
          if data.loc[i, "STATE"] == state[6:]:
             X.loc[i, state] = 1
          else:
             X.loc[i, state] = 0
    return X
    gc.collect()
  

    # FIPS CODE par STATE

  ######################################################################################################################################################################
  ### Code pour l'interface streamlit ##################################################################################################################################
  ######################################################################################################################################################################  

  classifier = st.selectbox("Veuillez sélectionner un modèle",("XGBoost", "Random Forest", "Regression Logistique","Arbre de Décision"),key="model_selector")
  # st.sidebar.subheader("Veuillez sélectionner les paramètres")

  ###################################
  ### Test des différents modèles ###
  ###################################
  st.subheader("Développement METIER - A vous de jouer :smiley:", divider="blue") 
  if st.checkbox("Voulez-vous tester les différents modèles ?"):
    Feature_importances = st.sidebar.radio("Voulez-vous réduire la dimension du jeu ?", ("Oui", "Non"), horizontal=True)
    if Feature_importances == "Oui":
      feat_imp = model_reduction(classifier, X_train_final, y_train)
      st.write("Les variables sont réduites")
      X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
    else:
      X_train_final, X_test_final = X_train_final, X_test_final
      st.write("Les variables ne sont pas réduites")
      #gc.collect()
    
    # Définition des paramètres des modèles sur l'interface streamlit
    ####################################
    ###        Modèle XGBoost        ###
    ####################################
    if classifier == "XGBoost": 
      # Ré-équilibrage ou non des données 
      class_weights_option = st.sidebar.radio("Voulez-vous rééquilibrer les classes ?", ["Oui", "Non"], horizontal=True)
      if class_weights_option == "Oui":
        classes_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        st.write("Les classes sont ré-équilibrées")
      elif class_weights_option == "Non":
        classes_weights = None
        st.write("Les classes ne sont pas ré-équilibrées.")

      n_estimators = st.sidebar.slider("Veuillez choisir le nombre d'estimateurs", 5, 30, 10, 5)
      tree_method = st.sidebar.radio("Veuillez choisir la méthode", ("approx", "hist"), horizontal=True)
      max_depth = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 3, 20, 5)
      learning_rate = st.sidebar.slider("Veuillez choisir le learning rate", 0.05, 0.25, 0.1, 0.05)
      gc.collect() 
    ####################################
    ###     Modèle Random Forest     ###
    ####################################
#elif classifier == "Random Forest":
#      class_weights_option = st.sidebar.radio("Voulez-vous rééquilibrer les classes ?", ["Oui", "Non"], horizontal=True)
#      if class_weights_option == "Oui":
#          class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
#          classes_weights = {i: weight for i, weight in enumerate(class_weights_array)}
#          st.write("Les classes sont ré-équilibrées")
#      else:
#          classes_weights = None
#          st.write("Les classes ne sont pas ré-équilibrées.")
#      n_estimators = st.sidebar.slider("Veuillez choisir le nombre d'estimateurs", 5, 30, 10, 5)
#      max_depth = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 3, 10)
#      min_samples_leaf = st.sidebar.slider("Veuillez choisir min_samples_leaf", 20, 60, 40, 5)
#      min_samples_split = st.sidebar.slider("Veuillez choisir min_samples_split", 30, 100, 100, 5)      
#      max_features = st.sidebar.radio("Veuillez choisir le nombre de features", ("sqrt", "log2"), horizontal=True)
#      gc.collect()
    ####################################
    ### Modèle Regression Logistique ###
    ####################################
    elif classifier == "Regression Logistique":
      class_weights_option = st.sidebar.radio("Voulez-vous rééquilibrer les classes ?", ["Oui", "Non"], horizontal=True)
      if class_weights_option == "Oui":
         classes_weights = 'balanced'
      else:
         classes_weights = None
      max_iter = st.sidebar.slider("Veuillez choisir le nombre d'itérations", 100, 2000, 1000, 50)
      solver = st.sidebar.radio("Veuillez choisir le solveur", ("lbfgs", "newton-cg", "sag", "saga"), horizontal=True)
      gc.collect()
    ####################################
    ###   Modèle Arbre de Décision   ###
    ####################################
    elif classifier == "Arbre de Décision":
      class_weights_option = st.sidebar.radio("Voulez-vous rééquilibrer les classes ?", ["Oui", "Non"], horizontal=True)
      if class_weights_option == "Oui":
         classes_weights = 'balanced'
      else:
         classes_weights = None
      criterion = st.sidebar.radio("Veuillez choisir le critère", ["gini", "entropy"], horizontal=True)
      max_features = st.sidebar.radio("Veuillez choisir le critère", ["log2", "sqrt"], horizontal=True)
      max_depth = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 10, 200, 10, 10)
      min_samples_split = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 10, 50, 20, 5)
      min_samples_leaf = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 10, 100, 50, 5)
      gc.collect()

    # Création d'un bouton pour le modèle avec les meilleurs paramètres
    if st.sidebar.button("Best Model Execution"):
      if classifier == "XGBoost":
        st.subheader("XGBoost Result")
        best_params = {"learning_rate": 0.015, 
                        "max_depth": 3, 
                        "n_estimators": 10}
        classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y = y_train)
        feat_imp = model_reduction(classifier, X_train_final, y_train)
        X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
        clf_xgb_best = XGBClassifier(objective = "multi:softprob",
                                      tree_method = "approx",
                                      **best_params).fit(X_train_final, y_train, sample_weight = classes_weights)
        # Enrégistrement du meilleur modèle
        joblib.dump(clf_xgb_best, "clf_xgb_best_model.joblib")
        # Chargement du meilleur modèle
        model = joblib.load("clf_xgb_best_model.joblib")
        gc.collect()

      #elif classifier == "Random Forest":
      #  st.subheader("Random Forest Result")
      #  best_params = {"n_estimators": 50,
      #                "max_depth": 100,
      #                "min_samples_leaf": 40,
      #                "min_samples_split": 50,
      #                "max_features": 'sqrt'}
      #  feat_imp = model_reduction(classifier, X_train_final, y_train)
      #  X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
      #  clf_rf_best = RandomForestClassifier(**best_params, class_weight='balanced').fit(X_train_final, y_train)
      #  joblib.dump(clf_rf_best, "clf_rf_best_model.joblib")
        # Chargement du meilleur modèle
      #  model = joblib.load("clf_rf_best_model.joblib")
      #  gc.collect()

      elif classifier == "Regression Logistique":
        st.subheader("Logistic Regression Result")
        best_params = {"max_iter":1000,
                       "C":0.1,
                       "solver":"sag"}
        feat_imp = model_reduction(classifier, X_train_final, y_train)
        X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
        clf_LogReg_best = LogisticRegression(**best_params, class_weight = "balanced", random_state = 42).fit(X_train_final, y_train)
        joblib.dump(clf_LogReg_best, "clf_LogReg_best_model.joblib")
        model = joblib.load("clf_LogReg_best_model.joblib")
        gc.collect()

      elif classifier == "Arbre de Décision":
        st.subheader("Decision Tree Result")
        best_params = {"criterion":"gini"}
        feat_imp = model_reduction(classifier, X_train_final, y_train)
        X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
        clf_DecTree_best = DecisionTreeClassifier(**best_params, class_weight = "balanced", random_state = 42).fit(X_train_final, y_train)
        joblib.dump(clf_DecTree_best, "clf_DecTree_best_model.joblib")
        model = joblib.load("clf_DecTree_best_model.joblib")

      y_pred = model.predict(X_test_final[feat_imp])
      accuracy = model.score(X_test_final[feat_imp], y_test)
      cm = np.round(confusion_matrix(y_test, y_pred, normalize = "true"), 4)
      recall = np.diag(cm) / np.sum(cm, axis = 1)
      st.write("Test Accuracy", round(accuracy, 4))
      st.write("Test Recall", round(np.mean(recall), 4))
      gc.collect()
      
      # Tracé des graphes (Feature Importances, Matrice de confusion, Precision-Recall)
      col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="center") #
      with col3:
          with st.container():
              #st.subheader("Courbe Precision Recall")
              fig = multiclass_PR_curve(model, X_test_final[feat_imp], y_test)
              fig.update_layout(
              plot_bgcolor='rgba(0, 0, 0, 0)',
              paper_bgcolor='rgba(0, 0, 0, 0)',
              title_text='Courbe Precision-Recall',
              title_font_color='white',
              titlefont=dict(size=15),
              margin=dict(l=55, r=55, t=55, b=55),
              legend=dict(font=dict(family="Arial", size=8, color="white"))
              )
              st.plotly_chart(fig)
              gc.collect()   
            
      with col2:
          with st.container():
              #st.subheader("Matrice de Confusion")
              figML = px.imshow(cm, labels={"x": "Classes Prédites", "y": "Classes réelles"}, color_continuous_scale = "RdYlGn", width=400, height=400, text_auto=True)  
              figML.update_layout(title_text='Confusion Matrix', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width=1000, height=500, legend=dict(
                  x=0.5, y=1.05, orientation="h", xanchor="center", yanchor="bottom", font=dict(family="Arial", size=15, color="white")),
                margin=dict(l=100, r=100, t=100, b=100), titlefont=dict(size=15), title_font_color='white')
              st.plotly_chart(figML)
              gc.collect()

      with col1:
          with st.container():
              #st.subheader("Feature Importance")
              if hasattr(model, 'feature_importances_'):
                  feat_imp = pd.Series(model.feature_importances_, index=X_test_final.columns).sort_values(ascending=True)
              elif hasattr(model, "coef_"):
                  avg_importance_values = np.mean(np.abs(model.coef_), axis = 0)
                  feat_imp = pd.Series(avg_importance_values, index=X_test_final.columns).sort_values(ascending=True)
              feat_imp = feat_imp.loc[feat_imp.values > 0]
              fig = px.bar(feat_imp, x=feat_imp.values, y=feat_imp.index, orientation='h')
              fig.update_layout(title_text='Feature Importance',
                                xaxis_title='Importance',
                                yaxis_title='Features',
                                paper_bgcolor='rgba(0,0,0,0)',  
                                plot_bgcolor='rgba(0,0,0,0)',  
                                width=1000, height=500,
                                legend=dict(x=0.5, y=0.93, orientation="h", xanchor="center", yanchor="bottom",
                                            font=dict(family="Arial", size=15, color="white")),
                                margin=dict(l=100, r=100, t=100, b=100),
                                titlefont=dict(size=15),
                                title_font_color= 'white')
              st.plotly_chart(fig)
              gc.collect()

    # Création d'un bouton utilisateur pour l'interactivité
    if st.sidebar.button("User Execution", key = "classify"):
      if classifier == "XGBoost":
        st.subheader("XGBoost User Results")
        model = XGBClassifier(n_estimators = n_estimators,
                              objective = "multi:softprob",
                              tree_method = tree_method, 
                              max_depth = max_depth,
                              learning_rate = learning_rate,                  
                              sample_weight = classes_weights).fit(X_train_final, y_train)
      elif classifier == "Random Forest":
        st.subheader("Random Forest User Results")
        model = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        class_weight = classes_weights).fit(X_train_final, y_train)
      elif classifier == "Regression Logistique":
         st.subheader("Logistic Regression User Results")
         model = LogisticRegression(random_state = 42, 
                                    max_iter = max_iter, 
                                    solver = solver,
                                    class_weight = classes_weights).fit(X_train_final, y_train)
      elif classifier == "Arbre de Décision":
        st.subheader("Decision Tree User Results")
        model = DecisionTreeClassifier(random_state = 42,
                                        criterion = criterion,
                                        max_features = max_features,
                                        max_depth = max_depth,
                                        min_samples_split = min_samples_split,
                                        min_samples_leaf = min_samples_leaf,
                                        class_weight = classes_weights).fit(X_train_final, y_train)
      #gc.collect()
      # st.write("Le score d'entrainement est :", np.round(model.score(X_train_final, y_train), 4))
      y_pred = model.predict(X_test_final)
      cm = np.round(confusion_matrix(y_test, y_pred, normalize = "true"), 4)
      accuracy = np.round(model.score(X_test_final, y_test), 4)
      recall = np.diag(cm) / np.sum(cm, axis = 1)
      st.write("Test Accuracy", round(accuracy, 4))
      st.write("Test Recall :", round(np.mean(recall), 4))
      
      # Tracé des graphes
      col1, col2 = st.columns(2, gap="small", vertical_alignment="center") #
      with col2:
          with st.container():
              #st.subheader("Courbe Precision Recall")
              fig= multiclass_PR_curve(model, X_test_final, y_test) 
              fig.update_layout({
              'plot_bgcolor': 'rgba(0, 0, 0, 0)',
              'paper_bgcolor': 'rgba(0, 0, 0, 0)'
          })
              st.plotly_chart(fig)
              gc.collect()
            
      with col1:
          with st.container():
              #st.subheader("Matrice de Confusion")
              figML = px.imshow(cm, labels={"x": "Classes Prédites", "y": "Classes réelles"}, color_continuous_scale = "RdYlGn", width=400, height=400, text_auto=True)
              #layout = go.Layout(title='Confusion Matrix', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')  
              figML.update_layout(title='Confusion Matrix', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width=1000, height=500, legend=dict(
                  x=0.5, y=1.05, orientation="h", xanchor="center", yanchor="bottom", font=dict(family="Arial", size=15, color="black")),
                margin=dict(l=100, r=100, t=100, b=100), titlefont=dict(size=20))
              st.plotly_chart(figML)
              gc.collect()

  #############################################################################
  ### Déploiement en production (prédiction basée sur de nouvelles données) ###
  #############################################################################

  # # Affichage des données de validation (interactivité)
  # if st.checkbox("Affichage d'exemples de données à utiliser pour la validation"):
  #   st.dataframe(Fires_test2)

  # Saisie des nouvelles données et prédiction des causes
  st.subheader("Prédiction 🔎", divider="blue") 
  if st.checkbox("PRODUCTION : Nouvelle prédiction"):
    df = pd.DataFrame(columns=["FIRE_YEAR", "LATITUDE", "LONGITUDE", "FIPS_CODE", "AVG_TEMP", "AVG_PCP", "DURATION", "FIRE_SIZE_CLASS", "STATE", "MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"])
    STATE = Fires_ML["STATE"].unique()
    
    config = {
      "FIRE_YEAR": st.column_config.NumberColumn("FIRE_YEAR", min_value = 2015, required = True, help = "Entrer une année supérieure à 2015"),
      "LATITUDE": st.column_config.NumberColumn("LATITUDE", min_value = 17.9, max_value = 70.4, required = True),
      "LONGITUDE": st.column_config.NumberColumn("LONGITUDE", min_value = -179, max_value = 0, required = True),
      "STATE": st.column_config.SelectboxColumn("STATE", options = STATE, required = True),
      "FIPS_CODE": st.column_config.NumberColumn("FIPS_CODE", min_value = 0, max_value = 810, required = True),
      "AVG_TEMP": st.column_config.NumberColumn("AVG_TEMP", min_value = -50, max_value = 50, required = True, help = "Entrer une température en °C"),
      "AVG_PCP": st.column_config.NumberColumn("AVG_PCP", min_value = 0, max_value = 1000, required = True, help = "Entrer un niveau de précipitation en mm"),
      "DURATION": st.column_config.NumberColumn("DURATION", min_value = 0, max_value = 200, required = True, help = "Entrer une durée en jour"),
      "MONTH_DISCOVERY": st.column_config.NumberColumn("MONTH_DISCOVERY", min_value = 1, max_value = 12, required = True, help = "Entrer une valeur comprise entre 1 et 12"),
      "DISCOVERY_WEEK": st.column_config.NumberColumn("DISCOVERY_WEEK", min_value = 1, max_value = 53, required = True, help = "Entrer une valeur comprise entre 1 et 53"),
      "DAY_OF_WEEK_DISCOVERY": st.column_config.NumberColumn("DAY_OF_WEEK_DISCOVERY", min_value = 0, max_value = 6, required = True, help = "Entrer une valeur comprise entre 0 et 6"),
      "FIRE_SIZE_CLASS": st.column_config.SelectboxColumn("FIRE_SIZE_CLASS", options=["A", "B", "C", "D", "E", "F", "G"], required = True),
      # "STATE": st.column_config.SelectboxColumn("STATE", options = STATE, required = True)
      }
    input_data = st.data_editor(df, column_config = config, num_rows='dynamic')
    if st.button("RUN"):
      input_data_ML = real_data_process(input_data).astype("float")
      # # Affichage des inputs labelisées
      # st.dataframe(input_data_ML)
      # Prédiction de la cause du (des) feu(x)
      if classifier == "XGBoost":
          clf = best_xgb_raw_model(X_train_final, y_train)
      #elif classifier == "Random Forest":
          clf = best_rf_raw_model(X_train_final, y_train)
      elif classifier == "Regression Logistique":
          clf = best_LogReg_raw_model(X_train_final, y_train)
      elif classifier == "Arbre de Décision":
          clf = best_DecTree_raw_model(X_train_final, y_train)
      y_pred = clf.predict(input_data_ML)
      y_pred = pd.DataFrame(y_pred, columns = ["FIRE CAUSE"], index = input_data.index)
      y_pred.loc[y_pred["FIRE CAUSE"] == 0, "FIRE CAUSE"] = "Humaine 🧑"
      y_pred.loc[y_pred["FIRE CAUSE"] == 1, "FIRE CAUSE"] = "Criminelle 🦹🏻‍♂️🔥"
      y_pred.loc[y_pred["FIRE CAUSE"] == 2, "FIRE CAUSE"] = "Naturelle 🌩️"
      st.dataframe(y_pred)
      gc.collect()

  #pd.set_option('future.no_silent_downcasting', True)

# Modèles de prédiction des classes
if page == pages[4] :  

 @st.cache_resource
 def load_FiresML2():
  FiresML2= df.loc[:,['MONTH_DISCOVERY','FIRE_SIZE_CLASS','STAT_CAUSE_DESCR','AVG_TEMP [°C]','AVG_PCP [mm]','LONGITUDE','LATITUDE']]   #'AVG_TEMP [°C]','AVG_PCP [mm]'
  FiresML2['FIRE_SIZE_CLASS'] = FiresML2['FIRE_SIZE_CLASS'].replace({"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1, "G": 1}).infer_objects(copy=False)
  #FiresML2['FIRE_SIZE_CLASS'] = FiresML2['FIRE_SIZE_CLASS'].replace({"A":0,"B":0,"C":0,"D":1,"E":1,"F":1,"G":1})
  FiresML2=FiresML2.dropna() 
  return FiresML2
 gc.collect()
 FiresML2=load_FiresML2()

 @st.cache_data(ttl=1200)
 def feats_target():
   feats = FiresML2.drop('FIRE_SIZE_CLASS', axis=1)
   target = FiresML2['FIRE_SIZE_CLASS'].astype('int')
   return feats, target
 feats,target=feats_target()
 gc.collect()

 @st.cache_data(ttl=1200)
 def data_split(X, y):
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42,stratify=target)
  return X_train, X_test, y_train, y_test 
 X_train, X_test, y_train, y_test = data_split(feats, target)
 num_train=X_train.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
 num_test=X_test.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
 sc = StandardScaler()
 num_train= sc.fit_transform(num_train)
 num_test= sc.transform(num_test)
 oneh = OneHotEncoder(drop = 'first', sparse_output=False)
 cat_train=X_train.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
 cat_test=X_test.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
 cat_train=oneh.fit_transform(cat_train)
 cat_test= oneh.transform(cat_test)
 gc.collect()

 @st.cache_data(ttl=1200)
 def label_circ(X_train, X_test):
   circular_cols = ['MONTH_DISCOVERY']
   circular_train = X_train[circular_cols].copy()
   circular_test = X_test[circular_cols].copy()
   circular_train['MONTH_DISCOVERY'] = circular_train['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h / 12))
   circular_train['MONTH_DISCOVERY'] = circular_train['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
   circular_test['MONTH_DISCOVERY'] = circular_test['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h /12))
   circular_test['MONTH_DISCOVERY'] = circular_test['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
   return circular_train,circular_test
 circular_train,circular_test=label_circ(X_train,X_test)
 gc.collect()

 @st.cache_data(ttl=1200)
 def X_train_X_test(X_train, X_test):
   X_train=np.concatenate((num_train,cat_train,circular_train),axis=1)
   X_test=np.concatenate((num_test,cat_test,circular_test), axis=1)
   return X_train,X_test
 X_train,X_test=X_train_X_test(X_train, X_test)
 gc.collect()

 @st.cache_data(ttl=1200)
 def y_train_ytest(y_train,y_test):
   le = LabelEncoder()
   y_train = le.fit_transform(y_train)
   y_test = le.transform(y_test)
   return y_train,y_test
 y_train,y_test=y_train_ytest(y_train,y_test)
 st.subheader("Objectif", divider="blue") 
 st.markdown("L'objectif du modèle est de définir la probabilité qu'un feu se transorme en feu de grande classe. Les classes ont été regroupées de la façon suivante : la classe 0 (petite classe) regroupe les feux de classes A à C (0 à 100 acres), la classe 1 (grande classe) regroupe les feux des classes D à G (100 à plus de 5000 acres).")  
 gc.collect()

 if st.checkbox("Affichage répartition des classes") :
     with st.container():
     #@st.cache_data(persist=True)
     #def Rep_Class():
        fig30= make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("Répartition avant regroupement","Répartition après regroupement"))
        fig30.add_trace(go.Histogram(histfunc="count",
        name="Répartition des classes avant regroupement",
          x=df['FIRE_SIZE_CLASS'], marker_color='Orangered'),1,1)    
        fig30.update_xaxes(categoryorder='array', categoryarray= ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        fig30.add_trace(go.Histogram(histfunc="count",
          name="Répartition Classe",
          x=FiresML2['FIRE_SIZE_CLASS'],marker_color='darkred'),1,2)
        fig30.update_layout(bargap=0.2,height=300, width=1100, coloraxis=dict(colorscale='Bluered_r'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')
        #return fig30
       #fig30=Rep_Class()
       #fig30
     joblib.dump(st.plotly_chart(fig30),"Répartition Classe")
     gc.collect()

 with st.sidebar :  
  with st.form(key='my_form2'):
   st.header("1 - Choix du modèle")
   classifier=st.selectbox("",("XGBoost","BalancedRandomForest"))      
   st.header("2 - Choix des paramètres")
   mois=st.slider('mois',1,12,1)
   Cause=st.selectbox("Cause",('Non défini','Origine humaine', 'Équipements', 'Criminel', 'Foudre','Non défini'),index=1)
   Température=st.slider('Température',-25.00,40.00,1.00)
   Précipitations=st.slider('Précipitation',0.00,917.00,800.00)
   Longitude=st.slider('Longitude',-178.00,-65.00,-119.00)
   Latitude=st.slider('Latitude',17.00,71.00,36.77)
   submit_button = st.form_submit_button(label='Execution')
  data={'MONTH_DISCOVERY':mois,'STAT_CAUSE_DESCR':Cause,'AVG_TEMP [°C]':Température,'AVG_PCP [mm]':Précipitations,"LONGITUDE":Longitude,"LATITUDE":Latitude}
  input_df=pd.DataFrame(data,index=[0])
  input_array=np.array(input_df)
  input_fires=pd.concat([input_df,feats],axis=0)    
  num_input_fires=input_fires.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
  num_input_fires= sc.transform(num_input_fires)    
  cat_input_fires=input_fires.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
  cat_input_fires=oneh.transform(cat_input_fires)
  circular_cols = ['MONTH_DISCOVERY']
  circular_input_fires = input_fires[circular_cols].copy()
  circular_input_fires['MONTH_DISCOVERY'] = circular_input_fires['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h / 12))
  circular_input_fires['MONTH_DISCOVERY'] = circular_input_fires['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
  df_fires_encoded=np.concatenate((num_input_fires,cat_input_fires,circular_input_fires),axis=1)
  
  #LAT = input_df.iloc[0]['LATITUDE']
  #LONG = input_df.iloc[0]['LONGITUDE']
  LAT=input_df[:1].LATITUDE.to_numpy()
  LONG=input_df[:1].LONGITUDE.to_numpy()
  gc.collect()
  #classifier=st.selectbox("Sélection du modèle",("BalancedRandomForest","XGBoost"))
  #with st.form(key='my_form'):
  #submit_button = st.form_submit_button(label='Submit')

 if classifier == "XGBoost":       
  model = joblib.load("model.joblib")
  y_pred=model.predict(X_test)
  prediction=model.predict(df_fires_encoded[:1])
  prediction_proba=model.predict_proba(df_fires_encoded[:1])
  df_prediction_proba=pd.DataFrame(prediction_proba)
  df_prediction_proba.columns=['Petite Classe','Grande Classe']
  df_prediction_proba.rename(columns={0:"Petite Classe",1:"Grande Classe"})
  Fires_class_pred=np.array(['Petite Classe','Grande Classe'])
  gc.collect()

  st.subheader("Importance features et performance du modèle XGBoost optimisé", divider="blue")
  st.write("Accuracy",round(model.score(X_test,y_test),4))
  st.write("Recall",round(recall_score(y_test,y_pred),4))  

  col1, col2,col3 = st.columns(3,gap="small",vertical_alignment="center")
  if st.checkbox('Affichage Courbe ROC'):
   with col3:
    with st.container(height=350):
     @st.cache_data(ttl=1200)
     def AUC():      
      precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      roc_auc = auc(fpr, tpr)
      figML2 = px.area(x=fpr, y=tpr,title=f'Courbe ROC (AUC={auc(fpr, tpr):.4f})',labels=dict(x='Taux faux positifs', y='Taux vrais positifs'))
      figML2.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
      figML2.update_yaxes(scaleanchor="x", scaleratio=1)
      figML2.update_xaxes(constrain='domain')
      figML2.update_layout(title_x = 0.2, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,margin=dict(l=0, r=0, t=20, b=0))
      return figML2
     figML2=AUC()
     figML2
  if st.checkbox("Affichage Matrice de Confusion"):
   with col2:
    with st.container(height=350):
     @st.cache_data(ttl=1200)
     def cm():
      cm = confusion_matrix(y_test, y_pred)
      figML1 = px.imshow(cm,labels={"x": "Classe prédite", "y": "Classe réelle"},width=800,height=800,text_auto=True)
      figML1.update_layout(title='Matrice de confusion',title_x = 0.35, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=1,orientation="h",xanchor="center",yanchor="bottom",font=dict(
      family="Arial",size=15,color="black")),margin=dict(l=0, r=0, t=2, b=0))
      figML1.update_traces(dict(showscale=False,coloraxis=None), selector={'type':'heatmap'})
      return figML1
     figML1=cm()
     figML1 
  if st.checkbox("Affichage Features Importance"):
    with col1 : 
     with st.container(height=350):
      feats1 = {}
      for feature, importance in zip(feats.columns,model.feature_importances_):
       feats1[feature] = importance
      importances= pd.DataFrame.from_dict(feats1, orient='index').rename(columns={0: 'Importance'})
      importances.sort_values(by='Importance', ascending=False).head(8)
      fig = px.bar(importances, x='Importance', y=importances.index)
      fig.update_layout(title='Features Importance',title_x = 0.4, title_y = 0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
      family="Arial",size=15,color="black")),margin=dict(l=0, r=0, t=50, b=0),titlefont=dict(size=15))
      st.plotly_chart(fig) 


  st.subheader("Prédiction de la classe de feux selon les paramètres choisis", divider="blue")
  # Function to generate the map
  @st.cache_data(ttl=1200)
  def generate_map(Fires_class_pred, prediction, df_prediction_proba, LAT, LONG):
      for i in range(len(Fires_class_pred)):
          if Fires_class_pred[prediction][0] == 'Petite Classe':
              color = 'darkblue'
          elif Fires_class_pred[prediction][0] == 'Grande Classe':
              color = 'red'
          else:
             color = 'gray'
      html = df_prediction_proba.to_html(classes="table table-striped table-hover table-condensed table-responsive")
      popup2 = folium.Popup(html)
      m = folium.Map(
           location=[30, -65.844032],
           zoom_start=3,
           tiles='http://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
           attr="Sources: National Geographic, Esri, Garmin, HERE, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, INCREMENT P"
      )
      folium.Marker([LAT, LONG], popup=popup2, icon=folium.Icon(color=color, icon='fire', prefix='fa')).add_to(m)
      return m
      gc.collect()

    # Function to display the legend
  @st.cache_data(ttl=1200)
  def display_legend():
       st.info('Cliquer sur le point localisé sur la carte pour afficher les probabilités de chaque classe', icon="ℹ️")
       st.markdown("")
       st.markdown("Légende :")
       col1, col2 = st.columns([0.15, 0.85], gap="small", vertical_alignment="center")
       with col1:
           st.image("feu_bleu.jpg", width=40)
       with col2:
           st.markdown(":blue[Probabilité classe 1 < 50%]")
       col1, col2 = st.columns([0.15, 0.85], gap="small", vertical_alignment="center")
       with col1:
           st.image("feu_rouge.jpg", width=40)
       with col2:
           st.markdown(":red[Probabilité classe 1 > 50%]")
       gc.collect()

# Main code
#st.subheader("Prédiction de la classe de feux selon les paramètres choisis", divider="blue")

  if st.checkbox("Affichage Prédiction de la Classe de Feux"):
        col1, col2 = st.columns([0.55, 0.45], gap="small", vertical_alignment="center")

        with col1:
            with st.container(height=350):
                m = generate_map(Fires_class_pred, prediction, df_prediction_proba, LAT, LONG)
                st_data = st_folium(m, width=800, returned_objects=[], key="map")
                gc.collect()

        with col2:
            display_legend()
            gc.collect() 
 

# if classifier == "BalancedRandomForest":
#  #dict_weights = {0:1, 1: 1.2933}
#  #model3=BalancedRandomForestClassifier(sampling_strategy="not minority", replacement=True,random_state=200,n_estimators=400,class_weight=dict_weights).fit(X_train,y_train)
#  #joblib.dump(model3, "model3.joblib")
#  model3 = joblib.load("model3.joblib")
#  y_pred=model3.predict(X_test)
#  prediction=model3.predict(df_fires_encoded[:1])
#  prediction_proba=model3.predict_proba(df_fires_encoded[:1])
#  df_prediction_proba=pd.DataFrame(prediction_proba)
#  df_prediction_proba.columns=['Petite Classe','Grande Classe']
#  df_prediction_proba.rename(columns={0:"Petite Classe",1:"Grande Classe"})
#  Fires_class_pred=np.array(['Petite Classe','Grande Classe'])   

#  st.subheader("Importance features et performance du modèle Balanced Random Forest", divider="blue")
#  st.write("Accuracy",round(model3.score(X_test,y_test),4))
#  st.write("Recall",round(recall_score(y_test,y_pred),4))
#  col1, col2, col3 = st.columns(3,gap="small",vertical_alignment="center")
 
#  if st.checkbox('Afficher la courbe ROCAUC'):
#   with col3:
#    with st.container(height=350):
#         @st.cache_data(ttl=1200)
#         def ROCAUC():
#          precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
#          fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#          roc_auc = auc(fpr, tpr)
#          figML2 = px.area(x=fpr, y=tpr,title=f'Courbe ROC (AUC={auc(fpr, tpr):.4f})',labels=dict(x='Taux faux positifs', y='Taux vrais positifs'))
#          figML2.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
#          figML2.update_yaxes(scaleanchor="x", scaleratio=1)
#          figML2.update_xaxes(constrain='domain')
#          figML2.update_layout(title_x = 0.2, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,margin=dict(l=0, r=0, t=20, b=0))
#          return figML2
#         figML2=ROCAUC()
#         figML2
#         gc.collect()

#  if st.checkbox("Afficher la matrice de confusion"):
#    with col2:
#     with st.container(height=350):
#      @st.cache_data(ttl=1200)
#      def MatriceConfusion():
#       cm = confusion_matrix(y_test, y_pred)
#       smallest_number = cm.min()
#      #largest_number = cm.max()
#      #colors = [(0, negative_color),(0.5, zero_color),(1, positive_color)]# Normalize the midpoint value to 0.5
#       figML1 = px.imshow(cm,labels={"x": "Classe prédite", "y": "Classe réelle"},width=800,height=800,text_auto=True)#color_continuous_scale='hot'
#       figML1.update_layout(title='Matrice de confusion',title_x = 0.35, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=1,orientation="h",xanchor="center",yanchor="bottom",font=dict(
#       family="Arial",size=15,color="white")),margin=dict(l=0, r=0, t=2, b=0))
#       figML1.update_traces(dict(showscale=False,coloraxis=None), selector={'type':'heatmap'})
#       return figML1
#      figML1=MatriceConfusion()
#      figML1
#      gc.collect()

#  if st.checkbox("Afficher Features Importance"):
#    with col1 : 
#     with st.container(height=350):
#      @st.cache_data(ttl=1200)
#      def FeatureImportance():    
#         feats2 = {}
#         for feature, importance in zip(feats.columns,model3.feature_importances_):
#             feats2[feature] = importance
#         importances2= pd.DataFrame.from_dict(feats2, orient='index').rename(columns={0: 'Importance'})
#         importances2.sort_values(by='Importance', ascending=False).head(8)    
#         figML3 = px.bar(importances2, x='Importance', y=importances2.index)
#         figML3.update_layout(title='Features Importance',title_x = 0.4, title_y = 0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
#         family="Arial",size=15,color="white")),margin=dict(l=0, r=0, t=50, b=0),titlefont=dict(size=15))
#         return figML3
#      figML3=FeatureImportance()
#      figML3
#      gc.collect()

#  st.subheader("Prédiction de la classe de feux selon les paramètres choisis", divider="blue")
  # Function to generate the map
#  @st.cache_data(ttl=1200)
#  def generate_map(Fires_class_pred, prediction, df_prediction_proba, LAT, LONG):
#      for i in range(len(Fires_class_pred)):
#          if Fires_class_pred[prediction][0] == 'Petite Classe':
#              color = 'darkblue'
#          elif Fires_class_pred[prediction][0] == 'Grande Classe':
#              color = 'red'
#          else:
#              color = 'gray'
#      html = df_prediction_proba.to_html(classes="table table-striped table-hover table-condensed table-responsive")
#      popup2 = folium.Popup(html)
#      m = folium.Map(
#          location=[30, -65.844032],
#          zoom_start=3,
#          tiles='http://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
#          attr="Sources: National Geographic, Esri, Garmin, HERE, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, INCREMENT P"
#      )
#      folium.Marker([LAT, LONG], popup=popup2, icon=folium.Icon(color=color, icon='fire', prefix='fa')).add_to(m)
#      return m
#      gc.collect()

#  # Function to display the legend
#  @st.cache_data(ttl=1200)
#  def display_legend():
#     st.info('Cliquer sur le point localisé sur la carte pour afficher les probabilités de chaque classe', icon="ℹ️")
#      st.markdown("")
#      st.markdown("Légende :")
#      col1, col2 = st.columns([0.15, 0.85], gap="small", vertical_alignment="center")
#      with col1:
#          st.image("feu_bleu.jpg", width=40)
#      with col2:
#          st.markdown(":blue[Probabilité classe 1 < 50%]")
#      col1, col2 = st.columns([0.15, 0.85], gap="small", vertical_alignment="center")
#      with col1:
#          st.image("feu_rouge.jpg", width=40)
#      with col2:
#          st.markdown(":red[Probabilité classe 1 > 50%]")
#      gc.collect()

# Main code
#st.subheader("Prédiction de la classe de feux selon les paramètres choisis", divider="blue")

#  if st.checkbox("Afficher la prédiction de la classe de feux"):
#      col1, col2 = st.columns([0.55, 0.45], gap="small", vertical_alignment="center")

#      with col1:
#          with st.container(height=350):
#              m = generate_map(Fires_class_pred, prediction, df_prediction_proba, LAT, LONG)
#              st_data = st_folium(m, width=800, returned_objects=[], key="map")
#              gc.collect()

#      with col2:
#          display_legend()
#          gc.collect() 
 
# Conclusion
if page == pages[5] : 
  st.write("### Conclusion et propositions d’optimisation")
  st.markdown("""
Le projet “Feux de forêts” nous a permis de mettre en pratique les compétences acquises durant notre formation en data analysis, en abordant toutes les étapes d’un projet de Data Science, de l’exploration des données à la modélisation et la data visualisation. Nous avons également abordé les étapes de prédiction, en utilisant des modèles avancés pour prévoir les occurrences de feux de forêts et ainsi mieux comprendre les facteurs qui les influencent.
### Résultats obtenus :
- **Amélioration des performances des modèles** : Grâce à l’utilisation de différentes méthodes comme class_weight et de classificateurs spécifiques pour les jeux de données déséquilibrés, tels que SMOTE ou EasyEnsemble, nous avons significativement amélioré les performances des modèles.
- **Modèles les plus performants** : 
 **BalancedRandomForest** : Ce modèle trouve que les données météorologiques comme la température et les précipitations sont très importantes pour prédire les feux de forêt. Il utilise aussi beaucoup le mois de l’année et la cause du feu pour faire ses prédictions.
 **XGBoost** : Ce modèle, en revanche, trouve que les informations géographiques comme l’État ou la longitude sont plus importantes. Il utilise un peu moins les données météorologiques et accorde moins d’importance au mois et à la cause du feu comparé au BalancedRandomForest.


### Pistes d’optimisation :
- **Méthodes de resampling** : Utiliser des méthodes plus précises que les méthodes aléatoires, comme SMOTETomek, SMOTEEN, KmeansSMOTE.
- **Données plus précises** : Ajouter des données plus détaillées sur les températures, par exemple des données journalières au lieu de moyennes mensuelles.
- **Réglage des hyperparamètres** : Continuer à optimiser les hyperparamètres, car toutes les combinaisons n’ont pas pu être testées.
### Impact et perspectives :
Ce projet a démontré l'importance de la data analysis et du machine learning dans la prévention et la gestion des incendies de forêt. En permettant une détection humaine ou naturelle des incendies et en ciblant les interventions là où elles sont le plus nécessaires, nous pouvons contribuer à réduire les coûts économiques et les impacts environnementaux des feux de forêt.
En termes d'expertise, ce projet nous a permis de développer nos compétences en Python et en modélisation via le Machine Learning, des domaines nouveaux pour la plupart d'entre nous. Nous avons également appris à utiliser des outils interactifs comme Streamlit pour la restitution de nos résultats.
Pour aller plus loin, il serait bénéfique de collaborer avec des spécialistes en lutte contre les incendies de forêt pour affiner nos modèles et mieux comprendre les enjeux opérationnels. De plus, l'intégration de données météorologiques plus précises pourrait améliorer encore davantage les performances de nos modèles.
En conclusion, ce projet nous a permis de mettre en pratique les compétences acquises durant notre formation et de contribuer à un enjeu crucial de préservation de l'environnement et de sécurité publique.
""")