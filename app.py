import joblib as jb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance | AfriAI Solutions", page_icon="🎓")

@st.cache_resource
def charger():
    return jb.load("encoders.joblib"), jb.load("best_model.joblib")

encoders, model = charger()
encoder = encoders[0] if isinstance(encoders, list) else encoders   # un seul encodeur ici

# Colonnes dans l'ordre exact utilisé pour entraîner le modèle
COLONNES = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
            "Sleep Hours", "Sample Question Papers Practiced"]

def predire(d):
    d = d.copy()
    d["Extracurricular Activities"] = encoder.transform(d["Extracurricular Activities"])
    return model.predict(d[COLONNES].values)

st.title("🎓 Performance d'un étudiant")
st.caption("AfriAI Solutions")
st.write("Ce modèle prédit l'indice de performance d'un étudiant (score de 10 à 100) "
         "à partir de ses heures d'étude, de ses notes précédentes, de ses activités "
         "extrascolaires, de son sommeil et du nombre d'épreuves d'entraînement traitées.")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

with onglet1:
    with st.form("simple"):
        heures = st.number_input("Heures d'étude par jour", min_value=1, max_value=9, value=5)
        notes = st.number_input("Notes précédentes", min_value=40, max_value=100, value=70)
        activites = st.selectbox("Activités extrascolaires", list(encoder.classes_))
        sommeil = st.number_input("Heures de sommeil", min_value=4, max_value=9, value=7)
        epreuves = st.number_input("Épreuves d'entraînement traitées", min_value=0, max_value=9, value=4)
        ok = st.form_submit_button("Prédire", type="primary")
    if ok:
        ligne = pd.DataFrame([{
            "Hours Studied": heures, "Previous Scores": notes,
            "Extracurricular Activities": activites,
            "Sleep Hours": sommeil, "Sample Question Papers Practiced": epreuves}])
        score = predire(ligne)[0]
        st.success(f"Indice de performance prédit : **{score:.1f} / 100**")

with onglet2:
    st.caption("Le CSV doit contenir les colonnes : " + ", ".join(COLONNES))
    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if fichier is not None:
        try:
            df = pd.read_csv(fichier)
            manquantes = [c for c in COLONNES if c not in df.columns]
            if manquantes:
                st.error(f"Colonnes manquantes : {manquantes}")
            else:
                df["Performance Index prédit"] = predire(df).round(1)
                st.dataframe(df)
                st.download_button("⬇️ Télécharger les prédictions",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur : {e}")
