import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from seq2rel import Seq2Rel
from seq2rel.common import util

TEXT_EXAMPLES = {
    "ade": (
        "Hydroxyurea is a cytostatic agent used to treat myeloproliferative disorders and"
        " long-term treatment is associated with mucocutaneous adverse events and nail hyperpigmentation"
    ),
    "bc5cdr": (
        "Neuroleptic malignant syndrome induced by ziprasidone on the second day of"
        " treatment. Neuroleptic malignant syndrome (NMS) is the rarest and most serious of the"
        " neuroleptic-induced movement disorders. We describe a case of neuroleptic malignant"
        " syndrome (NMS) associated with the use of ziprasidone. Although conventional neuroleptics"
        " are more frequently associated with NMS, atypical antipsychotic drugs like ziprasidone"
        " may also be a cause. The patient is a 24-year-old male with a history of schizophrenia"
        " who developed signs and symptoms of NMS after 2 days of treatment with an 80-mg/day dose"
        " of orally administrated ziprasidone. This case is the earliest (second day of treatment)"
        " NMS due to ziprasidone reported in the literature."
    ),
    "biogrid": (
        "DNA-dependent protein kinase (DNA-PK) is composed of a 460-kDa catalytic component (p460)"
        " and a DNA-binding component Ku protein. Immunoblot analysis after treatment of Jurkat"
        " cells with anti-Fas antibody demonstrated the cleavage of p460 concomitantly with an"
        " increase in CPP32/Yama/apopain activity. Recombinant CPP32/Yama/apopain specifically"
        " cleaved p460 in the DNA-PK preparation that had been purified from Raji cells into 230-"
        " and 160-kDa polypeptides, the latter of which was detected in anti-Fas-treated Jurkat"
        " cells. The regulatory component Ku protein was not significantly affected by"
        " CPP32/Yama/apopain. DNA-PK activity was decreased with the disappearance of p460 in the"
        " incubation of DNA-PK with CPP32/Yama/apopain. These results suggest that the catalytic"
        " component of DNA-PK is one of the target proteins for CPP32/Yama/apopain in Fas-mediated"
        "apoptosis."
    ),
    "docred": (
        "Lark Force was an Australian Army formation established in March 1941 during World War II"
        " for service in New Britain and New Ireland. Under the command of Lieutenant Colonel John"
        " Scanlan, it was raised in Australia and deployed to Rabaul and Kavieng, aboard SS"
        " Katoomba, MV Neptuna and HMAT Zealandia, to defend their strategically important harbours"
        "and airfields."
    ),
}


@st.cache(allow_output_mutation=True)
def load_model(model_name: str):
    return Seq2Rel(model_name)


def process_ent(text: str, ents: str) -> str:
    matched_ents = []
    for ent in ents.split(f" {util.COREF_SEP_SYMBOL} "):
        try:
            start = text.lower().index(ent.lower())
            end = start + len(ent)
            matched_ents.append(text[start:end])
        except ValueError:
            matched_ents.append(ent)

    ent_text = f"{matched_ents[0]}"
    if matched_ents[1:]:
        ent_text += f"{util.COREF_SEP_SYMBOL} {util.COREF_SEP_SYMBOL.join(matched_ents[1:])}"
    return ent_text


st.sidebar.write(
    f"""
    # Seq2Rel

    A demo for our [Seq2Rel](https://github.com/JohnGiorgi/seq2rel) models.

    Seq2Rel is a sequence-to-sequence based architecture for joint entity and relation extraction.

    Enter some text on the right, and the extracted entity mentions and relations will be visualized
    below. Coreferent mentions will be seperated by `{util.COREF_SEP_SYMBOL}`
    """
)


model_name = (
    st.sidebar.selectbox("Model name", ("ADE", "BC5CDR", "BioGRID", "DocRED")).strip().lower()
)

st.sidebar.subheader("Additional Settings")
debug = st.sidebar.checkbox("Debug", False)

if model_name:
    model = load_model(model_name)

input_text = st.text_area(
    "Enter some text",
    value=TEXT_EXAMPLES[model_name],
    help="Enter some text here. This will be auto-filled with a model-specific example.",
)
if input_text:
    net = Network(notebook=True)
    output = model(input_text)
    st.subheader("Input text")
    st.write(input_text)

    if debug:
        st.subheader("Raw output")
        st.write(output)

    st.subheader("Extracted relations")
    deserialize_annotations = util.deserialize_annotations(output)
    for prediction in deserialize_annotations:
        for rel_type, rels in prediction.items():
            for rel in rels:
                ent_1_text = process_ent(input_text, rel[0][0])
                ent_2_text = process_ent(input_text, rel[1][0])
                net.add_node(ent_1_text)
                net.add_node(ent_2_text)
                net.add_edge(ent_1_text, ent_2_text, title=rel_type)
    net.show("network.html")
    HtmlFile = open("network.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=1200, width=1000)
