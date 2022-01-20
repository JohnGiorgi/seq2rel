from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from seq2rel import Seq2Rel
from seq2rel.common import util

TEXT_EXAMPLES = {
    "bc5cdr": (
        "Bortezomib and dexamethasone as salvage therapy in patients with relapsed/refractory"
        " multiple myeloma: analysis of long-term clinical outcomes. Bortezomib"
        " (bort)-dexamethasone (dex) is an effective therapy for relapsed/refractory (R/R) multiple"
        " myeloma (MM). This retrospective study investigated the combination of bort (1.3 mg/m(2)"
        " on days 1, 4, 8, and 11 every 3 weeks) and dex (20 mg on the day of and the day after"
        " bort) as salvage treatment in 85 patients with R/R MM after prior autologous stem cell"
        " transplantation or conventional chemotherapy. The median number of prior lines of therapy"
        " was 2. Eighty-seven percent of the patients had received immunomodulatory drugs included"
        " in some line of therapy before bort-dex. The median number of bort-dex cycles was 6, up"
        " to a maximum of 12 cycles. On an intention-to-treat basis, 55 % of the patients achieved"
        " at least partial response, including 19 % CR and 35 % achieved at least very good partial"
        " response. Median durations of response, time to next therapy and treatment-free interval"
        " were 8, 11.2, and 5.1 months, respectively. The most relevant adverse event was"
        " peripheral neuropathy, which occurred in 78 % of the patients (grade II, 38 %; grade III,"
        " 21 %) and led to treatment discontinuation in 6 %. With a median follow up of 22 months,"
        " median time to progression, progression-free survival (PFS) and overall survival (OS)"
        " were 8.9, 8.7, and 22 months, respectively. Prolonged PFS and OS were observed in"
        " patients achieving CR and receiving bort-dex a single line of prior therapy. Bort-dex was"
        " an effective salvage treatment for MM patients, particularly for those in first relapse."
    ),
    "gda": (
        "A polymorphism in the cystatin C gene is a novel risk factor for late-onset Alzheimer's"
        " disease. OBJECTIVE: To investigate whether or not a coding polymorphism in the cystatin"
        " C gene (CST3) contributes risk for AD. DESIGN: A case-control genetic association study"
        " of a Caucasian dataset of 309 clinic- and community-based cases and 134 community-based"
        " controls. RESULTS: The authors find a signficant interaction between the GG genotype of"
        " CST3 and age/age of onset on risk for AD, such that in the over-80 age group the GG"
        " genotype contributes two-fold increased risk for the disease. The authors also see a"
        " trend toward interaction between APOE epsilon4-carrying genotype and age/age of onset"
        " in this dataset, but in the case of APOE the risk decreases with age. Analysis of only"
        " the community-based cases versus controls reveals a significant three-way interaction"
        " between APOE, CST3 and age/age of onset. CONCLUSION: The reduced or absent risk for AD"
        " conferred by APOE in older populations has been well reported in the literature,"
        " prompting the suggestion that additional genetic risk factors confer risk for"
        " later-onset AD. In the author's dataset the opposite effects of APOE and CST3 genotype on"
        " risk for AD with increasing age suggest that CST3 is one of the risk factors for"
        " later-onset AD. Although the functional significance of this coding polymorphism has not"
        " yet been reported, several hypotheses can be proposed as to how variation in an"
        " amyloidogenic cysteine protease inhibitor may have pathologic consequences for AD."
    ),
    "docred": (
        "Darksiders is a hack and slash action-adventure video game developed by Vigil Games and"
        " published by THQ. The game takes its inspiration from the Four Horsemen of the"
        " Apocalypse, with the player taking the role of the horseman War. The game was released"
        " for the PlayStation 3 and Xbox 360 on January 5, 2010 in North America, January 7 in"
        " Australia, January 8 in Europe, and March 18 in Japan."
    ),
}


@st.cache(allow_output_mutation=True, max_entries=1, ttl=3600)
def load_model(model_name: str):
    return Seq2Rel(model_name)


def process_ent(text: str, ents: Tuple[str, ...]) -> str:
    matched_ents = []
    for ent in ents:
        try:
            start = text.lower().index(ent.lower())
            end = start + len(ent)
            matched_ents.append(text[start:end])
        except ValueError:
            matched_ents.append(ent)

    ent_text = f"{matched_ents[0]}"
    if matched_ents[1:]:
        ent_text += f"{util.COREF_SEP_SYMBOL} {f'{util.COREF_SEP_SYMBOL} '.join(matched_ents[1:])}"
    return ent_text


st.sidebar.write(
    f"""
    # Seq2Rel

    A demo for our [Seq2Rel](https://github.com/JohnGiorgi/seq2rel) method.

    Seq2Rel is a generative, sequence-to-sequence based method for end-to-end document-level
    entity and relation extraction.

    1. Select a pretrained model below (it may take a few seconds to load).
    2. Enter some text on the right, and the extracted entity mentions and relations will be visualized
    below.

    Coreferent mentions will be seperated by a semicolon (`{util.COREF_SEP_SYMBOL}`). Hover over nodes and
    edges to see their predicted classes.
    """
)


model_name = (
    st.sidebar.selectbox(
        "Model name",
        ("BC5CDR", "GDA", "DocRED"),
        help="Name of pretrained model to load. Most models are named after the dataset they are trained on.",
    )
    .strip()
    .lower()
)


if model_name:
    model = load_model(model_name)

input_text = st.text_area(
    "Enter some text",
    value=TEXT_EXAMPLES[model_name],
    help="Enter some text here. This will be auto-filled with a model-specific example.",
)
if input_text:
    # Run the model and parse the output
    output = model(input_text)
    deserialize_annotations = util.deserialize_annotations(output)

    st.subheader(":memo: Input text")
    st.write(input_text)

    st.subheader(":gear: Model outputs")
    with st.expander("Click to expand"):
        st.markdown("#### Raw output")
        st.write("The raw output of the model as a string.")
        # We index by 0 because there will only ever be one output.
        st.markdown(f"`{output[0]}`")

        st.markdown("#### Parsed output")
        st.write("The parsed output of the model as a dictionary, keyed by relation class.")
        st.write(deserialize_annotations[0])

    st.subheader(":left_right_arrow: Extracted relations")
    st.write("The models outputs, visualized as a graph.")
    net = Network(width="100%", bgcolor="#F0F2F6", layout=True, notebook=True)
    for prediction in deserialize_annotations:
        for rel_type, rels in prediction.items():
            # TODO: This should be extended to n-ary relations.
            for rel in rels:
                ent_1, ent_1_type = process_ent(input_text, rel[0][0]), rel[0][1]
                ent_2, ent_2_type = process_ent(input_text, rel[1][0]), rel[1][1]
                net.add_node(ent_1, title=ent_1_type, color="#FF4B4B", borderWidth=0)
                net.add_node(ent_2, title=ent_2_type, color="#FF4B4B", borderWidth=0)
                net.add_edge(ent_1, ent_2, title=rel_type)
    net.show("network.html")
    HtmlFile = open("network.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=800)
