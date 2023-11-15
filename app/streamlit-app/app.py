# https://blog.streamlit.io/finding-your-look-alikes-with-semantic-search/
from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st
from sentence_transformers import SentenceTransformer
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

CATEGORIES_ACTIVE_CS_SUBSET = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.CY",
    "cs.DB",
    "cs.DL",
    "cs.HR",
    "cs.IR",
    "cs.LG",
    "cs.NE",
    "cs.PL",
    "cs.SE",
]

st.set_page_config(
    page_title="arXiv-insanity",
    page_icon="🤓",
    # layout="wide",
    initial_sidebar_state="expanded",
)

VESPA_URL = os.environ.get("VESPA_ENDPOINT", "http://localhost:8080")
VESPA_CERT_PATH = os.environ.get("VESPA_CERT_PATH", None)

app = Vespa(
    url=VESPA_URL,
    cert=VESPA_CERT_PATH,
)


def get_available_categories():
    return CATEGORIES_ACTIVE_CS_SUBSET


@dataclass
class AppStore:
    pass
    # api_key = API_KEY
    # index_name = INDEX_NAME
    # vespa_url = VESPA_URL


class HomePage:
    def __init__(self, app):
        self.app = app

    def render(self) -> None:
        st.write(self.load_css(), unsafe_allow_html=True)
        self.render_side_bar()
        self.render_header()
        (
            query_input_submitted,
            query_profile,
            query_categories,
        ) = self.render_search_form()
        if query_input_submitted:
            # with st.spinner("Wait for it..."):
            with st.spinner(f"Searching for {query_input_submitted}"):
                hits = self.app.vespa_execute_query(
                    query_input_submitted,
                    query_profile,
                    query_categories,
                )

            if hits:
                self.render_search_results(hits)
            else:
                st.text("No relevant documents were found 😥")
            # show number of results and time taken
            # st.write(templates.number_of_results(total_hits, results["took"] / 1000), unsafe_allow_html=True)

    def render_side_bar(self) -> None:
        st.sidebar.markdown("Nice description goes here!")
        # vespa_url = st.sidebar.text_input(
        #     label="Vespa search endpoint URL",
        #     # label_visibility="hidden",
        #     value=VESPA_URL,
        #     placeholder="http://...",
        #     key="vespa_url",
        # )
        # clip_model_name = st.sidebar.selectbox("Select article category", get_available_categories())
        st.sidebar.markdown(
            """Powered by:
- [Google Cloud](https://cloud.google.com)
- [Prefect](https://www.prefect.io)
- [Streamlit](https://www.streamlit.io)
- [Vespa](https://docs.vespa.ai)
- and much more ...""",
        )
        st.sidebar.markdown("This work would not exist ...")

        # st.sidebar.markdown("Made with ❤️ in Zürich by Filippo B")
        st.sidebar.markdown(
            "Made with ❤️‍🔥 in Zürich by [Filippo Broggini](https://www.linkedin.com/in/filippobroggini/)",
        )

    def render_header(self) -> None:
        out1, col1, out2 = st.columns([3, 1, 3])
        # col1.image("https://cord19.vespa.ai/static/media/Vespa_Logo_Mark_Full.49a249a9.svg", width=50)
        # col1 = st.columns([1])[0]
        # col1.markdown(f"# {self.app.title}")
        st.markdown(
            f"<h1 style='text-align: center;'>{self.app.title}</h1>",
            unsafe_allow_html=True,
        )

    def render_search_form(self):
        query_input = st.text_input(
            label="Main search bar",
            label_visibility="hidden",
            value="",
            placeholder="Search ...",
            key="query_input",
        )
        col1, col2 = st.columns([1, 1])
        query_categories = col2.multiselect(
            label="Categories (for bm25 profile only)",
            options=get_available_categories(),
            key="categories_selection",
            max_selections=1,
        )
        query_profile = col1.radio(
            "Query profile",
            options=("bm25", "semantic", "hybrid", "fusion"),
            index=0,
            horizontal=True,
        )
        if not query_input:
            st.markdown(
                """Try searching for:

* imagenet moment
* bert for reranking
* sparse retrieval
* computer vision model to detect dogs and cats""",
            )

        return query_input, query_profile, query_categories

    def render_search_results(self, hits, batch_size: int = 10):
        st.markdown("---")
        for hit in hits[:batch_size]:
            with st.container():
                title = hit["fields"]["title"].replace("<hi>", "*").replace("</hi>", "*")
                st.markdown(
                    f'[**{title}**](https://arxiv.org/abs/{hit["fields"]["id"]})',
                )
                # st.markdown(f'Relevance: {hit["relevance"]:10f}')
                html = self.render_tags(
                    [
                        f'Relevance: {hit["relevance"]:10f}',
                        f'{hit["fields"]["id"]}',
                        f'{" ".join(hit["fields"]["categories"])}',
                        f'{hit["fields"]["update_date"]}',
                    ],
                )
                st.write(html, unsafe_allow_html=True)

                st.markdown(
                    f'{hit["fields"]["abstract"].replace("<hi>", "**").replace("</hi>", "**")}',
                )

                st.markdown("---")

    def render_tags(self, tags: list[str]):
        html = ""
        for tag in tags:
            html += f'<a id="tags">{tag}</a>'

        return html

    def load_css(self) -> str:
        """Return all css styles."""
        common_tag_css = """
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    padding: .15rem .40rem;
                    position: relative;
                    text-decoration: none;
                    font-size: 95%;
                    border-radius: 5px;
                    margin-right: .5rem;
                    margin-top: .4rem;
                    margin-bottom: .5rem;
        """
        return f"""
            <style>
                #tags {{
                    {common_tag_css}
                    color: rgb(88, 88, 88);
                    border-width: 0px;
                    background-color: rgb(240, 242, 246);
                }}
                #tags:hover {{
                    color: black;
                    box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
                }}
                #active-tag {{
                    {common_tag_css}
                    color: rgb(246, 51, 102);
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(246, 51, 102);
                }}
                #active-tag:hover {{
                    color: black;
                    border-color: black;
                    background-color: rgb(240, 242, 246);
                    box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
                }}
            </style>
        """


class App:
    def __init__(self):
        self.title = "arXiv insanity"
        self.store = AppStore()
        self.home_page = HomePage(self)
        # self.embedder = SentenceTransformer("sentence-transformers/allenai-specter")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def render(self):
        HomePage(self).render()

    # @st.cache(ttl=7 * 24 * 60 * 60)
    def embed_query(self, query_text: str):
        if self.query_prefix:
            query_text = f"{self.query_prefix} {query_text}"
        return self.embedder.encode([query_text])[0].tolist()

    def vespa_execute_query(
        self,
        query_text: str,
        query_profile: str,
        query_categories: list[str],
    ) -> list:
        if query_categories:
            filter_categories: str = f'categories contains "{query_categories[0]}" and'
        else:
            filter_categories = ""
        result: VespaQueryResponse = None
        if query_profile == "bm25":
            with app.syncio(connections=1) as session:
                result = session.query(
                    body={
                        "yql": f"select * from sources article where {filter_categories} userQuery()",
                        "hits": 10,
                        "query": query_text,
                        "type": "all",
                        "ranking.profile": "bm25",
                    },
                )
        else:
            query_embedding = self.embed_query(query_text)
            if query_profile == "semantic":
                with app.syncio(connections=1) as session:
                    result = session.query(
                        body={
                            "yql": "select * from sources article where ({targetHits:1000}nearestNeighbor(abstract_embedding, query_embedding))",
                            # "query": query_text,
                            "hits": 10,
                            # "type": "any",
                            "input.query(query_embedding)": query_embedding,
                            "ranking.profile": "semantic",
                        },
                    )
            elif query_profile == "hybrid":
                with app.syncio(connections=1) as session:
                    result = session.query(
                        body={
                            "yql": "select * from sources article where userQuery() or ({targetHits:1000}nearestNeighbor(abstract_embedding, query_embedding))",
                            "hits": 10,
                            "type": "weakAnd",
                            "query": query_text.lower(),
                            "ranking.features.query(query_embedding)": query_embedding,
                            "ranking.profile": "hybrid",
                        },
                    )
            elif query_profile == "fusion":
                with app.syncio(connections=1) as session:
                    result = session.query(
                        body={
                            "yql": "select * from sources article where userQuery() or ({targetHits:1000}nearestNeighbor(abstract_embedding, query_embedding))",
                            "hits": 10,
                            # "type": "weakAnd",
                            "query": query_text.lower(),
                            "input.query(query_embedding)": query_embedding,
                            "ranking.profile": "fusion",
                        },
                    )
        return result.hits


if __name__ == "__main__":
    st_app = App()
    st_app.render()
