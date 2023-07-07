import streamlit as st
import pandas as pd
import json
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd


API_KEY = st.secrets["API"]



page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://www.surveylegend.com/wordpress/wp-content/themes/sage/resources/images/img/modal/upgrade-file-upload.png");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
right: 2rem;
background-image: url("");
background-size: cover;
}
[data-testid="stSidebar"]> div:first-child{
background-image: url("https://img.freepik.com/premium-vector/skyblue-gradient-background-advertisers-gradient-hq-wallpaper_189959-513.jpg");
background-size: cover;
}
</style>

"""
st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white';>Dynamic Analytics Crime Modelling 🚔👮</h1>", unsafe_allow_html=True)
st.markdown("---")


st.write("Please upload your Chicago Crime CSV file")

data = st.file_uploader("Upload a CSV")

query = st.text_input("Please do ask here to get insights from the CSV - ")
def agent(filename: str):

    llm = OpenAI(openai_api_key=API_KEY)
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm, df, verbose=False)

def query_llm(agent, query):

    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            The various types of chart, "bar", "line" and "pie" plot

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    response = agent.run(prompt)
    return response.__str__()

def decode_response(response: str) -> dict:

    return json.loads(response)


def write_response(response_dict: dict):

    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

if st.button("Submit Query", type="primary"):

    agent = agent(data)
    response = query_llm(agent=agent, query=query)
    decoded_response = decode_response(response)
    write_response(decoded_response)
