import sys,os
sys.path.append(os.getcwd())

import streamlit as st
from connector.base import Connector

INPUT_DATA_TYPES = ["text", "csv", "pdf"]

# -------- Helper functions -------- #
def is_empty(text):
    return text.strip() == ""

class StreamRunner():
    def __init__(self):
        st.set_page_config(page_title="LogSecure+")
        st.title("LogSecure+ - Analyze your logs for compliance, security and standards!")
        st.subheader("Enter details below")

        # init nones
        self.rule_definition = None
        self.input_data_type = None
        self.text_input_data = None
        self.csv_input_data = None
        self.pdf_input_data = None
        self.is_data_definition = False
        self.data_definition = None
        self.is_sample_examples = False
        self.sample_examples_data = None

    def build_text_input(self):
        self.text_input_data = st.text_area("Enter the text", height=200)

    def build_csv_input(self):
        self.csv_input_data = st.file_uploader("Upload CSV")

    def build_pdf_input(self):
        self.pdf_input_data = st.file_uploader("Upload PDF")

    def build_data_definition(self):
        self.data_definition = st.text_area("Explain data features, or what keys refer to what", height=200)

    def build_sample_examples(self):
        self.sample_examples_data = st.text_area("Explain how will compliance process the data", height=200)

    def build(self):
        self.rule_definition = st.text_area("What are the compliance(s)?", height=200)
        self.input_data_type = st.radio("What is the format of data?", INPUT_DATA_TYPES)
        if (self.input_data_type == "text"):
            # for text, show another simple text area
            self.build_text_input()
        elif (self.input_data_type == "csv"):
            # show upload button
            self.build_csv_input()
        elif (self.input_data_type == "pdf"):
            # show upload button
            self.build_pdf_input()

        self.is_data_definition = st.checkbox("Data definition? (how data is processed)", value=False)
        if (self.is_data_definition):
            self.build_data_definition()

        self.is_sample_examples = st.checkbox("Sample examples? (for better accuracy)", value=False)
        if (self.is_sample_examples):
            self.build_sample_examples()

        if st.button("Submit", use_container_width=True):
            self.submit()

    def submit(self):
        if is_empty(self.rule_definition):
            st.error("Please enter the compliance(s)")
            return

        data = None
        if self.input_data_type == "text" and is_empty(self.text_input_data):
            st.error("Please enter some data")
            return
        else:
            data = self.text_input_data

        # Similiarly do for pdf and csv

        # Get connector for inference engine
        connector = get_llm_connector()
        # Get output
        output = connector.evaluate({
            "rules": self.rule_definition,
            "data": data,
            "format": self.input_data_type,
            "understand_data": self.data_definition,
            "sample_examples": self.sample_examples_data
        })
        # Show output
        st.write(output)

@st.cache_resource
def get_llm_connector():
    return Connector("openai")

runner = StreamRunner()
runner.build()

# Also init connector!
get_llm_connector()