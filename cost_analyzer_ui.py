import time
import psycopg2
import streamlit as st
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
from vertexai.language_models import TextEmbeddingModel
import decimal
import json
import os
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Link

load_dotenv()

tracer_provider = TracerProvider()
cloud_trace_exporter = CloudTraceSpanExporter()
tracer_provider.add_span_processor(
    # BatchSpanProcessor buffers spans and sends them in batches in a
    # background thread. The default parameters are sensible, but can be
    # tweaked to optimize your performance
    BatchSpanProcessor(cloud_trace_exporter)
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer(__name__)

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super().default(o)


db_user = os.environ['DB_USER']
db_pass = os.environ['DB_PASSWORD']
db_host = os.environ['DB_HOST']
db_name = 'postgres'
db_port = '5432'  # Default PostgreSQL port

def get_documents(question, tracer_parent):
    with tracer.start_as_current_span("embbed_text", links=[Link(tracer_parent.context)]) as embbed_text:
    
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embeddings = model.get_embeddings([question])
        vector = embeddings[0].values
    #print(f"Length of Embedding Vector: {len(vector)}")
    #return vector
    
    
    with tracer.start_as_current_span("retrieve_relevant_content", links=[Link(tracer_parent.context)]) as retrieve_relevant_content:
        conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)

        # Assuming `embeddings` is a list of embedding vectors you received from the textembedding service
        # And `texts` is the list of texts corresponding to each embedding
        #for text, embedding in zip(texts, embeddings):
            # Convert the embedding to a format suitable for storage, e.g., a string
            #embedding_str = f"[{','.join(map(str, str(embedding)))}]"
        #print(embeddings)
            # Insert into database
        results = None
        with conn.cursor() as cur:
            cur.execute(f"""SELECT DISTINCT content, 1 - (embedding <=> '{vector}') AS similarity FROM embeddings_sample
                            WHERE 1 - (embedding <=> '{vector}') > .65
                            ORDER BY similarity DESC
                            LIMIT 3""")
            results = cur.fetchall()

        conn.close()

        matches = '\n'.join([match[0] for match in results])
                    
    return matches

def get_billing_data_for_project(project_id, tracer_parent):
    
    with tracer.start_as_current_span("get_project_cost", links=[Link(tracer_parent.context)]) as get_project_cost:
    
        conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)

        # Query the database for billing data related to the specified project
        query = """
        SELECT service_name, SUM(cost) as total_cost
        FROM gcp_billing_data
        WHERE project_id = %s
        GROUP BY service_name
        """
        with conn.cursor() as cursor:
            cursor.execute(query, (project_id,))
            results = cursor.fetchall()

        conn.close()

        results = json.dumps(results, cls=DecimalEncoder)
    return results


get_content_table_schema = FunctionDeclaration(
    name="get_content_table_schema",
    description="Get the table schema for content with column names that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "the intent based on the question from the user and each service of the cost data. this will help retrieve relevant information to help answer the user's question. ",
            }
        },
        "required": [
            "intent",
        ],
    },
)

get_cost_table_schema = FunctionDeclaration(
    name="get_cost_table_schema",
    description="Get the table schema for project cost data with column names that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "month": {
                "type": "string",
                "description": "Name of the month to use for cost analysis",
            },
            "year": {
                "type": "string",
                "description": "Year to filter the cost data on",
            },
            "project_id": {
                "type": "string",
                "description": "project if to filter the cost data",
            }
        },
        "required": [
            "project_id",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        get_content_table_schema,
        get_cost_table_schema,
    ],
)

model = GenerativeModel(
    "gemini-1.0-pro",
    generation_config={"temperature": 0},
    tools=[sql_query_tool],
)

st.set_page_config(
    page_title="DoiT Cost Analyzer",
    page_icon="vertex-ai.png",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("DoiT Cost Analyzer")
with col2:
    st.image("vertex-ai.png")

st.subheader("Powered by Function Calling in Gemini")

#st.markdown(
#    "[Source Code](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/function-calling/sql-talk-app/)   •   [Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)   •   [Codelab](https://codelabs.developers.google.com/codelabs/gemini-function-calling)   •   [Sample Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb)"
#)

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - Analyze the cost for project proj-webapp-2?
        - What is the service with highest cost in project for proj-webapp-2? and how to reduce it's cost?
        - Provide cost reduction guidance for proj-webapp-2
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    print(message)
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "\$"))  # noqa: W605
        if "backend_details" in message:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])

if prompt := st.chat_input("Ask me cost for a specific project..."):
    with tracer.start_as_current_span("cost_analyzer") as cost_analyzer:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            chat = model.start_chat()
            user_question = prompt

            prompt += """
                To answer the question above you will:
                1) retrieve the cost details for the project
                2) Identify the intent of the question above
                3) Search for content related to the intent of the user question
                4) Search for content related to each service in the cost details
                5) provide a detail analysis of the cost based on all the information for each service, and the content related to the intent question, include the cost for all services.

                Please give a concise, useful and actionable guidance to manage cost based on the data retrieved. Only use data provided. 
                Always give positive information about DoiT International and how it can help. Remove any mention of AWS or AWS services.
                Only use information that is provided, do not make up any information.
                """

            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            print("before function calling:", response)

            api_requests_and_responses = []
            backend_details = ""

            function_calling_in_process = True

            while function_calling_in_process:
                try:
                    params = {}
                    for key, value in response.function_call.args.items():
                        params[key] = value

                    print(response.function_call.name)
                    print(params)

                    if response.function_call.name == "get_content_table_schema":
                        api_response = get_documents(params["intent"], cost_analyzer)
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )

                    if response.function_call.name == "get_cost_table_schema":
                        api_response = get_billing_data_for_project(params["project_id"], cost_analyzer)
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )


                    print(api_response)
                    
                    with tracer.start_as_current_span("model_analyzer", links=[Link(cost_analyzer.context)]) as model_analyzer:

                        response = chat.send_message(
                            Part.from_function_response(
                                name=response.function_call.name,
                                response={
                                    "content": api_response,
                                },
                            ),
                        )
                    response = response.candidates[0].content.parts[0]

                    #print(response)

                    backend_details += "- Function call:\n"
                    backend_details += (
                        "   - Function name: ```"
                        + str(api_requests_and_responses[-1][0])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - Function parameters: ```"
                        + str(api_requests_and_responses[-1][1])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - API response: ```"
                        + str(api_requests_and_responses[-1][2])
                        + "```"
                    )
                    backend_details += "\n\n"
                    with message_placeholder.container():
                        st.markdown(backend_details)

                except AttributeError:
                    function_calling_in_process = False

            time.sleep(3)

            full_response = response.text
            with message_placeholder.container():
                st.markdown(full_response.replace("$", "\$"))  # noqa: W605
                with st.expander("Function calls, parameters, and responses:"):
                    st.markdown(backend_details)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "backend_details": backend_details,
                }
            )