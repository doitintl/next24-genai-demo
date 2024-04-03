import time
import psycopg2
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool
from vertexai.language_models import TextEmbeddingModel
import decimal
import json
import markdown
import logging
import google.cloud.logging
import os
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Link
from dotenv import load_dotenv

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
client.setup_logging()

load_dotenv()

# Initialize OpenTelemetry tracing for Google Cloud Trace
tracer_provider = TracerProvider()
cloud_trace_exporter = CloudTraceSpanExporter()
tracer_provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)


class DecimalEncoder(json.JSONEncoder):
    """
    Encoder subclass used to convert decimal.Decimal objects to strings.
    This is necessary because json.dumps does not handle Decimal objects by default.
    """
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super().default(o)

# Database credentials should be stored securely, such as in environment variables or secret management services.
db_user = os.environ['DB_USER']
db_pass = os.environ['DB_PASSWORD']
db_host = os.environ['DB_HOST']
db_name = 'postgres'
db_port = '5432'  # Default PostgreSQL port

def get_documents(question, tracer_parent):
    """
    Retrieves documents relevant to the provided question using text embeddings.
    """
    with tracer.start_as_current_span("embbed_text", links=[Link(tracer_parent.context)]) as embbed_text:
        embbed_text.set_attribute("prompt_length", len(question))
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embeddings = model.get_embeddings([question])
        vector = embeddings[0].values
        embbed_text.set_attribute("vector", len(vector))
    
    with tracer.start_as_current_span("retrieve_relevant_content", links=[Link(tracer_parent.context)]) as retrieve_relevant_content:
        conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)
        results = None
        with conn.cursor() as cur:
            cur.execute(f"""SELECT DISTINCT content, 1 - (embedding <=> '{vector}') AS similarity FROM embeddings_sample
                            WHERE 1 - (embedding <=> '{vector}') > .65
                            ORDER BY similarity DESC
                            LIMIT 10""")
            results = cur.fetchall()

        conn.close()

        matches = '\n'.join([match[0] for match in results])
                    
    return matches

def get_billing_data_for_project(project_id, tracer_parent):
    """
    Retrieves billing data for a specific project ID.
    """
    with tracer.start_as_current_span("get_project_cost", links=[Link(tracer_parent.context)]) as get_project_cost: 
        conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port)
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

# Definitions of functions and tools for the vertexai model
# These declarations help in structuring requests to the model and defining expected inputs and outputs
get_content_info = FunctionDeclaration(
    name="get_content_info",
    description="Get content relevant to the intent and services that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "the intent based on the question from the user and includes the services. this will help retrieve relevant information to help answer the user's question. ",
            }
        },
        "required": [
            "intent",
        ],
    },
)

get_cost_table_data = FunctionDeclaration(
    name="get_cost_table_data",
    description="Get the table cost data for a project id that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "project id to filter the cost data",
            }
        },
        "required": [
            "project_id",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        get_content_info,
        get_cost_table_data,
    ],
)

model = GenerativeModel(
    "gemini-1.0-pro",
    generation_config={"temperature": 0},
    tools=[sql_query_tool],
)

# This function is part of a larger system that analyzes costs and provides recommendations.
# Detailed logic and interactions with the model are encapsulated here.
def analyze_cost(prompt, parent_analyzer):
    # Initialize a list to hold state messages for debugging or logging
    state_messages = []

    # Start a new tracing span for cost analysis to enable detailed monitoring and tracing within GCP
    with tracer.start_as_current_span("cost_analyzer", links=[Link(parent_analyzer.context)]) as cost_analyzer:
        # Log the initial user prompt to state messages for reference
        state_messages.append({"role": "user", "content": prompt})

        
        full_response = "" # Prepare the full response variable to hold the final output
        chat = model.start_chat() # Start a chat session with the model for interactive communication
        user_task = prompt # The user task is the initial prompt

        # Enhance the prompt with additional instructions for the model to follow,
        # guiding it to perform a detailed cost analysis based on the given question
        prompt = f"""<context>
                DoiT International is a preferred GCP partner, offering cost optimization services, architecture guidance, and technical support.
                </context>
                <instructions>
                To solve the task you will:
                    1) retrieve the cost details for the project
                    2) Identify the intent of the question above
                    3) Search for content related to the intent of the user question include the name of the services in use
                    4) provide service cost breakdown and a detail analysis of the cost based on all the information for each service, and the content related to the intent question, include the cost for all services.

                Please give a concise, useful and actionable guidance to manage cost based on the data retrieved. Only use data provided. 
                Always give positive information about DoiT International and how it can help. Remove any mention of AWS or AWS services.
                Only use information that is provided, do not make up any information.
                </instructions>
                <task>
                {user_task}
                </task>
                """

        # Log the enhanced prompt's length and content for tracing
        cost_analyzer.set_attribute("prompt_length", len(prompt))
        cost_analyzer.set_attribute("prompt", prompt)

        # Send the enhanced prompt to the model and store the first response
        response = chat.send_message(prompt)
        response = response.candidates[0].content.parts[0]                                              
        cost_analyzer.set_attribute("response_length", len(str(response)))

        # Initialize a list to hold API requests and responses for debugging or processing
        api_requests_and_responses = []
        # String to accumulate details of backend operations for debugging or logging
        backend_details = ""

        # Flag to keep track of whether the model is still processing function calls
        function_calling_in_process = True

        # Loop to handle function calls made by the model within the chat conversation
        while function_calling_in_process:
            try:
                # Extract parameters from the model's function call response
                params = {}
                for key, value in response.function_call.args.items():
                    params[key] = value

                # Handle the "get_content_info" function call by fetching relevant documents
                if response.function_call.name == "get_content_info":
                    api_response = get_documents(params["intent"], cost_analyzer)
                    api_requests_and_responses.append(
                        [response.function_call.name, params, api_response]
                    )

                # Handle the "get_cost_table_data" function call by fetching billing data
                if response.function_call.name == "get_cost_table_data":
                    api_response = get_billing_data_for_project(params["project_id"], cost_analyzer)
                    api_requests_and_responses.append(
                        [response.function_call.name, params, api_response]
                    )

                # Log the API response and prepare the next message for the model based on the API call
                with tracer.start_as_current_span("model_analyzer", links=[Link(cost_analyzer.context)]) as model_analyzer:
                    #msg = Part.from_function_response(
                    #        name=response.function_call.name,
                    #        response={
                    #            "content": api_response,
                    #        },
                    #    )
                    
                    msg = f"the response from function {response.function_call.name} is {api_response}"
                    model_analyzer.set_attribute(f"prompt_length_{response.function_call.name}", len(str(msg)))
                    model_analyzer.set_attribute("content", msg)
                    # Send the constructed message to the model and get the next part of the response
                    response = chat.send_message(msg)
                    response = response.candidates[0].content.parts[0]

                    model_analyzer.set_attribute(f"response_length_{response.function_call.name}", len(str(response)))

                # Compile backend details for logging or debugging
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

            except AttributeError:
                # End the loop if no more function calls are to be processed
                function_calling_in_process = False

        # Process the final response text for output, ensuring correct formatting
        full_response = response.text.replace("$", "\$")
        
        # Add the final response and backend details to state messages for logging or debugging
        state_messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "backend_details": backend_details,
            }
        )
        
        # Return the fully processed response
        return full_response

    
from google.cloud import language_v2

def content_analyze_sentiment(text_content, parent_analyzer):
    """
    Analyzes the sentiment of the provided text content.
    """

    client = language_v2.LanguageServiceClient()
    document_type_in_plain_text = language_v2.Document.Type.PLAIN_TEXT
    language_code = "en"
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text,
        "language_code": language_code,
    }
    encoding_type = language_v2.EncodingType.UTF8

    with tracer.start_as_current_span("sentiment_analyzer", links=[Link(parent_analyzer.context)]) as sentiment_analyzer:
        response = client.analyze_sentiment(
            request={"document": document, "encoding_type": encoding_type}
        )
    
    return response

def positive_response(original_content, parent_analyzer):
    """
    Improves the sentiment of the provided text content with a more positive tone.
    """

    with tracer.start_as_current_span("improve_tone", links=[Link(parent_analyzer.context)]) as improve_tone:
        gemini_pro_model = GenerativeModel("gemini-1.0-pro")

        prompt = f"Improve the note to be more positive on the following text, ALWAYS include the service breakdown and the cost per service.: {original_content}"

        response = gemini_pro_model.generate_content([prompt])

        text = response.candidates[0].content.parts[0].text
    
    return text

def create_email(content, parent_analyzer):
    """
    Creates email to be sent to customer with a specific tone and content.
    """
    
    with tracer.start_as_current_span("create_email", links=[Link(parent_analyzer.context)]) as create_email:
        gemini_pro_model = GenerativeModel("gemini-1.0-pro", generation_config={"temperature": 0.5})

        prompt = f"""
        <task>
        Create an email for a Money Vigalance business leader as if it was coming from DoiT International based on the document below, provide some concise observations and recommendations that could be helpful and the detaile service breakdown with cost. Don't including the term money vigilance or the definition of money script. Follow the template and ensure DoiT international is mentioned where appropriate.
        
        <context>
        
        Characteristic of Money Vigilant money script person:
        
        For a business leader, the essence of Money Vigilance can be tailored to align with the specific responsibilities and challenges of overseeing technology strategy and financial health within an organization. This adaptation would emphasize prudent financial management, strategic investment in technology, and fostering a culture of financial awareness and accountability among team members.

        The Money Vigilant leader is keenly aware of the fiscal health of their technology department and the broader organization. They understand the critical importance of maintaining a balance between innovative tech investments and financial prudence. This leader believes in strategic spending that aligns with long-term organizational goals and demonstrates a strong commitment to budget adherence without stifling innovation.

        For the Money Vigilant leader, ensuring financial health is paramount. They exhibit a strong inclination towards investing in technology that offers clear ROI, advocating for spending that drives efficiency, enhances competitiveness, and generates revenue. They resist the lure of trendy but unproven technologies, emphasizing due diligence and the value of savings for future strategic moves.

        This individual is not one to await serendipitous financial boosts or speculative ventures. They encourage their team to be resourceful, advocating for projects that meet real-world needs through hard work and innovation. The concept of financial handouts is replaced with merit-based achievements, fostering a culture where team members are motivated to contribute to the organization's financial and technological advancements.

        High Money Vigilance scores in a leader correlate with robust financial strategies and technological foresight. They are adept at making informed choices, meeting departmental needs efficiently, and minimizing unnecessary expenditures. Their approach to credit and spending is conservative yet pragmatic, ensuring the company's resources are allocated wisely, favoring investments that promise tangible benefits over speculative gambles.

        Although focused on financial and technological stewardship, the Money Vigilant leader also understands the importance of addressing their own and their team's anxieties about the future. They lead by example, showing how prudent planning and fiscal responsibility can coexist with rewarding innovation and risk-taking within the bounds of financial health.

        To balance the inherent caution with the need for growth and innovation, the Money Vigilant leader encourages setting aside resources for "innovation experiments" or "skunkworks projects" that allow the team to explore new technologies or methodologies with potential strategic value. This balances the need for financial vigilance with the imperative to innovate and stay competitive.

        The Money Vigilant leader also values confidential discussions about financial and technological strategies with trusted advisors or mentors. These conversations are crucial for gaining fresh perspectives, validating strategies, and ensuring that vigilance does not morph into stagnation.

        Finally, this leader recognizes the importance of periodically reassessing personal and departmental approaches to financial planning and technological investment. They understand that being overly focused on cost-saving can inhibit growth and that strategic investment is key to long-term success. Thus, they strive to cultivate a balanced perspective that values financial health without compromising the innovative spirit essential for technological leadership.
        
        DoiT International approach: 
        
        DoiT Intenrational provides expert advise in the monitoring, cost optimization, and architecture of workloads in Google Cloud. We provide a monthly summary to customers to provide insight into their usage, cost and tips to improve their experience with Google Cloud based on their money script. We are always looking for ways to engage with our customers.
       
        <document>
        {content}
        
        <template>

        Salutation

        Summary

        Current cost breakdown:
            1.
            2.
            3. 

        Recommendation Moving Forward:
            1.
            2.
            3.

        Behavioral Tips:

        Closing Remarks
        
        """

        response = gemini_pro_model.generate_content([prompt])

        email = response.candidates[0].content.parts[0].text
    
    return email
    

from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST"])
def cost_analysis():
    with tracer.start_as_current_span("cost_project_analyzer") as cost_project_analyzer:
        query = json.loads(request.data)['query']
        response = analyze_cost(query, cost_project_analyzer)

        sentiment = content_analyze_sentiment(response, cost_project_analyzer)
        
        sentiment_score = sentiment.document_sentiment.score
        sentiment_magnitud = sentiment.document_sentiment.magnitude
        sentiment_overall = sentiment_score * sentiment_magnitud
        
        logging.info("============Original Response==============")
        logging.info(response)
        logging.info("============Original Sentiment==============")
        logging.info(f"sentiment score: {sentiment_score}")
        logging.info(f"sentiment magnitud: {sentiment_magnitud}")
            
        response = positive_response(response, cost_project_analyzer)
        
        sentiment = content_analyze_sentiment(response, cost_project_analyzer)
        
        sentiment_score = sentiment.document_sentiment.score
        sentiment_magnitud = sentiment.document_sentiment.magnitude
        sentiment_overall = sentiment_score * sentiment_magnitud


        content = {"html": markdown.markdown(response),
                   "raw": response,
                   "sentiment": sentiment.document_sentiment.score,
                   "sentiment_maginitud": sentiment.document_sentiment.magnitude
                  }

        logging.info("==================Improved Tone=================")
        logging.info(response)
        logging.info("===========Improved Sentiment===============")
        logging.info(f"sentiment score: {sentiment_score}")
        logging.info(f"sentiment magnitud: {sentiment_magnitud}")
        
        email = create_email(response, cost_project_analyzer)
        
        logging.info("==================Email=================")
        logging.info(email)
        
    
    return email