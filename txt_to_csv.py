from io import StringIO

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd


load_dotenv()

with open("recording.txt", "r", encoding="utf-8") as file:
    text = file.read()

with open("food_calories.csv", "r", encoding="utf-8") as file:
    csv = file.read()



txt_to_csv_template = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
              You are a data clerk. Read a text file and populate only the relevant column in a CSV file using the text information.

              No preambles, no scripts, no comments, no explanations.

              <|eot_id|><|start_header_id|>user<|end_header_id|>
              Read the following text and extract the relevant information to populate the csv file:
              TEXT:
                {TEXT}
              
              CSV:
                {CSV}
              
              Return only the CSV
              <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

llm = ChatGroq(temperature=0,model_name = "llama3-70b-8192")
cv_pdf_to_json_prompt = PromptTemplate(
                template= txt_to_csv_template,
                input_variables=["TEXT", "CSV"]
            )



txt_to_csv_agent = cv_pdf_to_json_prompt | llm | StrOutputParser()
txt_to_csv = txt_to_csv_agent.invoke({
                "TEXT":text,
                "CSV":csv
                })

data = StringIO(txt_to_csv)

df = pd.read_csv(data)

# Save to a CSV file
df.to_csv("consumed_food.csv", index=False)