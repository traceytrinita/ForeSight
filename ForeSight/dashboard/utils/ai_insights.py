import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if api_key else None

def generate_ai_insight(summary: dict) -> str:
    if client is None:
        return(
            "AI insight is unavailable. No API key provided."
            "Add your own key in a local .env file to enable this feature."
        )
    try: 
        response = client.responses.create(
            model = "gpt-5-mini",
            input = f"""
            You are a business analytics assistant for a retail decision-support dashboard. 

            Analyse the following business data and explain it clearly in professional business terms:

            {summary}

            Your response must:
            - identify the key trend or important finding
            - explain the business meaning
            - give a short actionable recommendation

            Keep the response short, clear, and professional.
            """
        )
        return response.output_text
    
    except Exception as e:
        return f"AI insight could not be generated: {str(e)}"
