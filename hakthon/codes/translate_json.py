import json
import asyncio
from groq import AsyncGroq
import os
from dotenv import load_dotenv

load_dotenv()
async_groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

async def translate_json():
    with open("disease_update (1).json", "r") as f:
        data = json.load(f)
        
    prompt = """
Translate the following JSON into Arabic. Keep the JSON structure, keys, and formatting exactly the same, ONLY translate the string values. 
Do not add any markdown formatting like ```json or anything else, just return the raw JSON text.
Make sure the Arabic is professional and accurate for agriculture.

JSON to translate:
""" + json.dumps(data, indent=2)

    try:
        print("Translating JSON... this might take a minute.")
        response = await async_groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        translated_data = json.loads(content)
        
        with open("disease_arabic.json", "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
            
        print("Translation complete! Saved to disease_arabic.json")
    except Exception as e:
        print(f"Error during translation: {e}")

if __name__ == "__main__":
    asyncio.run(translate_json())
