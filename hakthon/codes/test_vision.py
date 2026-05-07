import asyncio
import base64
import os
from groq import AsyncGroq

async def test_vision():
    GROQ_API_KEY = "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"
    client = AsyncGroq(api_key=GROQ_API_KEY)
    
    file_path = "/home/smart-farming/uploads/1543f1af-f911-43f4-bf9d-0f0a7436fd0a.jpeg"
    
    try:
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        print("File read successfully, calling Groq...")
        
        vision_res = await client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this a picture of a tomato plant, tomato leaf, or tomato fruit? Reply ONLY with 'yes' or 'no'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        print("Response:", vision_res.choices[0].message.content)
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_vision())
