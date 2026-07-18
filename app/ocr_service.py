import os
import base64
from groq import Groq

# Groq client initialize karein (ensure karein ke GROQ_API_KEY aapke environment mein set ho)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_text_from_image(image_path_or_bytes):
    """
    Is function ko aap apne existing backend router (jaise /api/process) mein 
    purani groq calling se replace kar dein.
    """
    # 1. Agar image path hai ya bytes, usey base64 string mein convert karein
    if isinstance(image_path_or_bytes, str):
        with open(image_path_or_bytes, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Agar bytes array hai (Uploaded file direct memory mein hai)
        base64_image = base64.b64encode(image_path_or_bytes).decode('utf-8')

    try:
        # 2. Latest vision model 'qwen/qwen3.6-27b' ka use karte hue API call karein
        completion = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please perform OCR on this image. Extract and read all the handwritten or printed text accurately. Do not add extra conversational commentary, just return the exact extracted text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,  # Low temperature taake exact text read ho aur hallucination na ho
        )
        
        # 3. Response return karein
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        raise e
