import os
import base64
import time
from groq import Groq

# Groq client initialize karein
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def run_ocr(image_bytes_or_pil) -> dict:
    """
    Calls Groq vision model for handwriting extraction.
    Returns a dict with 'success' and 'text' keys to match pipeline.py expectations.
    """
    start = time.time()
    
    # 1. Image ko bytes se lekar base64 string mein convert karein
    if hasattr(image_bytes_or_pil, "save"):  # Agar PIL Image object hai
        import io
        buffer = io.BytesIO()
        image_bytes_or_pil.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    elif isinstance(image_bytes_or_pil, str):  # Agar file path hai
        with open(image_bytes_or_pil, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    else:  # Agar direct bytes hain
        base64_image = base64.b64encode(image_bytes_or_pil).decode('utf-8')

    try:
        # 2. Latest vision model use karte hue API call karein
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract ALL handwritten or printed text from this image exactly as written. Output the raw text exactly as it appears, preserving line breaks. Do not add extra conversational commentary."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.1,
        )
        
        elapsed = round(time.time() - start, 2)
        extracted_text = completion.choices[0].message.content.strip() if completion.choices[0].message.content else ""
        
        # Consistent dict response structured for pipeline.py
        return {"success": True, "text": extracted_text, "error": None, "time_sec": elapsed}
        
    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        return {"success": False, "text": "", "error": str(e), "time_sec": 0}
