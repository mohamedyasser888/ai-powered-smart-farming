import gradio as gr
from pathlib import Path
import sys

# Add current directory to path so it can find src
sys.path.append(str(Path(__file__).parent))

from src.inference_production import create_production_predictor

print("Loading Production Model...")
predictor = create_production_predictor()
print("Model Loaded!")

def predict_image(image_path):
    if image_path is None:
        return "Please upload an image."
        
    result = predictor.predict(image_path)
    
    if not result['success']:
        return "Error: " + result.get('error', 'Unknown error')
        
    output = ""
    if not result['is_certain']:
        output += f"⚠️ **تنبيه:** المودل غير متأكد بنسبة كبيرة من هذه الصورة (الثقة: {result['confidence_percentage']}). ربما لأن الصورة ملتقطة من مسافة بعيدة أو الإضاءة مختلفة عن صور التدريب.\n\n"
        
    output += f"🍅 **المرض المتوقع:** {result['disease_name']}\n"
    output += f"📊 **نسبة الثقة:** {result['confidence_percentage']}\n\n"
    
    output += "🔍 **أعلى 3 احتمالات:**\n"
    for pred in result['top3_predictions']:
        output += f"- {pred['disease_name']}: {pred['confidence_percentage']}\n"
        
    return output

# Create Gradio Interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath", label="ارفع صورة لورقة الطماطم (Upload Tomato Leaf)"),
    outputs=gr.Markdown(label="النتيجة (Result)"),
    title="🍅 Tomato Disease Detection (Production Model Test)",
    description="اختبار المودل الجديد (EfficientNetV2-S). جرب رفع صورة لورقة طماطم مريضة أو صورة لأي شيء آخر لتجربة نسبة الثقة.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
