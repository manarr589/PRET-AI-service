import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
import wikipedia
from langdetect import detect # مكتبة كشف اللغة

app = Flask(__name__)

# 1. تحميل الموديل (مع معالجة أخطاء الطبقات)
try:
    model = tf.keras.models.load_model('models/keras_model.h5', compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f" Critical error loading model: {e}")

# دالة التصنيف (Inference logic)
def predict_waste_type(image_file):
    image = Image.open(image_file).convert('RGB').resize((224, 224))
    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, 0)
    
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    waste_types = ["Plastic", "Paper", "Oil"] 
    return waste_types[class_index]

# --- الروابط (Endpoints) ---

@app.route('/classify_waste', methods=['POST'])
def classify_waste():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    result = predict_waste_type(image_file)
    return jsonify({'waste_type': result, 'status': 'success'})

@app.route('/predict_waste', methods=['GET'])
def get_prediction():
    try:
        data = pd.read_csv('waste_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data['Date_Ordinal'] = data['Date'].map(datetime.toordinal)
        
        reg = LinearRegression()
        reg.fit(data[['Date_Ordinal']], data['Amount_KG'])
        
        next_month = (datetime.now() + relativedelta(months=1)).replace(day=1)
        pred = reg.predict(np.array([[next_month.toordinal()]]))
        
        return jsonify({
            'target_date': next_month.strftime('%Y-%m-%d'),
            'predicted_amount': round(pred[0], 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# الميزة الجديدة: موسوعة بريت الذكية (تلقائية اللغة)
@app.route('/ask_pret', methods=['POST'])
def ask_pret():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # 1. كشف لغة السؤال (عربي أم إنجليزي)
        language = detect(query) 
        
        # 2. تحديد لغة البحث في ويكيبيديا ورسالة الخطأ
        if language == 'ar':
            wikipedia.set_lang("ar")
            not_found_msg = "عذراً، لم أجد معلومات كافية عن هذا الموضوع في موسوعتي."
        else:
            wikipedia.set_lang("en")
            not_found_msg = "Sorry, I couldn't find enough information on this topic."

        # 3. جلب ملخص من جملتين
        summary = wikipedia.summary(query, sentences=2)
        
        return jsonify({
            'detected_language': language,
            'answer': summary,
            'source': 'PRET Encyclopedia (Wikipedia)'
        })
    except Exception:
        return jsonify({'answer': not_found_msg})

if __name__ == '__main__':
    # تشغيل السيرفر
    app.run(debug=True, port=5000)