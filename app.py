import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
import io
import wikipedia
from langdetect import detect
import os
import tensorflow as tf

app = Flask(__name__)

# ──────────────────────────────────────────────
# MongoDB Connection
# ──────────────────────────────────────────────
MONGO_URI = "mongodb+srv://pret_user:oupk4zU5yVk6g4Lf@cluster0.iutwdeh.mongodb.net/?appName=Cluster0"
MONGO_DB_NAME = "test" 

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command("ping")
    db = mongo_client[MONGO_DB_NAME]
    print("[OK] MongoDB connected successfully!")
except Exception as e:
    mongo_client = None
    db = None
    print(f"[WARN] MongoDB connection failed: {e}")

# ──────────────────────────────────────────────
# TensorFlow Model
# ──────────────────────────────────────────────

# 1. تحديد المسار الأساسي
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. بناء المسار (تأكد أن الحروف تطابق GitHub تماماً)
# إذا كان المجلد Models والملف keras_model.h5
model_path = os.path.join(BASE_DIR, 'Models', 'keras_model.h5')

model = None
try:
    if os.path.exists(model_path):
        # تحميل الموديل بدون تفعيل الـ Compile لتوفير الذاكرة وتجنب الأخطاء
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Success: Model loaded from {model_path}")
    else:
        print(f" Error: File not found at {model_path}")
        # هذا السطر سيطبع لك محتويات المجلد في الـ Logs لتكتشف الاسم الصحيح
        if os.path.exists(os.path.join(BASE_DIR, 'Models')):
            print(f"Folder 'Models' contains: {os.listdir(os.path.join(BASE_DIR, 'Models'))}")
except Exception as e:
    print(f" Warning: Model load failed: {str(e)}")

# ──────────────────────────────────────────────
# Helper: classify waste from image
# ──────────────────────────────────────────────
def predict_waste_type(image_file):
    if model is None:
        raise RuntimeError("Model is not loaded")
    
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, 0)
    
    prediction = model.predict(image_array)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    return {
        "class_index": class_index,
        "confidence": round(confidence * 100, 2)
    }

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    db_status = "disconnected"
    if mongo_client:
        try:
            mongo_client.admin.command("ping")
            db_status = "connected"
        except Exception:
            db_status = "unreachable"
    return jsonify({'server': 'ok', 'database': db_status})

@app.route('/classify_waste', methods=['POST'])
def classify_waste():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if model is None:
        return jsonify({'error': 'AI model not loaded on server'}), 503

    image_file = request.files['image']
    result = predict_waste_type(image_file)
    class_index = result['class_index']
    confidence  = result['confidence']

    # قائمة المواد حسب ترتيب تدريب الموديل الخاص بك
    waste_mapping = {0: "Plastic", 1: "Paper", 2: "Oil"}
    target_material_name = waste_mapping.get(class_index, "Unknown")

    material_info = None
    if db is not None:
        # البحث عن المادة في قاعدة البيانات بناءً على الاسم المكتشف
        mat = db['materials'].find_one({"name": target_material_name})
        if mat:
            material_info = {
                'id'   : str(mat['_id']),
                'name' : mat.get('name'),
                'price': mat.get('price')
            }

    return jsonify({
        'class_index'  : class_index,
        'confidence'   : confidence,
        'material_name': material_info['name'] if material_info else target_material_name,
        'material_id'  : material_info['id'] if material_info else None,
        'price'        : material_info['price'] if material_info else 0,
        'status'       : 'success'
    })
    
@app.route('/predict_waste', methods=['GET'])
def get_prediction():
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    wastes_collection = db['wastes']
    records = list(wastes_collection.find({}, {'_id': 0, 'createdAt': 1, 'total_weight': 1}))

    if len(records) < 2:
        return jsonify({'error': 'Insufficient data to make a prediction'}), 400

    try:
        data = pd.DataFrame(records)
        data['Date'] = pd.to_datetime(data['createdAt'], errors='coerce')
        data = data.dropna(subset=['Date', 'total_weight'])
        data['Date_Ordinal'] = data['Date'].map(datetime.toordinal)
        
        X = data[['Date_Ordinal']]
        y = data['total_weight']
        model_lr = LinearRegression() # سميته model_lr لتمييزه عن موديل الصور
        model_lr.fit(X, y)

        # ────────── التعديل الجديد هنا ──────────
        from datetime import timedelta
        target_date = datetime.now() + timedelta(days=4)
        
        # التنبؤ باستخدام DataFrame لتجنب تحذير الـ Logs
        prediction_date_df = pd.DataFrame([[target_date.toordinal()]], columns=['Date_Ordinal'])
        prediction = model_lr.predict(prediction_date_df)
        # ──────────────────────────────────────

        predicted_kg = max(0, float(prediction[0]))

        return jsonify({
            'status': 'success',
            'target_date': target_date.strftime('%Y-%m-%d'),
            'predicted_weight_kg': round(predicted_kg, 2),
            'records_count': len(data)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/waste_stats', methods=['GET'])
def waste_stats():
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    try:
        pipeline = [
            # 1. تصفية المستندات التي تحتوي على وزن
            { '$match': { 'total_weight': { '$exists': True, '$ne': None } } },

            # 2. الربط مع مجموعة المواد
            {
                '$lookup': {
                    'from': 'materials',
                    'localField': 'material_id',
                    'foreignField': '_id',
                    'as': 'material_info'
                }
            },

            # 3. التفكيك (تغيير جذري هنا لحل مشكلة الخطأ)
            # تم حذف preserveNullAndEmptyArrays تماماً واستبداله بتنسيق بسيط
            # إذا استمر الخطأ في هذا السطر، فالمشكلة في الـ Deployment نفسه
            { '$unwind': '$material_info' },

            # 4. التجميع
            {
                '$group': {
                    '_id': '$material_info._id',
                    'material_name': { '$first': '$material_info.name' },
                    'total_weight_kg': { '$sum': '$total_weight' },
                    'avg_price': { '$avg': '$price' },
                    'count': { '$sum': 1 },
                    'statuses': { '$addToSet': '$status' }
                }
            },

            # 5. التنسيق النهائي
            {
                '$project': {
                    '_id': 0,
                    'material_name': { '$ifNull': ['$material_name', 'Unknown'] },
                    'total_weight_kg': { '$round': [{ '$ifNull': ['$total_weight_kg', 0] }, 2] },
                    'avg_price': { '$round': [{ '$ifNull': ['$avg_price', 0] }, 2] },
                    'count': 1,
                    'statuses': 1
                }
            },
            { '$sort': { 'total_weight_kg': -1 } }
        ]

        results = list(db['wastes'].aggregate(pipeline))
        return jsonify({'success': True, 'stats': results, 'groups': len(results)})

    except Exception as e:
        # طباعة الخطأ بوضوح في Railway Logs
        print("--- DATABASE ERROR LOG ---")
        print(str(e))
        return jsonify({'error': str(e), 'status': 'failed'}), 500
    
@app.route('/ask_pret', methods=['POST'])
def ask_pret():
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # تحديد اللغة والرسائل الافتراضية
    try:
        lang = detect(query)
    except:
        lang = 'en'

    if lang == 'ar':
        not_found_msg = "عذراً، لم أجد معلومات كافية عن هذا الموضوع البيئي."
        off_topic_msg = "عذراً، أنا متخصص فقط في إعادة التدوير والبيئة."
        wikipedia.set_lang("ar")
    else:
        not_found_msg = "Sorry, I couldn't find enough information on this topic."
        off_topic_msg = "Sorry, I am only specialized in recycling and environmental topics."
        wikipedia.set_lang("en")

    environmental_keywords = [
        'recycling', 'waste', 'plastic', 'paper', 'oil', 'environment', 
        'pollution', 'climate', 'green', 'nature', 'sustainability',
        'تدوير', 'نفايات', 'بلاستيك', 'ورق', 'زيت', 'بيئة', 
        'تلوث', 'مناخ', 'استدامة', 'مخلفات'
    ]

    is_related = any(word in query.lower() for word in environmental_keywords)

    if not is_related:
        return jsonify({'answer': off_topic_msg})

    try:
        # استخدام sentences=2 للحصول على ملخص قصير ومفيد
        summary = wikipedia.summary(query, sentences=2)
        return jsonify({
            'detected_language': lang,
            'answer': summary,
            'source': 'Wikipedia'
        })
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        return jsonify({'answer': not_found_msg})
    except Exception:
        return jsonify({'answer': "حدث خطأ أثناء البحث، حاول مرة أخرى." if lang == 'ar' else "An error occurred during search."})

if __name__ == '__main__':
    # تأكد من أن البورت 8080 متاح أو غيره حسب إعدادات السيرفر
    app.run(host='0.0.0.0', port=8080, debug=False)
    
    
    