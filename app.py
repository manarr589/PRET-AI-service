import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, g
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
import io
import wikipedia
from langdetect import detect
import os
from google import genai
from google.genai import types
import jwt

# تأكدي من تسجيل الـ GEMINI_API_KEY في الـ Secrets على هاجنج فيس
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# إنشاء جلسة شات ثابتة مع إعطائه تعليمات صارمة بالتخصص البيئي
SYSTEM_INSTRUCTION = (
    "أنت مساعد ذكي متخصص فقط في مجال البيئة وإعادة التدوير والاستدامة واسمك PRET. "
    "يجب أن تجيب على الأسئلة البيئية فقط بذكاء وسياق متصل وتتذكر الأسئلة السابقة للمستخدم. "
    "إذا سألك المستخدم في أي موضوع خارج البيئة وإعادة التدوير، اعتذر منه بلطف وأخبره أنك متخصص في البيئة فقط."
)

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

JWT_SECRET = os.environ.get("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"

@app.before_request
def jwt_authentication_middleware():
    protected_routes = ['/ask_pret', '/classify_waste', '/predict_waste']
    
    if request.path in protected_routes:
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'status': 'error', 'message': 'عذراً، يجب تسجيل الدخول أولاً (Missing Token).'}), 401
            
        try:
            token = auth_header.split(" ")[1] if " " in auth_header else auth_header
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # حفظ المعرف في كلا الحقلين لضمان عمل الـ Middleware مع بقية الفانكشنز بسلاسة
            g.user_id = payload.get('sub')
            g.sub = payload.get('sub')
            
            if not g.user_id:
                return jsonify({'status': 'error', 'message': 'التوكن غير صالح: لا يحتوي على معرف مستخدم.'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'status': 'error', 'message': 'انتهت صلاحية جلسة الدخول، يرجى تسجيل الدخول مجدداً.'}), 401
        except (jwt.InvalidTokenError, IndexError):
            return jsonify({'status': 'error', 'message': 'توكن غير صالح أو تم التلاعب به.'}), 401

# ──────────────────────────────────────────────
# TensorFlow Model
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'Models', 'keras_model.h5')
model = None

try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Success: Model loaded from {model_path}")
    else:
        print(f" Error: File not found at {model_path}")
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
        return jsonify({'error': 'AI model not loaded'}), 503
        
    image_file = request.files['image']
    result = predict_waste_type(image_file)
    class_index = result['class_index']
    
    waste_mapping = {
        0: "Plastic", 
        1: "Paper", 
        2: "Cardboard", 
        3: "Oil" , 
        4: "Metal",
        5: "Glass",
        6: "Non-Recyclable Waste",
    }
    
    target_name = waste_mapping.get(class_index, "Unknown")
    material_info = None
    if db is not None:
        mat = db['materials'].find_one({"name": target_name})
        if mat:
            material_info = {
                'id': str(mat['_id']),
                'name': mat.get('name'),
                'price': mat.get('price')
            }
            
    return jsonify({
        'class_index': class_index,
        'confidence': result['confidence'],
        'material_name': material_info['name'] if material_info else target_name,
        'price': material_info['price'] if material_info else 0,
        'status': 'success'
    })

@app.route('/predict_waste', methods=['GET'])
def get_prediction():
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503
        
    wastes_collection = db['wastes']
    records = list(wastes_collection.find({}, {'_id': 1, 'createdAt': 1, 'total_weight': 1}))
    
    if len(records) < 2:
        return jsonify({'error': 'Insufficient data to make a prediction'}), 400
        
    try:
        data = pd.DataFrame(records)
        data['_id'] = data['_id'].apply(str)
        data['Date'] = pd.to_datetime(data['createdAt'], errors='coerce')
        data = data.dropna(subset=['Date', 'total_weight'])
        
        # 👑 الإصلاح الأول: تطبيق الدالة على كائن التاريخ الفردي داخل الـ lambda بشكل صحيح
        data['Date_Ordinal'] = data['Date'].apply(lambda x: x.toordinal())
        
        X = data[['Date_Ordinal']]
        y = data['total_weight']
        
        average_weight = float(y.mean())
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        
        target_date = datetime.now() + timedelta(days=4)
        prediction_date_df = pd.DataFrame([[target_date.toordinal()]], columns=['Date_Ordinal'])
        prediction = model_lr.predict(prediction_date_df)
        
        raw_prediction = float(prediction[0])
        if raw_prediction <= 0:
            predicted_kg = round(average_weight, 2)
            prediction_type = "average (due to negative trend)"
        else:
            predicted_kg = round(raw_prediction, 2)
            prediction_type = "linear regression"
            
        return jsonify({
            'status': 'success',
            'target_date': target_date.strftime('%Y-%m-%d'),
            'predicted_weight_kg': predicted_kg,
            'prediction_method': prediction_type,
            'average_history_weight': round(average_weight, 2),
            'records_count': len(data), 
