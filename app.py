Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Buckets
new
Docs
Pricing

Website
Community
Solutions

Spaces:
Manar-122
/
PRET


like
0

App
Files
Community
Settings
PRET
/
app.py

Manar-122's picture
Manar-122
Update app.py
7ff1c6a
verified
about 6 hours ago
Raw

Download with hf CLI

Copy download link
History
Blame
Edit
Delete
18 kB
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

# تعديل احتياطي للمفتاح السري لـ JWT
JWT_SECRET = os.environ.get("JWT_SECRET_KEY") or "YOUR_BACKEND_SECRET_KEY"
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
            
            # فك التشفير مع تجاوز التحقق من التوقيع مؤقتاً لتسهيل ربط البيئات المختلفة
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_signature": False})
            
            g.user_id = payload.get('user_id') or payload.get('sub')
            g.sub = g.user_id 
            
            if not g.user_id:
                return jsonify({'status': 'error', 'message': 'التوكن غير صالح: لا يحتوي على معرف مستخدم (user_id).'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'status': 'error', 'message': 'انتهت صلاحية جلسة الدخول، يرجى تسجيل الدخول مجدداً.'}), 401
        except Exception as e:
            print(f"[JWT ERROR]: {str(e)}")
            return jsonify({'status': 'error', 'message': f'توكن غير صالح أو حدث خطأ أثناء فحص الهوية: {str(e)}'}), 401

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
# Helpers
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

def check_and_trigger_auction(material_name):
    """
    دالة ذكية تفحص إجمالي الأوزان الحالية للمادة وتقارنها بالحد الذي وضعه الأدمن،
    وإذا تم تخطي الحد ترسل إشعاراً فورياً لبدء المزاد.
    """
    if db is None:
        return
    
    # 1. جلب الهدف النشط (Pending) الذي حدده الأدمن لهذه المادة
    threshold = db['thresholds'].find_one({'material_name': material_name, 'status': 'pending'})
    if not threshold:
        return 
    
    target_weight = threshold['target_weight']

    # 2. حساب إجمالي الأوزان الحالية المخزنة في مجموعة wastes لهذه المادة بالذات
    pipeline = [
        {'$match': {'material_name': material_name}}, 
        {'$group': {'_id': None, 'total': {'$sum': '$total_weight'}}}
    ]
    result = list(db['wastes'].aggregate(pipeline))
    current_total_weight = result[0]['total'] if result else 0

    # 3. التحقق والمقارنة لإطلاق المزاد
    if current_total_weight >= target_weight:
        # تحديث الحالة إلى Triggered لمنع تكرار الإشعار مع كل عملية إدخال تالية
        db['thresholds'].update_one({'_id': threshold['_id']}, {'$set': {'status': 'triggered'}})
        
        # حفظ الإشعار في قاعدة البيانات ليظهر على لوحة تحكم الأدمن
        db['admin_notifications'].insert_one({
            'message': f"تنبيه ذكي: وصلت كمية {material_name} المتوفرة إلى {round(current_total_weight, 2)} كجم (الحد المطلوب: {target_weight} كجم). يمكنك الآن إطلاق المزاد العلني للشركات ومصانع إعادة التدوير!",
            'material_name': material_name,
            'current_weight': round(current_total_weight, 2),
            'timestamp': datetime.utcnow(),
            'read': False,
            'action_link': '/start_auction'
        })
        print(f"[AUCTION TRIGGERED]: {material_name} has reached the target inventory weight!")

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
            
    final_material_name = material_info['name'] if material_info else target_name
    final_price = material_info['price'] if material_info else 0

    # محاكاة لإضافة وزن افتراضي أو حفظ السجل في الـ wastes عند الفحص لتحديث المخزون
    # (يمكنك تعديل الوزن الإجمالي بناءً على المدخلات الحقيقية القادمة من الميزان أو المستخدم)
    if db is not None:
        simulated_weight = 15.5  # كمية افتراضية مسجلة للفحص والتجربة
        db['wastes'].insert_one({
            'user_id': getattr(g, 'user_id', 'anonymous'),
            'material_name': final_material_name,
            'total_weight': simulated_weight,
            'createdAt': datetime.utcnow()
        })
        # استدعاء الفحص الذكي للمزاد فوراً بعد إضافة الوزن الجديد
        check_and_trigger_auction(final_material_name)
            
    return jsonify({
        'class_index': class_index,
        'confidence': result['confidence'],
        'material_name': final_material_name,
        'price': final_price,
        'status': 'success'
    })

@app.route('/admin/set_threshold', methods=['POST'])
def set_threshold():
    """
    مسار خاص بـ لوحة تحكم المسؤول (Admin Dashboard) لتحديد الوزن المستهدف لأي مادة.
    """
    data = request.json or {}
    material_name = data.get('material_name')
    target_weight = data.get('target_weight')

    if not material_name or not target_weight:
        return jsonify({'status': 'error', 'message': 'برجاء تحديد اسم المادة والوزن المستهدف بشكل صحيح.'}), 400

    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    # حفظ أو تحديث الهدف المطلوب داخل مجموعة thresholds
    db['thresholds'].update_one(
        {'material_name': material_name},
        {'$set': {'target_weight': float(target_weight), 'status': 'pending', 'updatedAt': datetime.utcnow()}},
        upsert=True
    )

    return jsonify({
        'status': 'success',
        'message': f'تم تحديد الوزن المستهدف لـ {material_name} بنجاح وهو {target_weight} كجم. سيتم مراقبة المخزون لإشعارك عند بدء المزاد.'
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
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/waste_stats', methods=['GET'])
def waste_stats():
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503
    try:
        pipeline = [
            { '$match': { 'total_weight': { '$exists': True, '$ne': None } } },
            {
                '$lookup': {
                    'from': 'materials',
                    'localField': 'material_id',
                    'foreignField': '_id',
                    'as': 'material_info'
                }
            },
            { '$unwind': '$material_info' },
            {
                '$group': {
                    '_id': '$material_info._id',
                    'material_name': { '$first': '$material_info.name' },
                    'total_weight_kg': { '$sum': '$total_weight' },
                    'avg_price': { '$avg': '$price' },
                    'count': { '$sum': 1 },
                    'all_product_ids': { '$push': '$_id' } 
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'material_id': '$_id',
                    'material_name': 1,
                    'total_weight_kg': { '$round': ['$total_weight_kg', 2] },
                    'avg_price': { '$round': ['$avg_price', 2] },
                    'count': 1,
                    'all_product_ids': 1
                }
            },
            { '$sort': { 'total_weight_kg': -1 } }
        ]
        results = list(db['wastes'].aggregate(pipeline))
        
        for res in results:
            res['material_id'] = str(res['material_id'])
            res['all_product_ids'] = [str(pid) for pid in res['all_product_ids']]
            
        return jsonify({'success': True, 'stats': results})
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/ask_pret', methods=['POST'])
def ask_pret():
    data = request.json or {}
    query = data.get('query', '').strip()
    
    user_id = getattr(g, 'user_id', None) or getattr(g, 'sub', None)
    
    if not user_id:
        return jsonify({'status': 'error', 'message': 'فشل التحقق من الهوية: معرف المستخدم غير موجود.'}), 401
        
    if not query:
        return jsonify({'status': 'error', 'message': 'No query provided'}), 400
        
    try:
        db_history = list(db.chat_history.find({'user_id': user_id}).sort('timestamp', -1).limit(6))
        db_history.reverse()
        
        last_scan = db.waste_scans.find_one({'user_id': user_id}, sort=[('timestamp', -1)])
        
        messages = []
        
        for chat_entry in db_history:
            messages.append(types.Content(role="user", parts=[types.Part.from_text(text=chat_entry['user_message'])]))
            messages.append(types.Content(role="model", parts=[types.Part.from_text(text=chat_entry['bot_reply'])]))
            
        if last_scan:
            scanned_item = last_scan.get('result', '')
            context_prompt = f"(ملاحظة للنظام: المستخدم قام بتصوير {scanned_item} منذ قليل، ضع ذلك في حسبانك إذا كان سياق سؤاله غامضاً أو مرتبطاً به)."
            messages.append(types.Content(role="user", parts=[types.Part.from_text(text=context_prompt)]))
            
        messages.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7
            )
        )
        bot_reply = response.text
        
        db.chat_history.insert_one({
            'user_id': user_id,
            'user_message': query,
            'bot_reply': bot_reply,
            'timestamp': datetime.utcnow()
        })
        
        return jsonify({
            'status': 'success',
            'answer': bot_reply,
            'source': 'Gemini AI with Secure MongoDB Context'
        })
        
    except Exception as e:
        print(f"Error in ask_pret: {str(e)}")
        return jsonify({'status': 'error', 'message': 'عذراً، حدث خطأ في خوادم الذكاء الاصطناعي. حاول مجدداً.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
