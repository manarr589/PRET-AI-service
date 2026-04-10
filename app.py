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

app = Flask(__name__)

# ──────────────────────────────────────────────
# MongoDB Connection
# ──────────────────────────────────────────────
MONGO_URI    = "mongodb://mongo:rRvQcNzThKlzXcRIJxcaSclhhuqYRXFl@junction.proxy.rlwy.net:35860"
MONGO_DB_NAME = "test" # <- change to your actual DB name if different

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
model = None
try:
    model = tf.keras.models.load_model('models/keras_model.h5', compile=False)
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")

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
    class_index = np.argmax(prediction)
    waste_types = ["Plastic", "Paper", "Oil"]
    return waste_types[class_index]

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Quick check — server alive + DB reachable."""
    db_status = "disconnected"
    if mongo_client:
        try:
            mongo_client.admin.command("ping")
            db_status = "connected"
        except Exception:
            db_status = "unreachable"
    return jsonify({'server': 'ok', 'database': db_status})

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
    
    # رجّع dictionary مش string
    return {
        "class_index": class_index,
        "confidence": round(confidence * 100, 2)
    }


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

    material = None
    if db is not None:
        materials = list(db['materials'].find({}).sort('createdAt', 1))

        if class_index < len(materials):
            mat = materials[class_index]
            material = {
                'id'   : str(mat['_id']),
                'name' : mat.get('name'),
                'price': mat.get('price')
            }

    return jsonify({
        'class_index'  : class_index,
        'confidence'   : confidence,
        'material_name': material['name'] if material else None,
        'material_id'  : material['id']   if material else None,
        'price'        : material['price'] if material else None,
        'status'       : 'success'
    })
    
@app.route('/predict_waste', methods=['GET'])
def get_prediction():
    """
    Predicts next month's total waste weight using linear regression.

    Schema used  (Waste collection):
        - createdAt   : datetime added automatically by { timestamps: true }
        - total_weight: number  (kg)
        - status      : 'pending' | 'in_auction' | 'sold'

    Optional query param:
        ?status=sold   -> filter by status (default: all records)
    """
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    status_filter = request.args.get('status')   # e.g. ?status=sold

    query = {}
    if status_filter:
        query['status'] = status_filter

    # ── Fetch from 'wastes' collection ──
    wastes = db['wastes']
    records = list(wastes.find(query, {'_id': 0, 'createdAt': 1, 'total_weight': 1}))

    if not records:
        return jsonify({'error': 'No waste records found in database'}), 404

    # ── Build DataFrame ──
    data = pd.DataFrame(records)
    data['createdAt'] = pd.to_datetime(data['createdAt'], errors='coerce')
    data = data.dropna(subset=['createdAt', 'total_weight'])

    if data.empty:
        return jsonify({'error': 'No valid records after parsing dates'}), 422

    data['date_ordinal'] = data['createdAt'].map(datetime.toordinal)

    # ── Linear Regression ──
    reg = LinearRegression()
    reg.fit(data[['date_ordinal']], data['total_weight'])

    next_month = (datetime.now() + relativedelta(months=1)).replace(day=1)
    pred = reg.predict(np.array([[next_month.toordinal()]]))

    return jsonify({
        'target_date'       : next_month.strftime('%Y-%m-%d'),
        'predicted_weight_kg': round(float(pred[0]), 2),
        'records_used'      : len(data),
        'status_filter'     : status_filter or 'all'
    })


@app.route('/waste_stats', methods=['GET'])
def waste_stats():
    """
    Aggregates waste totals grouped by material name.
    Joins 'wastes' -> 'materials' using material_id.

    Returns list of:
        { material_name, total_weight_kg, avg_price, count, statuses }
    """
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    pipeline = [
        # Join material info
        {
            '$lookup': {
                'from'        : 'materials',
                'localField'  : 'material_id',
                'foreignField': '_id',
                'as'          : 'material'
            }
        },
        { '$unwind': { 'path': '$material', 'preserveNullAndEmpty': True } },

        # Group by material
        {
            '$group': {
                '_id'             : '$material._id',
                'material_name'   : { '$first': '$material.name' },
                'total_weight_kg' : { '$sum': '$total_weight' },
                'avg_price'       : { '$avg': '$price' },
                'count'           : { '$sum': 1 },
                'statuses'        : { '$addToSet': '$status' }
            }
        },

        # Clean up output
        {
            '$project': {
                '_id'            : 0,
                'material_name'  : { '$ifNull': ['$material_name', 'Unknown'] },
                'total_weight_kg': { '$round': ['$total_weight_kg', 2] },
                'avg_price'      : { '$round': ['$avg_price', 2] },
                'count'          : 1,
                'statuses'       : 1
            }
        },
        { '$sort': { 'total_weight_kg': -1 } }
    ]

    results = list(db['wastes'].aggregate(pipeline))
    return jsonify({'stats': results, 'groups': len(results)})


@app.route('/waste_by_status', methods=['GET'])
def waste_by_status():
    """
    Returns total weight and count grouped by status:
    pending | in_auction | sold
    """
    if db is None:
        return jsonify({'error': 'Database not connected'}), 503

    pipeline = [
        {
            '$group': {
                '_id'            : '$status',
                'total_weight_kg': { '$sum': '$total_weight' },
                'count'          : { '$sum': 1 },
                'avg_price'      : { '$avg': '$price' }
            }
        },
        {
            '$project': {
                '_id'            : 0,
                'status'         : '$_id',
                'total_weight_kg': { '$round': ['$total_weight_kg', 2] },
                'avg_price'      : { '$round': ['$avg_price', 2] },
                'count'          : 1
            }
        }
    ]

    results = list(db['wastes'].aggregate(pipeline))
    return jsonify({'by_status': results})



# الميزة الجديدة: موسوعة بريت الذكية (تلقائية اللغة)
@app.route('/ask_pret', methods=['POST'])
def ask_pret():
    data = request.json
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    query = data.get('query', '').lower() # نحول الكلام لـ small لسهولة البحث
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # 1. قائمة الكلمات المفتاحية المتعلقة بالتدوير والبيئة (بالعربي والإنجليزي)
    environmental_keywords = [
        'recycling', 'waste', 'plastic', 'paper', 'oil', 'environment', 
        'pollution', 'climate', 'green', 'nature', 'sustainability',
        'تدوير', 'نفايات', 'بلاستيك', 'ورق', 'زيت', 'بيئة', 
        'تلوث', 'مناخ', 'استدامة', 'مخلفات', 'قمامة'
    ]

    # 2. التحقق: هل السؤال له علاقة بالموضوع؟
    is_related = any(word in query for word in environmental_keywords)

    if not is_related:
        # رد مخصص لو السؤال بره الموضوع
        language = detect(query)
        if language == 'ar':
            return jsonify({'answer': "عذراً، أنا متخصص فقط في الأمور المتعلقة بإعادة التدوير والبيئة. كيف يمكنني مساعدتك في هذا المجال؟"})
        else:
            return jsonify({'answer': "Sorry, I am only specialized in recycling and environmental topics. How can I help you in this field?"})

    # 3. لو السؤال له علاقة، نبدأ عملية البحث في ويكيبيديا
    try:
        language = detect(query)
        
        # 2. Set Wikipedia language and error message accordingly
        if language == 'ar':
            wikipedia.set_lang("ar")
            not_found_msg = "عذراً، لم أجد معلومات كافية عن هذا الموضوع البيئي."
        else:
            wikipedia.set_lang("en")
            not_found_msg = "Sorry, I couldn't find enough information on this environmental topic."

        summary = wikipedia.summary(query, sentences=2)
        
        return jsonify({
            'detected_language': language,
            'answer': summary,
            'source': 'PRET Encyclopedia (Wikipedia)'
        })
    except Exception as e:
        return jsonify({'answer': not_found_msg})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)