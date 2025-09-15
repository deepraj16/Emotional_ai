from flask import Flask, render_template, request, jsonify, send_file
import io
import base64
import os
import time
import logging
from functools import lru_cache
import threading
from llm_with_emotions import (  
    detect_emotions,
    create_pie_chart,
    create_bar_chart,
    create_donut_chart,
    create_horizontal_bar_chart,
    create_polar_chart,
    create_stacked_chart
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance
import matplotlib.pyplot as plt

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

emotion_cache = {}
cache_lock = threading.Lock()
CACHE_SIZE = 100
CACHE_TTL = 3600  # 1 hour

def get_cache_key(text):
    """Generate cache key for text"""
    return hash(text.strip().lower())

def get_from_cache(text):
    """Get emotion result from cache"""
    with cache_lock:
        key = get_cache_key(text)
        if key in emotion_cache:
            result, timestamp = emotion_cache[key]
            if time.time() - timestamp < CACHE_TTL:
                logger.info("Cache hit for emotion detection")
                return result
            else:
                del emotion_cache[key]
    return None

def save_to_cache(text, result):
    """Save emotion result to cache"""
    with cache_lock:
        key = get_cache_key(text)
        if len(emotion_cache) >= CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(emotion_cache.keys(), key=lambda k: emotion_cache[k][1])
            del emotion_cache[oldest_key]
        emotion_cache[key] = (result, time.time())

def fig_to_base64(fig):
    """Convert Matplotlib figure to base64 string for HTML rendering."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)  # Reduced DPI for faster processing
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_base64

def validate_text_input(text):
    """Validate and clean text input"""
    if not text or not text.strip():
        return None, "Text input cannot be empty"
    
    text = text.strip()
    if len(text) > 5000:  # Limit text length for performance
        return None, "Text too long. Maximum 5000 characters allowed."
    
    return text, None

@app.route("/", methods=["GET", "POST"])
def index():
    """Main web interface"""
    chart_urls = {}
    emotions = {}
    error_message = None
    processing_time = 0

    if request.method == "POST":
        start_time = time.time()
        text = request.form.get("text", "")
        
        # Validate input
        cleaned_text, validation_error = validate_text_input(text)
        if validation_error:
            error_message = validation_error
        else:
            try:
                # Try to get from cache first
                cached_result = get_from_cache(cleaned_text)
                if cached_result:
                    emotions = cached_result
                else:
                    # Detect emotions using LLM
                    emotions = detect_emotions(cleaned_text)
                    save_to_cache(cleaned_text, emotions)

                # Generate charts only if emotions were detected
                if "message" not in emotions and emotions:
                    chart_urls = {
                        "Pie Chart": fig_to_base64(create_pie_chart(emotions)),
                        "Bar Chart": fig_to_base64(create_bar_chart(emotions)),
                        "Donut Chart": fig_to_base64(create_donut_chart(emotions)),
                        "Horizontal Bar Chart": fig_to_base64(create_horizontal_bar_chart(emotions)),
                        "Polar Chart": fig_to_base64(create_polar_chart(emotions)),
                        "Stacked Chart": fig_to_base64(create_stacked_chart(emotions)),
                    }
                
                processing_time = round(time.time() - start_time, 2)
                logger.info(f"Processing completed in {processing_time}s")
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                error_message = f"Error processing your request: {str(e)}"

    return render_template("index.html", 
                         emotions=emotions, 
                         chart_urls=chart_urls, 
                         error_message=error_message,
                         processing_time=processing_time)

@app.route("/api/emotions", methods=["POST"])
def api_detect_emotions():
    """REST API endpoint for emotion detection"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400
        
        text = data['text']
        cleaned_text, validation_error = validate_text_input(text)
        
        if validation_error:
            return jsonify({"error": validation_error}), 400
        
        # Try cache first
        cached_result = get_from_cache(cleaned_text)
        if cached_result:
            processing_time = round(time.time() - start_time, 2)
            return jsonify({
                "emotions": cached_result,
                "processing_time": processing_time,
                "cached": True
            })
        
        # Detect emotions
        emotions = detect_emotions(cleaned_text)
        save_to_cache(cleaned_text, emotions)
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            "emotions": emotions,
            "processing_time": processing_time,
            "cached": False
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/api/emotions/chart", methods=["POST"])
def api_generate_chart():
    """REST API endpoint for generating emotion charts"""
    try:
        data = request.get_json()
        if not data or 'emotions' not in data:
            return jsonify({"error": "Missing 'emotions' field in request body"}), 400
        
        emotions = data['emotions']
        chart_type = data.get('chart_type', 'pie').lower()
        
        # Validate emotions data
        if not emotions or "message" in emotions:
            return jsonify({"error": "No valid emotions data provided"}), 400
        
        # Generate specific chart type
        chart_functions = {
            'pie': create_pie_chart,
            'bar': create_bar_chart,
            'donut': create_donut_chart,
            'horizontal': create_horizontal_bar_chart,
            'polar': create_polar_chart,
            'stacked': create_stacked_chart
        }
        
        if chart_type not in chart_functions:
            return jsonify({"error": f"Invalid chart type. Available: {list(chart_functions.keys())}"}), 400
        
        fig = chart_functions[chart_type](emotions)
        chart_base64 = fig_to_base64(fig)
        
        return jsonify({
            "chart": chart_base64,
            "chart_type": chart_type
        })
        
    except Exception as e:
        logger.error(f"Chart generation error: {str(e)}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route("/api/emotions/all-charts", methods=["POST"])
def api_generate_all_charts():
    """REST API endpoint for generating all chart types"""
    try:
        data = request.get_json()
        if not data or 'emotions' not in data:
            return jsonify({"error": "Missing 'emotions' field in request body"}), 400
        
        emotions = data['emotions']
        
        # Validate emotions data
        if not emotions or "message" in emotions:
            return jsonify({"error": "No valid emotions data provided"}), 400
        
        # Generate all charts
        charts = {}
        chart_functions = {
            "pie": create_pie_chart,
            "bar": create_bar_chart,
            "donut": create_donut_chart,
            "horizontal": create_horizontal_bar_chart,
            "polar": create_polar_chart,
            "stacked": create_stacked_chart
        }
        
        for chart_name, chart_func in chart_functions.items():
            try:
                fig = chart_func(emotions)
                charts[chart_name] = fig_to_base64(fig)
            except Exception as e:
                logger.error(f"Error generating {chart_name} chart: {str(e)}")
                charts[chart_name] = None
        
        return jsonify({"charts": charts})
        
    except Exception as e:
        logger.error(f"All charts generation error: {str(e)}")
        return jsonify({"error": f"Charts generation failed: {str(e)}"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "cache_size": len(emotion_cache),
        "timestamp": time.time()
    })

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the emotion cache"""
    with cache_lock:
        emotion_cache.clear()
    return jsonify({"message": "Cache cleared successfully"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
 
    logger.info("Starting Flask Emotion Detection API...")
    logger.info(f"Cache size limit: {CACHE_SIZE}")
    logger.info(f"Cache TTL: {CACHE_TTL} seconds")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True  
    )