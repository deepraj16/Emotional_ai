import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging
import time
from functools import lru_cache
import os

# Configure matplotlib for better performance
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.max_open_warning'] = 0  # Disable max figure warning

# Setup logging
logger = logging.getLogger(__name__)

# Initialize LLM (Mistral) - Use environment variable for API key
API_KEY = os.getenv("MISTRAL_API_KEY", "lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc")
llm = ChatMistralAI(
    api_key=API_KEY,
    temperature=0.1,  
    max_tokens=500,   
    timeout=30       
)
parser = JsonOutputParser()

# Optimized emotion detection prompt
EMOTION_PROMPT_TEMPLATE = """
Analyze the following text and classify it into **all possible emotions** you detect 
(not just happy/sad/angry). Consider nuanced emotions like contentment, anticipation, 
curiosity, trust, disgust, surprise, fear, etc.

IMPORTANT RULES:
1. If no emotion is detected, return: {{"message": "No, I cannot find emotions."}}
2. Otherwise, return ONLY a JSON object with emotions as keys and percentages as values
3. Percentages MUST be integers that sum to exactly 100
4. Include 2-6 emotions maximum for better visualization

Example format:
{{
  "sadness": 35,
  "hope": 25,
  "fear": 15,
  "anger": 15,
  "excitement": 10
}}

Text: "{text}"

JSON Response:
"""

def detect_emotions(text):
    """
    Optimized LLM-based emotion detection with better error handling and performance.
    Returns percentages summing to 100% or a message if no emotions found.
    """
    if not text or not text.strip():
        return {"message": "No, I cannot find emotions."}
    
    # Clean and truncate text if too long
    text = text.strip()[:2000]  # Limit text length for faster processing
    
    prompt = PromptTemplate(
        template=EMOTION_PROMPT_TEMPLATE,
        input_variables=["text"]
    )

    chain = prompt | llm | parser

    try:
        start_time = time.time()
        emotions = chain.invoke({"text": text})
        processing_time = time.time() - start_time
        logger.info(f"LLM emotion detection completed in {processing_time:.2f}s")

        # Handle "no emotions" case
        if "message" in emotions:
            return emotions

        # Validate and clean emotions data
        if not isinstance(emotions, dict) or not emotions:
            logger.warning("Invalid emotions format from LLM")
            return {"message": "No, I cannot find emotions."}

        # Filter out non-numeric values and ensure integers
        clean_emotions = {}
        for emotion, value in emotions.items():
            try:
                clean_emotions[emotion.lower()] = int(float(value))
            except (ValueError, TypeError):
                logger.warning(f"Invalid emotion value: {emotion}={value}")
                continue

        if not clean_emotions:
            return {"message": "No, I cannot find emotions."}

        # Ensure percentages sum to 100
        total = sum(clean_emotions.values())
        if total <= 0:
            return {"message": "No, I cannot find emotions."}
        
        if total != 100:
            factor = 100.0 / total
            clean_emotions = {k: round(v * factor) for k, v in clean_emotions.items()}
            
            # Adjust for rounding errors
            diff = 100 - sum(clean_emotions.values())
            if diff != 0:
                max_emotion = max(clean_emotions, key=clean_emotions.get)
                clean_emotions[max_emotion] += diff

        return clean_emotions

    except Exception as e:
        logger.error(f"LLM emotion detection failed: {str(e)}")
        return {"message": "No, I cannot find emotions."}


# Optimized chart creation functions with better performance
def create_pie_chart(emotions, title="Emotion Distribution - Pie Chart"):
    """Create optimized pie chart"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
    
    wedges, texts, autotexts = ax.pie(
        emotions.values(),
        labels=emotions.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.05] * len(emotions),
        shadow=True,
        textprops={'fontsize': 10}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def create_bar_chart(emotions, title="Emotion Distribution - Bar Chart"):
    """Create optimized bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    
    bars = ax.bar(emotions.keys(), emotions.values(), color=colors, 
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, emotions.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(emotions.values()) * 1.1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_donut_chart(emotions, title="Emotion Distribution - Donut Chart"):
    """Create optimized donut chart"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(emotions)))
    
    wedges, texts, autotexts = ax.pie(
        emotions.values(),
        labels=emotions.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 10}
    )
    
    # Create center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white', ec='gray', linewidth=2)
    ax.add_artist(centre_circle)
    ax.text(0, 0, "Emotions", ha="center", va="center", 
            fontsize=12, fontweight="bold", color='gray')
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def create_horizontal_bar_chart(emotions, title="Emotion Distribution - Horizontal Bar Chart"):
    """Create optimized horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    colors = plt.cm.plasma(np.linspace(0, 1, len(emotions)))
    
    y_pos = np.arange(len(emotions))
    bars = ax.barh(y_pos, list(emotions.values()), color=colors, 
                   edgecolor="black", linewidth=0.5, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, emotions.values())):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                f'{val}%', va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions.keys())
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(emotions.values()) * 1.1)
    
    plt.tight_layout()
    return fig

def create_polar_chart(emotions, title="Emotion Distribution - Polar Chart"):
    """Create optimized polar chart"""
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection="polar"), facecolor='white')
    
    emotions_list = list(emotions.keys())
    values = list(emotions.values())
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
    
    # Close the plot
    values += values[:1]
    angles += angles[:1]
    emotions_list += emotions_list[:1]
    
    # Plot
    ax.plot(angles, values, "o-", linewidth=3, markersize=8, color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions_list[:-1], fontsize=10)
    ax.set_ylim(0, max(emotions.values()) * 1.1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for angle, value, emotion in zip(angles[:-1], values[:-1], emotions_list[:-1]):
        ax.text(angle, value + max(emotions.values()) * 0.05, f'{value}%', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_stacked_chart(emotions, title="Emotion Distribution - Stacked Chart"):
    """Create optimized stacked chart"""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    
    cumulative = np.cumsum([0] + list(emotions.values())[:-1])
    colors = plt.cm.Set2(np.linspace(0, 1, len(emotions)))
    
    for i, (emotion, value, start, color) in enumerate(zip(emotions.keys(), emotions.values(), cumulative, colors)):
        bar = ax.barh(0, value, left=start, color=color, height=0.6, 
                     edgecolor='white', linewidth=2)
        
        # Add labels in the center of each segment
        if value > 5:  # Only add label if segment is large enough
            ax.text(start + value/2, 0, f"{emotion}\n{value}%", 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

# Cached color palettes for better performance
@lru_cache(maxsize=10)
def get_color_palette(palette_name, n_colors):
    """Get cached color palette"""
    if palette_name == "Set3":
        return plt.cm.Set3(np.linspace(0, 1, n_colors))
    elif palette_name == "viridis":
        return plt.cm.viridis(np.linspace(0, 1, n_colors))
    elif palette_name == "Pastel1":
        return plt.cm.Pastel1(np.linspace(0, 1, n_colors))
    elif palette_name == "plasma":
        return plt.cm.plasma(np.linspace(0, 1, n_colors))
    elif palette_name == "Set2":
        return plt.cm.Set2(np.linspace(0, 1, n_colors))
    else:
        return plt.cm.tab10(np.linspace(0, 1, n_colors))

def validate_emotions_data(emotions):
    """Validate emotions data structure"""
    if not emotions or not isinstance(emotions, dict):
        return False, "Invalid emotions data structure"
    
    if "message" in emotions:
        return False, emotions["message"]
    
    # Check if all values are numeric and positive
    for emotion, value in emotions.items():
        if not isinstance(value, (int, float)) or value < 0:
            return False, f"Invalid value for emotion '{emotion}': {value}"
    
    # Check if percentages sum to approximately 100
    total = sum(emotions.values())
    if not (95 <= total <= 105):  # Allow small rounding errors
        return False, f"Percentages sum to {total}, expected ~100"
    
    return True, "Valid"

def get_emotion_insights(emotions):
    """Generate insights from emotion data"""
    if not emotions or "message" in emotions:
        return None
    
    insights = {}
    total_emotions = len(emotions)
    dominant_emotion = max(emotions, key=emotions.get)
    dominant_percentage = emotions[dominant_emotion]
    
    insights['dominant_emotion'] = dominant_emotion
    insights['dominant_percentage'] = dominant_percentage
    insights['emotion_count'] = total_emotions
    insights['emotion_diversity'] = 'high' if total_emotions > 4 else 'moderate' if total_emotions > 2 else 'low'
    
    # Categorize emotions
    positive_emotions = {'joy', 'happiness', 'excitement', 'love', 'contentment', 'hope', 'trust', 'anticipation'}
    negative_emotions = {'sadness', 'anger', 'fear', 'disgust', 'anxiety', 'frustration', 'disappointment'}
    
    positive_score = sum(v for k, v in emotions.items() if k.lower() in positive_emotions)
    negative_score = sum(v for k, v in emotions.items() if k.lower() in negative_emotions)
    neutral_score = 100 - positive_score - negative_score
    
    insights['sentiment_breakdown'] = {
        'positive': positive_score,
        'negative': negative_score,
        'neutral': neutral_score
    }
    
    return insights

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper
