import os
import pandas as pd
import requests
from flask import Flask, request, send_from_directory, jsonify, render_template
from werkzeug.utils import secure_filename
import nltk
from tqdm import tqdm
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from lxml import html
from dateutil import parser as dateparser
from wordcloud import WordCloud
import time
import tempfile

nltk.download(['vader_lexicon', 'stopwords', 'punkt'])

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ARTICLES_FOLDER = 'Products'
OUTPUT_FOLDER = 'static/output'
WORDCLOUD_FOLDER = 'static/wordclouds'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTICLES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WORDCLOUD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

# Selenium configuration
custom_cache_dir = os.path.join(tempfile.gettempdir(), "selenium_cache")
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['WDM_LOCAL'] = '1'
os.environ['WDM_CACHE_DIR'] = custom_cache_dir

def get_html_sources(url, product_id):
    options = Options()
    options.add_argument('--headless --no-sandbox --disable-dev-shm-usage --user-agent=Mozilla/5.0')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    sources = {}

    try:
        driver.get(url)
        time.sleep(2)
        sources[f'{product_id}_top'] = driver.page_source

        try:
            select = Select(driver.find_element(By.ID, 'cm-cr-sort-dropdown'))
            select.select_by_value('recent')
            time.sleep(6)
            sources[f'{product_id}_recent'] = driver.page_source
        except Exception as e:
            print(f"[WARNING] Couldn't switch to recent reviews: {e}")
            sources[f'{product_id}_recent'] = sources[f'{product_id}_top']
    finally:
        driver.quit()
    return sources

def scrape_reviews_apify(url):
    XPATH_REVIEWS = '//div[contains(@id,"customer_review-")]'
    XPATH_RATING = './/i/span/text()'
    XPATH_DATE = './/span[contains(@data-hook,"review-date")]/text()'
    XPATH_AUTHOR = './/span[contains(@class,"a-profile-name")]/text()'
    XPATH_REVIEW_TEXT = './/div[@data-hook="review-collapsed"]/span/text()'

    product_id = url.split('/dp/')[1].split('/')[0] if '/dp/' in url else 'unknown'
    sources = get_html_sources(url, product_id)
    all_reviews = []

    for source_key, page_html in sources.items():
        parser_obj = html.fromstring(page_html)
        reviews = parser_obj.xpath(XPATH_REVIEWS)

        for review in reviews:
            try:
                review_data = {
                    'author': ''.join(review.xpath(XPATH_AUTHOR)).strip(),
                    'rating': ''.join(review.xpath(XPATH_RATING)).replace(' out of 5 stars', '').strip(),
                    'date': dateparser.parse(''.join(review.xpath(XPATH_DATE)).strip().split('on ')[-1]).strftime('%d %b %Y'),
                    'text': ''.join(review.xpath(XPATH_REVIEW_TEXT)).strip(),
                    'product_id': product_id,
                    'link': url
                }
                if review_data['text']:
                    all_reviews.append(review_data)
            except Exception as e:
                print(f"[ERROR] Parsing review failed: {e}")
                continue

    # Sentiment analysis based on ratings
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_rating = 0.0
    valid_reviews = 0

    for review in all_reviews:
        try:
            rating = float(review['rating'])
            total_rating += rating
            valid_reviews += 1

            if rating >= 4:
                sentiment_counts['positive'] += 1
                review['sentiment'] = 'positive'
            elif rating == 3:
                sentiment_counts['neutral'] += 1
                review['sentiment'] = 'neutral'
            else:
                sentiment_counts['negative'] += 1
                review['sentiment'] = 'negative'
        except:
            review['sentiment'] = 'unknown'

    average_rating = round(total_rating / valid_reviews, 2) if valid_reviews else 0.0
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    accuracy = round((sentiment_counts[dominant_sentiment] / valid_reviews) * 100, 2) if valid_reviews else 0.0

    return {
        "product_id": product_id,
        "product_url": url,
        "summary": {
            "total_reviews": len(all_reviews),
            "average_rating": average_rating,
            "sentiment_breakdown": sentiment_counts,
            "overall_sentiment": dominant_sentiment,
            "accuracy_score": accuracy
        },
        "reviews": all_reviews
    }

def sanitize_filename(text):
    return re.sub(r'\W+', '_', text.strip())

def extract_products(input_file):
    data = pd.read_excel(input_file)
    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting"):
        url_id, url = row['URL_ID'], row['URL']
        safe_id = sanitize_filename(url_id)
        json_path = os.path.join(ARTICLES_FOLDER, f'{safe_id}.json')

        if not os.path.exists(json_path):
            reviews_data = scrape_reviews_apify(url)
            if reviews_data.get("reviews"):
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(reviews_data, f)

                # Generate word cloud
                text_blob = ' '.join(r["text"] for r in reviews_data["reviews"])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)
                wordcloud.to_file(os.path.join(WORDCLOUD_FOLDER, f'{safe_id}.png'))

def analyze_product(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get("reviews"):
        return {}

    summary = data["summary"]
    return {
        'URL_ID': data["product_id"],
        'OVERALL_RATING': summary["average_rating"],
        'OVERALL_SENTIMENT': summary["overall_sentiment"].capitalize(),
        'ACCURACY_SCORE': f"{summary['accuracy_score']}%",
        'POSITIVE_REVIEWS': summary["sentiment_breakdown"]["positive"],
        'NEUTRAL_REVIEWS': summary["sentiment_breakdown"]["neutral"],
        'NEGATIVE_REVIEWS': summary["sentiment_breakdown"]["negative"],
        'TOTAL_REVIEWS': summary["total_reviews"],
        'FULL_ANALYSIS': json.dumps(data),
        'URL': data["product_url"]
    }

def generate_results(input_file, output_file):
    data = pd.read_excel(input_file)
    results = []

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Analyzing"):
        safe_id = sanitize_filename(row['URL_ID'])
        json_path = os.path.join(ARTICLES_FOLDER, f'{safe_id}.json')

        if os.path.exists(json_path):
            analysis = analyze_product(json_path)
            analysis.update({'URL_ID': row['URL_ID'], 'URL': row['URL']})
            results.append(analysis)

    pd.DataFrame(results).to_excel(output_file, index=False, engine='openpyxl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extract_products(filepath)
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx')
        generate_results(filepath, output_file)

        return jsonify({"download_link": "/download"})

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/download')
def download_file():
    return send_from_directory(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
