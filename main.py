import os
import pandas as pd
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
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from lxml import html
from dateutil import parser as dateparser
from wordcloud import WordCloud
import time
import tempfile
import joblib
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK data
nltk.download(['stopwords', 'punkt'], quiet=True)

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
app.config['WORDCLOUD_FOLDER'] = WORDCLOUD_FOLDER

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

# Load the trained Naive Bayes model and vectorizer
try:
    NB_model = joblib.load('NB_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
except FileNotFoundError:
    raise Exception("Naive Bayes model or vectorizer not found. Please train the model using train_model.py first.")

# Selenium configuration
custom_cache_dir = os.path.join(tempfile.gettempdir(), "selenium_cache")
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['WDM_LOCAL'] = '1'
os.environ['WDM_CACHE_DIR'] = custom_cache_dir



def get_html_sources(url, product_id):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            '''
        })

        sources = {}
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//div[contains(@id,"customer_review-")]'))
            )
            sources[f'{product_id}_top'] = driver.page_source

            try:
                select = Select(WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'cm-cr-sort-dropdown'))
                ))
                select.select_by_value('recent')
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, '//div[contains(@id,"customer_review-")]'))
                )
                sources[f'{product_id}_recent'] = driver.page_source
            except Exception as e:
                print(f"[WARNING] Couldn't switch to recent reviews for {url}: {e}")
                sources[f'{product_id}_recent'] = sources[f'{product_id}_top']
        except TimeoutException as e:
            print(f"[ERROR] Timeout loading reviews for {url}: {e}")
        except WebDriverException as e:
            print(f"[ERROR] WebDriver error for {url}: {e}")
        finally:
            driver.quit()
        return sources
    except Exception as e:
        print(f"[ERROR] Failed to initialize WebDriver for {url}: {e}")
        return {}

from urllib.parse import unquote

@app.route('/product-details/<path:product_name>')
def get_product_details(product_name):
    decoded_name = unquote(product_name)
    safe_name = sanitize_filename(decoded_name)
    json_path = os.path.join(ARTICLES_FOLDER, f'{safe_name}.json')

    print(f"[DEBUG] Request for product: {product_name}, decoded: {decoded_name}, safe_name: {safe_name}, json_path: {json_path}")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Add word cloud URL for other tabs (optional)
            wordcloud_filename = f'{safe_name}.png'
            wordcloud_path = os.path.join(app.config['WORDCLOUD_FOLDER'], wordcloud_filename)
            data['wordcloud_url'] = f"/static/wordclouds/{wordcloud_filename}" if os.path.exists(wordcloud_path) else None
            print(f"[DEBUG] Loaded product data for {safe_name}: {data['summary']}")
            return jsonify(data)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {json_path}: {e}")
            return jsonify({"error": f"Failed to load product data: {str(e)}"}), 500
    else:
        print(f"[ERROR] JSON file not found: {json_path}")
        return jsonify({"error": "Product not found", "wordcloud_url": None}), 404


def preprocess_text(text):
    """Preprocess text for Naive Bayes model."""
    if not isinstance(text, str) or not text.strip():
        print("[WARNING] Empty or invalid text for preprocessing")
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(reviews):
    """Predict sentiment using the Naive Bayes model and map to labels."""
    if not reviews:
        print("[WARNING] No reviews to analyze.")
        return []

    processed_reviews = []
    for review in reviews:
        text = review.get('text', '')
        processed = preprocess_text(text)
        if not processed:
            print(f"[WARNING] Processed text is empty for review: {text[:50]}...")
        processed_reviews.append(processed)

    reviews_TFIDF = vectorizer.transform(processed_reviews)
    try:
        predictions = NB_model.predict(reviews_TFIDF)
        print(f"[DEBUG] Model predictions: {list(predictions)}")
        sentiments = [sentiment_mapping.get(p, "unknown") for p in predictions]
        print(f"[DEBUG] Predicted sentiments: {sentiments}")
        return sentiments
    except Exception as e:
        print(f"[ERROR] Sentiment prediction failed: {e}")
        return ['unknown'] * len(reviews)

def scrape_reviews_apify(url):
    XPATH_REVIEWS = '//div[contains(@id,"customer_review-")]'
    XPATH_RATING = './/i[@data-hook="review-star-rating"]/span/text()'
    XPATH_DATE = './/span[contains(@data-hook,"review-date")]/text()'
    XPATH_AUTHOR = './/span[contains(@class,"a-profile-name")]/text()'
    XPATH_REVIEW_TEXT = './/span[@data-hook="review-body"]//text()'

    XPATH_PRODUCT_NAME = '//span[@id="productTitle"]/text()'

        # Extract product name from first available source
    product_name = "Unknown Product"


    product_id = url.split('/dp/')[1].split('/')[0] if '/dp/' in url else 'unknown'
    sources = get_html_sources(url, product_id)
    all_reviews = []

    if not sources:
        print(f"[ERROR] No HTML sources retrieved for {url}")
        return {
            "product_id": product_id,
            "product_url": url,
            "summary": {
                "total_reviews": 0,
                "average_rating": 0.0,
                "sentiment_breakdown": {},
                "overall_sentiment": "unknown",
                "accuracy_score": 0.0
            },
            "reviews": []
        }

    if sources:
            first_source = next(iter(sources.values()))
            parser_obj = html.fromstring(first_source)
            name_elem = parser_obj.xpath(XPATH_PRODUCT_NAME)
            if name_elem:
                product_name = name_elem[0].strip()

    for source_key, page_html in sources.items():
        parser_obj = html.fromstring(page_html)
        reviews = parser_obj.xpath(XPATH_REVIEWS)

        if not reviews:
            print(f"[WARNING] No reviews found for {url} in {source_key}")

        for review in reviews:
            try:
                review_text = ''.join(review.xpath(XPATH_REVIEW_TEXT)).strip()
                review_data = {
                    'author': ''.join(review.xpath(XPATH_AUTHOR)).strip() or 'Anonymous',
                    'rating': ''.join(review.xpath(XPATH_RATING)).replace(' out of 5 stars', '').strip(),
                    'date': dateparser.parse(''.join(review.xpath(XPATH_DATE)).strip().split('on ')[-1]).strftime('%d %b %Y') if review.xpath(XPATH_DATE) else 'Unknown',
                    'text': review_text,
                    'product_id': product_id,
                    'link': url
                }
                if review_data['text']:
                    all_reviews.append(review_data)
                else:
                    print(f"[WARNING] Empty review text for {url}")
            except Exception as e:
                print(f"[ERROR] Parsing review failed for {url}: {e}")
                continue

    sentiments = predict_sentiment(all_reviews)
    sentiment_counts = Counter(sentiments)
    total_reviews = len(sentiments)

    for review, sentiment in zip(all_reviews, sentiments):
        review['sentiment'] = sentiment if sentiment else 'unknown'

    total_rating = 0.0
    valid_reviews = 0
    for review in all_reviews:
        try:
            rating = float(review['rating'])
            total_rating += rating
            valid_reviews += 1
        except:
            continue
    average_rating = round(total_rating / valid_reviews, 2) if valid_reviews else 0.0

    dominant_sentiment = 'unknown'
    accuracy = 0.0
    if total_reviews > 0 and sentiment_counts:
        try:
            dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get).lower()
            accuracy = round((sentiment_counts[dominant_sentiment] / total_reviews) * 100, 2)
        except Exception as e:
            print(f"[ERROR] Failed to determine dominant sentiment for {url}: {e}")

    print(f"[DEBUG] Scraped {url}: total_reviews={total_reviews}, sentiment_counts={dict(sentiment_counts)}, dominant_sentiment={dominant_sentiment}, accuracy={accuracy}")

    return {
        "product_id": product_id,
        "product_name": product_name,
        "product_url": url,
        "summary": {
            "total_reviews": total_reviews,
            "average_rating": average_rating,
            "sentiment_breakdown": dict(sentiment_counts),
            "overall_sentiment": dominant_sentiment,
            "accuracy_score": accuracy
        },
        "reviews": all_reviews
    }

def sanitize_filename(text):
    return re.sub(r'\W+', '_', str(text).strip())

def extract_products(input_file):
    try:
        data = pd.read_excel(input_file)
        if 'URL_ID' not in data.columns or not ('URL' in data.columns or 'PRODUCT_URL' in data.columns):
            raise ValueError("Input Excel file must contain 'URL_ID' and either 'URL' or 'PRODUCT_URL' columns")
        if 'PRODUCT_URL' in data.columns and 'URL' not in data.columns:
            data = data.rename(columns={'PRODUCT_URL': 'URL'})
    except Exception as e:
        print(f"[ERROR] Failed to read input file {input_file}: {e}")
        return

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Extracting"):
        url_id, url = row['URL_ID'], row['URL']
        safe_id = sanitize_filename(str(url_id))
        json_path = os.path.join(ARTICLES_FOLDER, f'{safe_id}.json')

        if not os.path.exists(json_path):
            print(f"[INFO] Scraping reviews for {url_id}: {url}")
            reviews_data = scrape_reviews_apify(url)
            if reviews_data.get("reviews"):
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(reviews_data, f, ensure_ascii=False)

                text_blob = ' '.join(r["text"] for r in reviews_data["reviews"])
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_blob)

                    # Save with URL_ID naming
                    wordcloud.to_file(os.path.join(WORDCLOUD_FOLDER, f'{safe_id}.png'))

                    # Also save with product_id naming if available
                    product_id = reviews_data.get("product_id")
                    if product_id and product_id != "unknown":
                        wordcloud.to_file(os.path.join(WORDCLOUD_FOLDER, f'{product_id}.png'))
                except ValueError as e:
                    print(f"[WARNING] Failed to generate word cloud for {url_id}: {e}")
            else:
                print(f"[WARNING] Skipping word cloud generation for {url_id}: No reviews available")


def analyze_product(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON file {json_path}: {e}")
        return {}

    if not data.get("reviews") or not data.get("summary"):
        print(f"[WARNING] Invalid or empty data in {json_path}")
        return {}

    summary = data["summary"]
    overall_sentiment = summary.get("overall_sentiment", "unknown")
    if not isinstance(overall_sentiment, str):
        print(f"[WARNING] Invalid overall_sentiment in {json_path}: {overall_sentiment}")
        overall_sentiment = "unknown"
    overall_sentiment = overall_sentiment.capitalize()

    try:
        full_analysis = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to serialize FULL_ANALYSIS for {json_path}: {e}")
        full_analysis = json.dumps({"error": "Invalid data", "original_data": str(data)})

    full_analysis = full_analysis.replace("'", "\\'").replace('"', '\\"')

    result = {
        'URL_ID': data.get("product_id", "unknown"),
        'OVERALL_RATING': summary.get("average_rating", 0.0),
        'OVERALL_SENTIMENT': overall_sentiment,
        'ACCURACY_SCORE': f"{summary.get('accuracy_score', 0.0)}%",
        'POSITIVE_REVIEWS': summary.get("sentiment_breakdown", {}).get("positive", 0),
        'NEUTRAL_REVIEWS': summary.get("sentiment_breakdown", {}).get("neutral", 0),
        'NEGATIVE_REVIEWS': summary.get("sentiment_breakdown", {}).get("negative", 0),
        'TOTAL_REVIEWS': summary.get("total_reviews", 0),
        'FULL_ANALYSIS': full_analysis,
        'URL': data.get("product_url", "")
    }
    print(f"[DEBUG] Analyzed {json_path}: {result}")
    return result

def generate_results(input_file, output_file):
    try:
        data = pd.read_excel(input_file)
        if 'URL_ID' not in data.columns or not ('URL' in data.columns or 'PRODUCT_URL' in data.columns):
            raise ValueError("Input Excel file must contain 'URL_ID' and either 'URL' or 'PRODUCT_URL' columns")
        if 'PRODUCT_URL' in data.columns and 'URL' not in data.columns:
            data = data.rename(columns={'PRODUCT_URL': 'URL'})
    except Exception as e:
        print(f"[ERROR] Failed to read input file {input_file}: {e}")
        return

    results = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Analyzing"):
        safe_id = sanitize_filename(row['URL_ID'])
        json_path = os.path.join(ARTICLES_FOLDER, f'{safe_id}.json')

        if os.path.exists(json_path):
            analysis = analyze_product(json_path)
            if analysis:
                analysis.update({'URL_ID': row['URL_ID'], 'URL': row['URL']})
                results.append(analysis)
        else:
            print(f"[WARNING] JSON file not found for {row['URL_ID']}: {json_path}")

    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"[DEBUG] Saved results to {output_file}: {df.to_dict(orient='records')}")
    else:
        print("[ERROR] No valid analysis results to save")

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

        try:
            extract_products(filepath)
            output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx')
            generate_results(filepath, output_file)
            return jsonify({"download_link": "/download"})
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/download')
def download_file():
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Output file not found"}), 404

@app.route('/results')
def get_results():
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx')
    if not os.path.exists(output_file):
        return jsonify({"error": "No analysis results available"}), 404

    try:
        data = pd.read_excel(output_file)
        results = data.to_dict(orient='records')
        print(f"[DEBUG] Serving /results: {results}")
        return jsonify(results)
    except Exception as e:
        print(f"[ERROR] Failed to read results: {e}")
        return jsonify({"error": f"Failed to read results: {str(e)}"}), 500




@app.route('/static/wordclouds/<filename>')
def serve_wordcloud(filename):
    wordcloud_path = os.path.join(app.config['WORDCLOUD_FOLDER'], filename)

    # First try to serve the exact filename
    if os.path.exists(wordcloud_path):
        return send_from_directory(app.config['WORDCLOUD_FOLDER'], filename)

    # If not found, try alternative naming patterns
    try:
        # Check if this is a product_id looking for URL_ID version
        if re.match(r'^[A-Z0-9]{10}$', filename.split('.')[0]):  # Amazon ASIN pattern
            # Look for matching URL_ID version
            results_file = os.path.join(app.config['OUTPUT_FOLDER'], 'analysis_results.xlsx')
            if os.path.exists(results_file):
                df = pd.read_excel(results_file)
                product_row = df[df['URL_ID'] == filename.split('.')[0]]
                if not product_row.empty:
                    url_id = product_row.iloc[0]['URL_ID']
                    safe_id = sanitize_filename(url_id)
                    alt_filename = f'{safe_id}.png'
                    if os.path.exists(os.path.join(app.config['WORDCLOUD_FOLDER'], alt_filename)):
                        return send_from_directory(app.config['WORDCLOUD_FOLDER'], alt_filename)

        # Default fallback
        default_path = os.path.join(app.config['WORDCLOUD_FOLDER'], 'default.png')
        if os.path.exists(default_path):
            print(f"[INFO] Serving default word cloud for {filename}")
            return send_from_directory(app.config['WORDCLOUD_FOLDER'], 'default.png')
        else:
            print(f"[ERROR] Default word cloud not found: {default_path}")
            return jsonify({"error": f"Word cloud not found for {filename} and default.png is missing"}), 404
    except Exception as e:
        print(f"[ERROR] Failed to serve word cloud: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
