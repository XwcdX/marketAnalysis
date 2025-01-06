from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid

app = Flask(__name__)
CORS(app)

TEMP_FOLDER = os.path.join('static', 'tmp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Preprocess data function
def preprocess_data(data):
    """Preprocess the dataset."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    required_columns = {'Invoice', 'Customer ID', 'Description', 'Country', 'Quantity', 'Price'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data is missing one or more required columns: {required_columns}")
    
    try:
        logging.info("Starting data preprocessing.")
        data = data[~data['Invoice'].astype(str).str.startswith('C', na=False)] 
        data = data.dropna(subset=['Customer ID', 'Description', 'Country'])
        data = data[(data['Quantity'] > 0) & (data['Price'] > 0)].drop_duplicates()
        
        data['TotalPurchase'] = data['Quantity'] * data['Price']
        
        data['Country_Code'] = data['Country'].astype('category').cat.codes
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
        data['Hour'] = data['InvoiceDate'].dt.hour
        data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
        data['Month'] = data['InvoiceDate'].dt.month

        logging.info("Data preprocessing completed.")
        return data

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def visualize_country(data, plot_path):
    if 'Country' not in data or 'Invoice' not in data:
        raise ValueError("Missing required columns: 'Country', 'Invoice'.")

    country_purchase_frequency = (
        data.groupby('Country')['Invoice']
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    country_purchase_frequency.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Countries by Purchase Frequency', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Number of Invoices', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def visualize_product(data, plot_path):
    if 'Description' not in data:
        raise ValueError("Missing required column: 'Description'.")

    product_frequency = data['Description'].value_counts().head(20)
    product_frequency.plot(kind='barh', color='lightcoral')
    plt.title('Top 20 Products by Purchase Frequency', fontsize=14)
    plt.xlabel('Number of Invoices', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def visualize_day(data, plot_path):
    if 'DayOfWeek' not in data:
        raise ValueError("Missing required column: 'DayOfWeek'.")

    day_of_week_trend = (
        data.groupby('DayOfWeek').size().reset_index(name='TransactionCount')
    )
    sns.barplot(
        x='DayOfWeek',
        y='TransactionCount',
        data=day_of_week_trend,
        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    plt.title('Transactions by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def visualize_month(data, plot_path):
    if 'Month' not in data:
        raise ValueError("Missing required column: 'Month'.")

    monthly_trend = data.groupby('Month').size().reset_index(name='TransactionCount')
    plt.plot(monthly_trend['Month'], monthly_trend['TransactionCount'], marker='o')
    plt.title('Transactions by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def visualize_hour(data, plot_path):
    if 'Hour' not in data:
        raise ValueError("Missing required column: 'Hour'.")

    hourly_trend = data.groupby('Hour').size().reset_index(name='TransactionCount')
    sns.lineplot(x='Hour', y='TransactionCount', data=hourly_trend)
    plt.title('Transactions by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Transactions')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

# Extract customer data for clustering
def extract_customer_data(data):
    """Extract customer-level data for clustering."""
    required_columns = {'Customer ID', 'TotalPurchase', 'Invoice', 'Country_Code', 'Hour', 'DayOfWeek', 'Month'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data is missing one or more required columns: {required_columns}")

    try:
        customer_data = data.groupby('Customer ID').agg({
            'TotalPurchase': 'sum',
            'Invoice': 'nunique',
            'Country_Code': 'first',
            'Hour': lambda x: x.mode()[0] if not x.mode().empty else 0,
            'DayOfWeek': lambda x: x.mode()[0] if not x.mode().empty else 'Monday',
            'Month': 'nunique'
        }).reset_index()

        customer_data.columns = ['Customer ID', 'TotalPurchase', 'JumlahTransaksi', 'Country_Code',
                                 'MostActiveHour', 'MostActiveDay', 'MonthCount']

        # Map day names to numeric values
        day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                       'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        customer_data['MostActiveDay'] = (
            customer_data['MostActiveDay']
            .map(day_mapping)
            .fillna(0)  # Fallback for unmapped days
            .astype(int)
        )

        logging.info("Customer data extraction completed.")
        return customer_data

    except Exception as e:
        logging.error(f"Error during customer data extraction: {e}")
        raise

# Scale features function
def scale_features(data):
    """Scale features using StandardScaler."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    logging.info("Features scaled successfully.")
    return scaled_data

# Find optimal clusters function using Elbow and Silhouette method
def find_optimal_clusters(scaled_data, k_range):
    """Determine the optimal number of clusters using Elbow and Silhouette methods."""
    distortions = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, distortions, 'bo-', markersize=8)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-', markersize=8)
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()

    plot_path = os.path.join(TEMP_FOLDER, f"cluster_analysis_{uuid.uuid4().hex}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# K-Means clustering function
def cluster_customers(customer_data, scaled_data, n_clusters):
    """Apply K-Means clustering to the customer data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_data = customer_data.copy()
    customer_data['Cluster'] = kmeans.fit_predict(scaled_data)
    logging.info(f"Clustering completed with {n_clusters} clusters.")
    return customer_data, kmeans

# Visualize clusters function
def visualize_clusters(customer_data):
    """Generate a scatter plot for customer clusters."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_data,
        x='TotalPurchase',
        y='JumlahTransaksi',
        hue='Cluster',
        palette='viridis'
    )
    plt.title('Segmentasi Pelanggan')
    plt.xlabel('Total Purchase')
    plt.ylabel('Jumlah Transaksi')

    plot_filename = f"customer_clusters_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(TEMP_FOLDER, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_filename

def cluster_analysis(customer_data):
    analysis = customer_data.groupby('Cluster').agg({
        'TotalPurchase': 'mean',
        'JumlahTransaksi': 'mean',
        'Country_Code': 'mean',
        'MostActiveHour': 'mean',
        'MostActiveDay': 'mean',
        'MonthCount': 'mean'
    }).reset_index()

    return analysis.to_dict(orient='records')

def hour_analysis(customer_data, plot_path):
    """Generate hour analysis visualization."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=customer_data, x='Cluster', y='MostActiveHour', palette='viridis')
    plt.title('Distribusi Jam Aktif per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Jam Aktif')

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def day_analysis(customer_data, plot_path):
    """Generate day analysis visualization."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=customer_data, x='Cluster', y='MostActiveDay', palette='Set2')
    plt.title('Distribusi Hari Aktif per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Hari Aktif (0=Senin, 6=Minggu)')

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
def month_analysis(customer_data, plot_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=customer_data, x='Cluster', y='MonthCount', palette='coolwarm')
    plt.title('Distribusi Bulan Aktif per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Jumlah Bulan Aktif')
    
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def country_analysis(customer_data, plot_path):
    """Generate country analysis visualization."""
    country_distribution = customer_data.groupby(['Cluster', 'Country_Code']).size().reset_index(name='Frequency')
    country_distribution = country_distribution.sort_values(['Cluster', 'Frequency'], ascending=[True, False])
    top_countries = country_distribution.groupby('Cluster').head(5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_countries, x='Country_Code', y='Frequency', hue='Cluster', dodge=False)
    plt.title('Top 5 Countries per Cluster')
    plt.xlabel('Country Code')
    plt.ylabel('Frequency')
    plt.tight_layout()

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

# Prepare basket data for frequent itemset mining
def prepare_basket_data(data):
    """Prepare transaction data for frequent itemset mining."""
    basket = data.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = (basket > 0)
    return basket

# Perform Apriori algorithm and generate association rules
def perform_apriori(basket, min_support):
    """Perform Apriori analysis and generate association rules."""
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=1)
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    filtered_rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1.5)]
    
    logging.info("Frequent itemset mining completed.")
    return filtered_rules


# Train Naive Bayes classifier
def train_naive_bayes(customer_data):
    """Train a Naive Bayes classifier for sentiment analysis."""
    required_columns = {'TotalPurchase', 'JumlahTransaksi'}
    if not required_columns.issubset(customer_data.columns):
        raise ValueError(f"Customer data must contain columns: {required_columns}")

    customer_data = customer_data.copy() 
    customer_data['Sentiment'] = customer_data['TotalPurchase'].apply(lambda x: 1 if x > 500 else 0)

    X = customer_data[['TotalPurchase', 'JumlahTransaksi']]
    y = customer_data['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    logging.info("Naive Bayes training and prediction completed.")

    plot_filename = f"naive_bayes_cm_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(TEMP_FOLDER, plot_filename)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "classification_report": report,
        "confusion_matrix_plot": plot_path
    }

# Train K-Nearest Neighbors (KNN) classifier
def train_knn(customer_data, n_neighbors=5):
    """Train a K-Nearest Neighbors classifier."""
    required_columns = {'TotalPurchase', 'JumlahTransaksi'}
    if not required_columns.issubset(customer_data.columns):
        raise ValueError(f"Customer data must contain columns: {required_columns}")

    customer_data = customer_data.copy()
    customer_data['Category'] = customer_data['TotalPurchase'].apply(lambda x: 1 if x > 500 else 0)

    X = customer_data[['TotalPurchase', 'JumlahTransaksi']]
    y = customer_data['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    logging.info("KNN training and prediction completed.")

    plot_filename = f"knn_cm_{uuid.uuid4().hex}.png"
    plot_path = os.path.join(TEMP_FOLDER, plot_filename)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "classification_report": report,
        "confusion_matrix_plot": plot_path
    }

# Recommend products based on association rules
def recommend_products(rules, purchased_products, product_mapping):
    """Generate product recommendations based on association rules."""
    if rules.empty:
        logging.warning("Rules dataset is empty.")
        return {"recommendations": []}

    recommendations = set()
    for product in purchased_products:
        matching_rules = rules[rules['antecedents'].str.contains(product, na=False, case=False)]
        for _, rule in matching_rules.iterrows():
            recommendations.update(rule['consequents'].split(', '))

    # Exclude already purchased products
    filtered_recommendations = [
        product_mapping.get(product.strip(), product)
        for product in recommendations
        if product.strip() not in purchased_products
    ]

    logging.info(f"Generated {len(filtered_recommendations)} recommendations.")
    return {"recommendations": filtered_recommendations[:5]}


# Global variables for dataset and processed data
global_data = None
customer_data = None
rules = None
kmeans_model = None
clustered_data = None

@app.route('/load', methods=['POST'])
def load():
    """Load dataset from an Excel file and store in global_data."""
    global global_data
    file_path = request.json.get('file_path')

    if not file_path:
        return jsonify({"error": "File path not provided."}), 400

    if not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        sheets_to_load = sheet_names[:2]
        global_data = pd.concat(
            [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets_to_load],
            ignore_index=True
        )

        return jsonify({"message": f"Dataset loaded successfully from {len(sheets_to_load)} sheets."}), 200

    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404

    except ValueError as ve:
        return jsonify({"error": f"Error reading Excel file: {str(ve)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error during dataset loading: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/preprocess', methods=['POST'])
def preprocess():
    global global_data, customer_data
    if global_data is None:
        return jsonify({"error": "No dataset loaded."}), 400

    try:
        processed_data = preprocess_data(global_data)
        global_data = processed_data
        customer_data = extract_customer_data(global_data)

        return jsonify({"message": "Data preprocessing and customer extraction completed."}), 200

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/apriori', methods=['POST'])
def apriori_analysis():
    global rules, global_data
    if global_data is None:
        return jsonify({"error": "No dataset loaded."}), 400

    min_support = request.json.get('min_support', 0.015)

    try:
        basket = prepare_basket_data(global_data)
        rules = perform_apriori(basket, min_support)
        results = []
        for _, rule in rules.iterrows():
            results.append({
                "antecedents": rule['antecedents'],
                "consequents": rule['consequents'],
                "support": round(rule['support'], 4),
                "confidence": round(rule['confidence'], 4),
                "lift": round(rule['lift'], 4),
            })
        if not results:
            return jsonify({"message": "No rules found.", "results": []}), 200

        return jsonify({"message": "Apriori analysis completed.", "results": results}), 200
    except Exception as e:
        logging.error(f"Error in Apriori analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    global rules, global_data
    if global_data is None:
        return jsonify({"error": "No dataset loaded."}), 400
    if rules is None:
        return jsonify({"error": "No association rules available. Run the Apriori analysis first."}), 400

    purchased_products = request.json.get('products', [])
    if not isinstance(purchased_products, list) or not purchased_products:
        return jsonify({"error": "Invalid or empty product list provided."}), 400

    try:
        product_mapping = dict(zip(global_data['Description'].drop_duplicates(), 
                                   global_data['Description'].drop_duplicates()))
        recommendations = recommend_products(rules, purchased_products, product_mapping)
        return jsonify(recommendations), 200

    except Exception as e:
        logging.error(f"Error in recommendation generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/visualize/<string:chart_type>', methods=['GET'])
def visualize(chart_type):
    global global_data
    if global_data is None:
        return jsonify({"error": "No dataset loaded."}), 400

    try:
        plot_filename = f"{chart_type}_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(TEMP_FOLDER, plot_filename)
        if chart_type == 'country':
            visualize_country(global_data, plot_path)
        elif chart_type == 'product':
            visualize_product(global_data, plot_path)
        elif chart_type == 'day':
            visualize_day(global_data, plot_path)
        elif chart_type == 'month':
            visualize_month(global_data, plot_path)
        elif chart_type == 'hour':
            visualize_hour(global_data, plot_path)
        else:
            return jsonify({
                "error": "Invalid chart type.",
                "valid_types": ["country", "product", "day", "month", "hour"]
            }), 400
        return jsonify({"image_url": plot_path}), 200
    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train/naive_bayes', methods=['GET'])
def train_nb():
    global customer_data
    if customer_data is None:
        return jsonify({"error": "No customer data available."}), 400

    try:
        report = train_naive_bayes(customer_data)
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"Error during Naive Bayes training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train/knn', methods=['POST'])
def train_knn_model():
    global customer_data
    if customer_data is None:
        return jsonify({"error": "No customer data available."}), 400

    n_neighbors = request.json.get('n_neighbors', 5)

    try:
        report = train_knn(customer_data, n_neighbors)
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"Error during KNN training: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/analyze/<string:analysis_type>', methods=['GET'])
def analyze(analysis_type):
    global customer_data
    if customer_data is None:
        return jsonify({"error": "No customer data available."}), 400

    try:
        plot_filename = f"{analysis_type}_analysis_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(TEMP_FOLDER, plot_filename)
        if analysis_type == 'hour':
            hour_analysis(customer_data, plot_path)
        elif analysis_type == 'day':
            day_analysis(customer_data, plot_path)
        elif analysis_type == 'month':
            month_analysis(customer_data, plot_path)
        elif analysis_type == 'country':
            country_analysis(customer_data, plot_path)
        else:
            return jsonify({
                "error": "Invalid analysis type.",
                "valid_types": ["hour", "day", "country"]
            }), 400
        return jsonify({"image_url": plot_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/cluster', methods=['POST'])
def cluster():
    """Perform clustering on customer data."""
    global customer_data, clustered_data, kmeans_model
    n_clusters = request.json.get('n_clusters', 3)

    if customer_data is None:
        return jsonify({"error": "No customer data available."}), 400

    try:
        scaled_data = scale_features(customer_data[['TotalPurchase', 'JumlahTransaksi', 'Country_Code', 'MostActiveHour', 'MostActiveDay', 'MonthCount']])
        customer_data, kmeans_model = cluster_customers(customer_data, scaled_data, n_clusters)
        clustered_data = customer_data.copy()

        return jsonify({"clusters": customer_data.to_dict(orient='records')}), 200
    except Exception as e:
        logging.error(f"Error during clustering: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/optimal_clusters', methods=['POST'])
def optimal_clusters():
    global customer_data
    if customer_data is None:
        return jsonify({"error": "No customer data available. Please preprocess first."}), 400

    try:
        data = request.get_json()
        k_range = data.get('k_range', list(range(2, 10)))

        if not isinstance(k_range, list) or not all(isinstance(k, int) for k in k_range):
            return jsonify({"error": "Invalid k_range. Must be a list of integers."}), 400

        scaled_data = scale_features(customer_data[['TotalPurchase', 'JumlahTransaksi', 'Country_Code', 'MostActiveHour', 'MostActiveDay', 'MonthCount']])
        plot_path = find_optimal_clusters(scaled_data, k_range)

        return jsonify({"message": "Optimal cluster analysis completed.", "image_url": f"{plot_path}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/visualize_clusters', methods=['GET'])
def visualize_clusters_route():
    """Route to visualize customer clusters."""
    global clustered_data
    if clustered_data is None:
        return jsonify({"error": "No clustered data available. Please perform clustering first."}), 400

    try:
        plot_path = os.path.join(TEMP_FOLDER, visualize_clusters(clustered_data))
        return jsonify({"message": "Cluster visualization completed.", "image_url": f"{plot_path}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/cluster_analysis', methods=['GET'])
def cluster_analysis_route():
    """Route to perform cluster analysis."""
    global clustered_data
    if clustered_data is None or 'Cluster' not in clustered_data.columns:
        return jsonify({"error": "No clustered data available. Please perform clustering first."}), 400

    try:
        analysis_results = cluster_analysis(clustered_data)
        if not analysis_results:
            return jsonify({"error": "Failed to perform cluster analysis."}), 500

        return jsonify({"message": "Cluster analysis completed.", "analysis": analysis_results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/cluster_centers', methods=['GET'])
def get_cluster_centers():
    """Retrieve the cluster centers from the K-Means model."""
    global kmeans_model
    if kmeans_model is None:
        return jsonify({"error": "No clustering model available. Please perform clustering first."}), 400

    try:
        centers = kmeans_model.cluster_centers_.tolist()
        return jsonify({"cluster_centers": centers}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Route for Dashboard (Home)
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# Route for Load & Preprocess Data
@app.route('/loadnPreprocess')
def load_data():
    return render_template('load.html')

# Route for Train Model
@app.route('/trainModel')
def train_model():
    return render_template('train.html')

# Route for Apriori Analysis
@app.route('/aprioriAnalysis')
def aprioriAnalysis():
    return render_template('apriori.html')

# Route for Cluster Analysis
@app.route('/clusterAnalysis')
def clusterAnalysis():
    return render_template('cluster.html')

if __name__ == '__main__':
    app.run(debug=True)