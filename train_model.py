import pandas as pd
import re
import nltk
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ndcg_score, average_precision_score,
    r2_score, mean_squared_error, mean_absolute_error
)


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


MODEL_PATHS = {
    'classifier': 'best_model.pkl',
    'vectorizer': 'tfidf_vectorizer.pkl',
    'linear_reg': 'linear_regression_model.pkl',
    'knn_reg': 'knn_regression_model.pkl',
    'cleaned_jobs': 'cleaned_jobs.pkl'
}

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def load_or_clean_data():
    """Load or clean job data with caching"""
    if os.path.exists(MODEL_PATHS['cleaned_jobs']):
        print("Loading cleaned jobs from cache...")
        jobs = joblib.load(MODEL_PATHS['cleaned_jobs'])
    else:
        print("Cleaning and preprocessing job data...")
        jobs = pd.read_csv('job_listings.csv')
        jobs = jobs[['Job Title', 'Job Description', 'skills']].copy()
        
        jobs['Job Title'] = jobs['Job Title'].apply(clean_text)
        jobs['Job Description'] = jobs['Job Description'].apply(clean_text)
        jobs['skills'] = jobs['skills'].apply(clean_text)
        
        jobs['combined'] = jobs['Job Title'] + ' ' + jobs['Job Description'] + ' ' + jobs['skills']
        jobs = jobs.drop_duplicates(subset='combined')
        joblib.dump(jobs, MODEL_PATHS['cleaned_jobs'])
    
    return jobs

def train_models():
    """Train and save all models"""
    print("\nTraining Machine Learning Models...")
    manual = pd.read_csv('manual_test_set.csv')
    job_lookup = jobs.set_index('Job Title')['combined'].to_dict()
    manual['job_text'] = manual['Job_Title'].map(job_lookup)
    manual.dropna(subset=['job_text'], inplace=True)

    combined_text = pd.concat([manual['Resume_Text'], manual['job_text']])
    train_vectorizer = TfidfVectorizer(max_features=3000)
    train_vectorizer.fit(combined_text)
    resume_vecs = train_vectorizer.transform(manual['Resume_Text'])
    job_vecs = train_vectorizer.transform(manual['job_text'])

    cos_sim = [cosine_similarity(resume_vecs[i], job_vecs[i])[0][0] for i in range(len(manual))]

    def skill_overlap(r, j):
        return len(set(r.split()) & set(j.split()))

    def title_score(r, t):
        return len(set(r.split()) & set(t.lower().split()))

    overlap = [skill_overlap(manual.iloc[i]['Resume_Text'], manual.iloc[i]['job_text']) for i in range(len(manual))]
    title_match = [title_score(manual.iloc[i]['Resume_Text'], manual.iloc[i]['Job_Title']) for i in range(len(manual))]

    X = pd.DataFrame({
        'tfidf_sim': cos_sim,
        'skill_overlap': overlap,
        'title_match': title_match
    })
    y = manual['Match_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }

    best_model = None
    best_f1 = 0
    metrics_list = []

    print("\nModel Evaluation Metrics:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'MAP': average_precision_score(y_test, y_proba),
            'NDCG': ndcg_score([y_test], [y_proba]) if len(set(y_test)) > 1 else 0
        }
        
        metrics_list.append(metrics)
        
        if metrics['F1'] > best_f1:
            best_model = model
            best_f1 = metrics['F1']
        
        # Print metrics
        print(f"\n{name}")
        for metric, value in metrics.items():
            if metric != 'Model':
                print(f"{metric:10}: {value:.4f}")

    # Create and display metrics comparison table
    metrics_df = pd.DataFrame(metrics_list).set_index('Model')
    print("\nModel Evaluation Summary:")
    print(metrics_df.to_string(float_format="%.3f"))

    # Regression models
    print("\nTraining Regression Models...")
    reg_data = pd.DataFrame({
        'tfidf_score': X['tfidf_sim'],
        'relevance_score': y
    })

    X_reg = reg_data[['tfidf_score']]
    y_reg = reg_data['relevance_score']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_reg, y_train_reg)
    
    # KNN Regression
    knn_reg = KNeighborsRegressor(n_neighbors=3)
    knn_reg.fit(X_train_reg, y_train_reg)

    # Save all models
    joblib.dump(best_model, MODEL_PATHS['classifier'])
    joblib.dump(train_vectorizer, MODEL_PATHS['vectorizer'])
    joblib.dump(lr_model, MODEL_PATHS['linear_reg'])
    joblib.dump(knn_reg, MODEL_PATHS['knn_reg'])
    
    print("\nModels trained and saved successfully!")
    
    # Return models for immediate use
    return {
        'classifier': best_model,
        'vectorizer': train_vectorizer,
        'linear_reg': lr_model,
        'knn_reg': knn_reg
    }

def load_models():
    """Load pre-trained models"""
    print("Loading pre-trained models...")
    return {
        'classifier': joblib.load(MODEL_PATHS['classifier']),
        'vectorizer': joblib.load(MODEL_PATHS['vectorizer']),
        'linear_reg': joblib.load(MODEL_PATHS['linear_reg']),
        'knn_reg': joblib.load(MODEL_PATHS['knn_reg'])
    }

def get_top_missing_skills(resume_text, vectorizer, job_vectors, jobs, top_n=3):
    """Get top missing skills from top matching jobs"""
    resume_vector = vectorizer.transform([resume_text])
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        job_title = jobs.iloc[idx]['Job Title']
        job_skills = set(jobs.iloc[idx]['skills'].split())
        resume_skills = set(resume_text.split())
        missing_skills = list(job_skills - resume_skills)[:15]  # Top 15 missing skills
        
        results.append({
            'Job Title': job_title,
            'Similarity Score': similarities[idx],
            'Missing Skills': missing_skills
        })
    
    return results

# Main execution
if __name__ == "__main__":
    # Initialize text processing
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Load or clean job data
    jobs = load_or_clean_data()
    print(f"Unique jobs: {len(jobs)}")
    
    # Load resumes
    print("Loading and cleaning resumes...")
    resumes = pd.read_csv('resumes.csv')
    resumes = resumes[['Resume']].copy()
    resumes['Resume'] = resumes['Resume'].apply(clean_text)
    resumes['combined'] = resumes['Resume']
    
    # TF-IDF Matching (always runs)
    print("\nRunning TF-IDF Matching for All Resumes...")
    vectorizer = TfidfVectorizer(max_features=5000)
    job_vectors = vectorizer.fit_transform(jobs['combined'])
    
    all_matches = []
    for i, resume in resumes.iterrows():
        resume_vector = vectorizer.transform([resume['combined']])
        similarities = cosine_similarity(resume_vector, job_vectors).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        
        resume_matches = [{
            'Job Title': jobs.iloc[idx]['Job Title'],
            'Score': similarities[idx]
        } for idx in top_indices]
        
        all_matches.append({
            'Resume ID': i,
            'Top Matches': resume_matches
        })
    
    print("\nTF-IDF Matching Results:")
    for resume in all_matches:
        print(f"\nResume {resume['Resume ID']} - Top 5 Matches:")
        for match in resume['Top Matches']:
            print(f"{match['Job Title']}  |  Score: {match['Score']:.4f}")
    
    # Model handling
    models_exist = all(os.path.exists(path) for path in MODEL_PATHS.values())
    
    if models_exist:
        models = load_models()
    else:
        models = train_models()
    
    # Skill Gap Analysis
    print("\nRunning Skill Gap Analysis...")
    vectorizer = models['vectorizer']
    job_vectors = vectorizer.transform(jobs['combined'])
    
    # Analyze first 3 resumes
    for resume_idx in range(min(3, len(resumes))):
        print(f"\n{'='*50}")
        print(f"Skill Gap Analysis for Resume {resume_idx}")
        print(f"{'='*50}")
        
        resume_text = resumes.iloc[resume_idx]['combined']
        
        # Get top 3 job matches with missing skills
        top_jobs_with_missing = get_top_missing_skills(resume_text, vectorizer, job_vectors, jobs, top_n=3)
        
        for i, job_match in enumerate(top_jobs_with_missing, 1):
            print(f"\nTop Match #{i}: {job_match['Job Title']} (Score: {job_match['Similarity Score']:.4f})")
            print(f"Missing Skills ({len(job_match['Missing Skills'])}):")
            print(', '.join(job_match['Missing Skills']) if job_match['Missing Skills'] else "No missing skills")
    
    print("\nAnalysis complete!")
