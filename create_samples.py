#!/usr/bin/env python3
"""
Sample Data Creator for Job Matching App
Run this script to create sample files for testing the application.
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

def create_sample_job_dataset():
    """Create a sample job dataset"""
    print("📊 Creating sample job dataset...")
    
    sample_jobs = pd.DataFrame({
        'title': [
            'Senior Data Scientist',
            'Machine Learning Engineer',
            'Python Developer',
            'Software Engineer',
            'Data Analyst',
            'Backend Developer',
            'AI Research Scientist',
            'Full Stack Developer',
            'DevOps Engineer',
            'Product Manager',
            'Frontend Developer',
            'Business Intelligence Analyst',
            'Cloud Solutions Architect',
            'Database Administrator',
            'Cybersecurity Analyst',
            'Mobile App Developer',
            'UI/UX Designer',
            'Quality Assurance Engineer',
            'Technical Writer',
            'Systems Administrator'
        ],
        'company': [
            'TechCorp Inc.',
            'AI Solutions Ltd.',
            'DataFlow Systems',
            'InnovateTech',
            'Analytics Pro',
            'CodeBase Solutions',
            'Research Labs',
            'WebDev Corp',
            'CloudOps Inc.',
            'Product Innovations',
            'Frontend Masters',
            'Business Intel Co.',
            'Cloud Architects Ltd.',
            'DataBase Pro',
            'SecureNet Inc.',
            'MobileTech',
            'Design Studio',
            'QualityFirst',
            'DocuTech',
            'SysAdmin Solutions'
        ],
        'description': [
            'Senior Data Scientist position requiring Python, machine learning, pandas, scikit-learn, TensorFlow, statistical analysis, and data visualization experience. Must have PhD or Masters in Data Science.',
            'Machine Learning Engineer role focusing on deep learning, neural networks, TensorFlow, PyTorch, computer vision, NLP, and model deployment. Experience with MLOps preferred.',
            'Python Developer position for backend development using Django, Flask, FastAPI, PostgreSQL, Redis, and REST API development. Strong OOP and testing skills required.',
            'Software Engineer role requiring Java, Spring Boot, microservices, Docker, Kubernetes, AWS, and agile development practices. Full-stack experience a plus.',
            'Data Analyst position requiring SQL, Excel, Tableau, Power BI, statistical analysis, and business intelligence. Experience with Python and R preferred.',
            'Backend Developer role using Python, Node.js, Express, MongoDB, PostgreSQL, Redis, and API development. Microservices architecture experience required.',
            'AI Research Scientist position for cutting-edge machine learning research, deep learning, computer vision, NLP, reinforcement learning, and academic publication experience.',
            'Full Stack Developer role with React, Angular, Node.js, Express, MongoDB, PostgreSQL, and modern JavaScript frameworks. DevOps experience a plus.',
            'DevOps Engineer position requiring Docker, Kubernetes, AWS, Azure, CI/CD pipelines, Infrastructure as Code, monitoring, and automation tools.',
            'Technical Product Manager role for data products requiring understanding of machine learning, data science, agile methodologies, and stakeholder management.',
            'Frontend Developer position with React, Vue.js, Angular, JavaScript, TypeScript, HTML, CSS, and modern web development practices. UX/UI collaboration required.',
            'Business Intelligence Analyst role requiring SQL, Tableau, Power BI, data warehousing, ETL processes, and business analysis skills.',
            'Cloud Solutions Architect position requiring AWS, Azure, GCP, serverless computing, microservices, and enterprise architecture design experience.',
            'Database Administrator role requiring MySQL, PostgreSQL, Oracle, database optimization, backup strategies, and performance tuning expertise.',
            'Cybersecurity Analyst position requiring network security, penetration testing, SIEM tools, incident response, and security compliance knowledge.',
            'Mobile App Developer role requiring React Native, Flutter, iOS, Android, mobile UI/UX, and app store deployment experience.',
            'UI/UX Designer position requiring Figma, Sketch, Adobe Creative Suite, user research, prototyping, and design systems experience.',
            'Quality Assurance Engineer role requiring test automation, Selenium, API testing, performance testing, and agile testing methodologies.',
            'Technical Writer position requiring documentation, API documentation, user guides, technical communication, and collaboration with engineering teams.',
            'Systems Administrator role requiring Linux, Windows Server, networking, virtualization, monitoring tools, and infrastructure management.'
        ]
    })
    
    # Save the dataset
    sample_jobs.to_csv('cleaned_jobs_deduped.csv', index=False)
    print(f"✅ Created cleaned_jobs_deduped.csv with {len(sample_jobs)} jobs")
    
    return sample_jobs

def create_sample_models(jobs_df):
    """Create sample ML models"""
    print("🤖 Creating sample ML models...")
    
    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    # Fit vectorizer on job descriptions
    X = vectorizer.fit_transform(jobs_df['description'])
    
    # Create sample labels (simulate job-resume compatibility)
    np.random.seed(42)  # For reproducible results
    y = np.random.choice([0, 1], size=len(jobs_df), p=[0.3, 0.7])  # 70% positive matches
    
    # Create and train Logistic Regression model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X, y)
    
    # Save models using joblib (more compatible)
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    print("✅ Created best_model.pkl (Logistic Regression)")
    print("✅ Created tfidf_vectorizer.pkl (TF-IDF Vectorizer)")
    
    return model, vectorizer

def create_sample_resume():
    """Create a sample resume for testing"""
    print("📄 Creating sample resume...")
    
    sample_resume = """
John Smith
Senior Data Scientist | Machine Learning Engineer

Email: john.smith@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johnsmith

PROFESSIONAL SUMMARY
Experienced Data Scientist with 5+ years in machine learning, statistical analysis, and data-driven solutions. 
Proven track record in developing predictive models, implementing ML pipelines, and delivering business insights.

TECHNICAL SKILLS
• Programming Languages: Python, R, SQL, JavaScript, Java
• Machine Learning: Scikit-learn, TensorFlow, PyTorch, Keras, XGBoost
• Data Analysis: Pandas, NumPy, Matplotlib, Seaborn, Plotly
• Databases: PostgreSQL, MySQL, MongoDB, Redis
• Cloud Platforms: AWS (EC2, S3, SageMaker), Google Cloud Platform
• Tools: Git, Docker, Jupyter, VS Code, Tableau, Power BI
• Web Development: Flask, Django, FastAPI, React, HTML, CSS

PROFESSIONAL EXPERIENCE

Senior Data Scientist | TechCorp Inc. | 2021 - Present
• Developed machine learning models for customer segmentation using clustering algorithms
• Built recommendation systems using collaborative filtering and deep learning
• Implemented MLOps pipelines for model deployment and monitoring
• Led data science team of 4 members on predictive analytics projects
• Increased model accuracy by 25% through feature engineering and hyperparameter tuning

Machine Learning Engineer | DataFlow Systems | 2019 - 2021
• Designed and deployed real-time ML models for fraud detection
• Optimized data processing pipelines using Apache Spark and Kafka
• Developed REST APIs for model serving using Flask and Docker
• Collaborated with software engineers on production ML system architecture
• Reduced model inference time by 40% through optimization techniques

Data Analyst | Analytics Pro | 2018 - 2019
• Performed statistical analysis and created business intelligence dashboards
• Automated reporting processes using Python and SQL
• Conducted A/B testing for product feature optimization
• Created data visualizations using Tableau and Power BI
• Supported business decision-making with data-driven insights

EDUCATION
Master of Science in Data Science | University of Technology | 2018
Bachelor of Science in Computer Science | State University | 2016

PROJECTS
• Customer Churn Prediction: Built ML model predicting customer churn with 92% accuracy
• Stock Price Forecasting: Developed LSTM-based time series model for financial predictions  
• Natural Language Processing: Created sentiment analysis system for social media data
• Computer Vision: Implemented image classification model using convolutional neural networks

CERTIFICATIONS
• AWS Certified Machine Learning - Specialty
• Google Professional Data Engineer
• TensorFlow Developer Certificate
• Certified Analytics Professional (CAP)

PUBLICATIONS
• "Advanced Machine Learning Techniques for Business Intelligence" - Data Science Journal, 2022
• "Scalable ML Pipelines in Production Environments" - AI Conference Proceedings, 2021
"""
    
    with open('sample_resume.txt', 'w', encoding='utf-8') as f:
        f.write(sample_resume.strip())
    
    print("✅ Created sample_resume.txt")
    
    return sample_resume

def create_requirements_file():
    """Create requirements.txt file"""
    print("📦 Creating requirements.txt...")
    
    requirements = """
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
nltk>=3.8
plotly>=5.15.0
reportlab>=4.0.0
joblib>=1.3.0
numpy>=1.24.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("✅ Created requirements.txt")

def test_created_files():
    """Test if all created files work correctly"""
    print("\n🧪 Testing created files...")
    
    # Test CSV loading
    try:
        jobs_df = pd.read_csv('cleaned_jobs_deduped.csv')
        print(f"✅ CSV loaded successfully: {len(jobs_df)} rows")
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return False
    
    # Test model loading
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test sample prediction
    try:
        sample_text = ["python machine learning data science experience"]
        X_test = vectorizer.transform(sample_text)
        prediction = model.predict_proba(X_test)
        print(f"✅ Sample prediction works: {prediction[0]}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test resume file
    try:
        with open('sample_resume.txt', 'r', encoding='utf-8') as f:
            resume_content = f.read()
        print(f"✅ Resume file loaded: {len(resume_content)} characters")
    except Exception as e:
        print(f"❌ Resume file loading failed: {e}")
        return False
    
    return True

def main():
    print("🚀 Sample Data Creator for Job Matching App")
    print("=" * 50)
    
    # Check if files already exist
    existing_files = []
    files_to_check = [
        'cleaned_jobs_deduped.csv',
        'best_model.pkl',
        'tfidf_vectorizer.pkl',
        'sample_resume.txt',
        'requirements.txt'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            existing_files.append(file)
    
    if existing_files:
        print(f"⚠ Existing files found: {existing_files}")
        overwrite = input("Overwrite existing files? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Aborted. Remove existing files manually if you want to recreate them.")
            return
    
    # Create all sample files
    print("\n📁 Creating sample files...")
    
    # 1. Create job dataset
    jobs_df = create_sample_job_dataset()
    
    # 2. Create ML models
    model, vectorizer = create_sample_models(jobs_df)
    
    # 3. Create sample resume
    sample_resume = create_sample_resume()
    
    # 4. Create requirements file
    create_requirements_file()
    
    # 5. Test everything
    if test_created_files():
        print("\n🎉 SUCCESS! All sample files created and tested.")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the Streamlit app: streamlit run app.py")
        print("3. Test with the sample resume: sample_resume.txt")
        print("\n📊 Files created:")
        for file in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   • {file} ({size:,} bytes)")
    else:
        print("\n❌ Some files failed testing. Check the errors above.")

if __name__ == "__main__":
    main()