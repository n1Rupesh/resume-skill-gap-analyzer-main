#!/usr/bin/env python3
"""
Pickle File Converter - Fix Compatibility Issues
Run this script to convert your old pickle files to compatible formats.
"""

import pickle
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def convert_pickle_files():
    """Convert pickle files to more compatible formats"""
    
    files_to_convert = ['best_model.pkl', 'tfidf_vectorizer.pkl']
    
    for filename in files_to_convert:
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
        
        print(f"🔄 Converting {filename}...")
        
        # Try different loading methods
        obj = None
        
        # Method 1: Standard pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            print(f"✅ Loaded {filename} with standard pickle")
        except Exception as e:
            print(f"⚠ Standard pickle failed: {e}")
        
        # Method 2: Pickle with latin1 encoding
        if obj is None:
            try:
                with open(filename, 'rb') as f:
                    obj = pickle.load(f, encoding='latin1')
                print(f"✅ Loaded {filename} with latin1 encoding")
            except Exception as e:
                print(f"⚠ Latin1 pickle failed: {e}")
        
        # Method 3: Try joblib
        if obj is None:
            try:
                obj = joblib.load(filename)
                print(f"✅ Loaded {filename} with joblib")
            except Exception as e:
                print(f"⚠ Joblib failed: {e}")
        
        # Save in multiple formats if successfully loaded
        if obj is not None:
            base_name = filename.replace('.pkl', '')
            
            # Save with joblib (recommended)
            try:
                joblib.dump(obj, f'{base_name}_joblib.pkl')
                print(f"✅ Saved {base_name}_joblib.pkl")
            except Exception as e:
                print(f"❌ Failed to save with joblib: {e}")
            
            # Save with current pickle protocol
            try:
                with open(f'{base_name}_new.pkl', 'wb') as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"✅ Saved {base_name}_new.pkl")
            except Exception as e:
                print(f"❌ Failed to save with pickle: {e}")
            
            # Save with compatible protocol
            try:
                with open(f'{base_name}_compatible.pkl', 'wb') as f:
                    pickle.dump(obj, f, protocol=2)
                print(f"✅ Saved {base_name}_compatible.pkl")
            except Exception as e:
                print(f"❌ Failed to save compatible pickle: {e}")
        else:
            print(f"❌ Could not load {filename} with any method")

def test_converted_files():
    """Test if the converted files can be loaded"""
    print("\n🧪 Testing converted files...")
    
    test_files = [
        'best_model_joblib.pkl',
        'tfidf_vectorizer_joblib.pkl',
        'best_model_new.pkl',
        'tfidf_vectorizer_new.pkl'
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            try:
                if 'joblib' in filename:
                    obj = joblib.load(filename)
                else:
                    with open(filename, 'rb') as f:
                        obj = pickle.load(f)
                print(f"✅ {filename} loads successfully")
                print(f"   Type: {type(obj)._name_}")
                
                # Basic validation
                if hasattr(obj, 'predict'):
                    print(f"   Has predict method: ✅")
                if hasattr(obj, 'transform'):
                    print(f"   Has transform method: ✅")
                    
            except Exception as e:
                print(f"❌ {filename} failed to load: {e}")
        else:
            print(f"⚠ {filename} not found")

def create_sample_models():
    """Create sample models if originals can't be loaded"""
    print("\n🔧 Creating sample models for testing...")
    
    # Create a sample TF-IDF vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    sample_texts = [
        "python machine learning data science",
        "javascript web development frontend",
        "sql database management systems",
        "artificial intelligence deep learning"
    ]
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer.fit(sample_texts)
    
    # Create a sample logistic regression model
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    X_sample = vectorizer.transform(sample_texts)
    y_sample = np.array([1, 0, 1, 1])  # Sample labels
    
    model = LogisticRegression(random_state=42)
    model.fit(X_sample, y_sample)
    
    # Save sample models
    joblib.dump(model, 'sample_model.pkl')
    joblib.dump(vectorizer, 'sample_vectorizer.pkl')
    
    print("✅ Created sample_model.pkl and sample_vectorizer.pkl")
    print("   You can use these for testing the Streamlit app")

def main():
    print("🔧 Pickle File Converter for Job Matching App")
    print("=" * 50)
    
    # Check if original files exist
    required_files = ['best_model.pkl', 'tfidf_vectorizer.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠ Missing files: {missing_files}")
        print("\nOptions:")
        print("1. Place the missing files in this directory")
        print("2. Create sample models for testing")
        
        choice = input("\nCreate sample models? (y/n): ").lower().strip()
        if choice == 'y':
            create_sample_models()
        return
    
    # Convert existing files
    convert_pickle_files()
    
    # Test converted files
    test_converted_files()
    
    print("\n📋 Next Steps:")
    print("1. Update your Streamlit app to use the converted files:")
    print("   - Replace 'best_model.pkl' with 'best_model_joblib.pkl'")
    print("   - Replace 'tfidf_vectorizer.pkl' with 'tfidf_vectorizer_joblib.pkl'")
    print("2. Or rename the converted files to replace the originals")
    print("3. Test your Streamlit app")

if __name__ == "__main__":
    main()