#LIBRARIES
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re
import csv


#LOAD PICKLE FILES
model = pickle.load(open('data and pickle files/best_model.pkl','rb')) 
vectorizer = pickle.load(open('data and pickle files/count_vectorizer.pkl','rb')) 

#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

#TEXT CLASSIFICATION
def text_classification(text):
    if len(text) < 1:
        st.write("  ")
        p="False"
        return p
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)
            
            return p

def app_review(text):
    if len(text) < 1:
        print("")
        #st.write("  ")
    else:
        legit =  0
        fake = 0
        app_name = text+".csv"
        with st.spinner("Classification in progress..."):
            try:
                with open(app_name) as file_obj:
          
                    # Create reader object by passing the file 
                    # object to reader method
                    reader_obj = csv.reader(file_obj)
          
                # Iterate over each row in the csv 
                # file using reader object
                    for row in reader_obj:
                        print(row[0])
                        val = row[0]
                        p=text_classification(val)
                        print(p)
                        if p == 'True':
                            legit+=1
                        else:
                            fake+=1
                    if(legit>fake):
                        print(legit)
                        print(fake)
                        print("legit")
                        st.success("The app is Legitimate.")
                    else:
                        print("fake")
                        st.error("The app is Fraudulent.")
            except OSError as e:
                st.error("App not found")
                    
    

#PAGE FORMATTING AND APPLICATION
def main():
    st.title("Fraud App Detection using Online Consumer Reviews ")
    
    
    # --EXPANDERS--    
    abstract = st.expander("Abstract")
    if abstract:
        abstract.write("Sample Abstract.")
        #st.write(abstract)
    
    links = st.expander("Related Links")
    if links:
        links.write("[Dataset utilized](https://www.kaggle.com/akudnaver/amazon-reviews-dataset)")
        links.write("[Github]")
        
    # --CHECKBOXES--
    st.subheader("Information on the Classifier")
    if st.checkbox("About Classifer"):
        st.markdown('**Model:** Logistic Regression')
        st.markdown('**Vectorizer:** Count')
        st.markdown('**Test-Train splitting:** 40% - 60%')
        st.markdown('**Spelling Correction Library:** TextBlob')
        st.markdown('**Stemmer:** PorterStemmer')
        
    if st.checkbox("Evaluation Results"):
        st.markdown('**Accuracy:** 85%')
        st.markdown('**Precision:** 80%')
        st.markdown('**Recall:** 92%')
        st.markdown('**F-1 Score:** 85%')


    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Fake Review Classifier")
    review = st.text_input("Enter App Name: ")
    review=review.strip()
    if st.button("Check"):
        app_review(review)
    #print(app_review(review))
#RUN MAIN        
main()
