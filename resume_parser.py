import boto3
import os
import tarfile

# AWS S3 Configuration
aws_access_key = aws_access_key
aws_secret_key = aws_secret_key
s3_bucket_name = "resume-parser-ml-model"
s3_file_path = "lambda/sagemaker_model.tar.gz"  # Specify the desired path in S3

# Step 1: Prepare the required files
requirements_txt = """
sentence-transformers
transformers
scipy
torch
docx
boto3
"""

# Write the requirements.txt to tmp folder
with open("/tmp/requirements.txt", "w") as f:
    f.write(requirements_txt)

# Prepare the inference.py script
inference_code = """
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.spatial.distance import euclidean
from docx import Document
import boto3
from io import BytesIO
import os

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_embeddings(text):
    return model.encode(text, convert_to_tensor=True)

def calculate_context_presence_score(string_a, string_b):
    tokens_a = tokenizer.tokenize(string_a)
    tokens_b = tokenizer.tokenize(string_b)
    token_presence_ratio = sum(token in tokens_b for token in tokens_a) / len(tokens_a)
    
    embedding_a = compute_embeddings(string_a)
    embedding_b = compute_embeddings(string_b)
    cosine_scores = util.pytorch_cos_sim(embedding_a, embedding_b)

    def scale_score(score, min_val=1, max_val=10):
        if score < 0:
            return min_val
        return min_val + (max_val - min_val) / (1 + np.exp(-9 * (score - 0.3)))

    cosine_scores_np = cosine_scores.cpu().numpy()
    cosine_similarity_score = cosine_scores_np[0][0]
    scaled_similarity_score = scale_score(cosine_similarity_score)
    
    if scaled_similarity_score < 3:
        scaled_similarity_score -= (1 - token_presence_ratio)
    elif 3 <= scaled_similarity_score < 9.5:
        scaled_similarity_score += token_presence_ratio
    
    scaled_similarity_score = max(min(scaled_similarity_score, 10), 0)

    return token_presence_ratio, cosine_similarity_score, scaled_similarity_score

# Get latest resume file from S3
def get_latest_resume_from_s3(bucket, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if 'Contents' not in response:
        raise FileNotFoundError("No resumes found in S3 under the given prefix.")

    latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
    print(f"Using resume file: {latest_file['Key']}")
    obj = s3.get_object(Bucket=bucket, Key=latest_file['Key'])
    return obj['Body'].read()

# Extract text from .docx bytes
def extract_text_from_docx_bytes(docx_bytes):
    doc = Document(BytesIO(docx_bytes))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Hardcoded job description
job_description = "We are looking for a Data Engineer with skills in Python, PySpark, Spark, SQL, Hive, AWS Glue, Redshift, ETL."

# S3 configuration
s3_bucket = 'resume-parser-ml-model'
resume_folder = 'resume-upload-destination/'

# Read latest resume file
resume_bytes = get_latest_resume_from_s3(s3_bucket, resume_folder)
resume_text = extract_text_from_docx_bytes(resume_bytes)

# Calculate similarity scores
token_presence_ratio, cosine_similarity_score, scaled_similarity_score = calculate_context_presence_score(resume_text, job_description)

# Print scores
print(f"Token Presence Ratio: {token_presence_ratio}")
print(f"Cosine Similarity Score: {cosine_similarity_score}")
print(f"Scaled Similarity Score: {scaled_similarity_score}")
"""

# Save the inference script to /tmp/inference.py
with open('/tmp/inference.py', 'w') as f:
    f.write(inference_code)

# Step 2: Prepare the model directory (for example, using a pre-trained model)
from sentence_transformers import SentenceTransformer

# Initialize your model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Save the model
model_dir = '/tmp/sagemaker_model/model'
model.save(model_dir)

# Step 3: Create the tar.gz file with the model, inference.py, and requirements.txt
tar_file = '/tmp/sagemaker_model.tar.gz'

with tarfile.open(tar_file, 'w:gz') as tar:
    tar.add('/tmp/requirements.txt', arcname='requirements.txt')
    tar.add('/tmp/inference.py', arcname='inference.py')
    tar.add(model_dir, arcname='model')  # This will include your saved model

# Step 4: Upload to S3
s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key, 
                  aws_secret_access_key=aws_secret_key)

# Upload the tar.gz file to the specified S3 path
s3.upload_file(tar_file, s3_bucket_name, s3_file_path)

# Print confirmation message
print(f'File uploaded to s3://{s3_bucket_name}/{s3_file_path}')
