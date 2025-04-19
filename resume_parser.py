import json
import boto3
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer
import numpy as np
from docx import Document
import sys

# Initialize models and S3 client
s3 = boto3.client('s3')
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

    cosine_similarity_score = float(cosine_scores[0][0])
    scaled_similarity_score = scale_score(cosine_similarity_score)

    if scaled_similarity_score < 3:
        scaled_similarity_score -= (1 - token_presence_ratio)
    elif 3 <= scaled_similarity_score < 9.5:
        scaled_similarity_score += token_presence_ratio

    scaled_similarity_score = max(min(scaled_similarity_score, 10), 0)

    return token_presence_ratio, cosine_similarity_score, scaled_similarity_score

def process_resume_and_job_description(resume_text, job_description):
    token_presence_score, cosine_similarity_score, custom_model_score = calculate_context_presence_score(resume_text, job_description)
   
    return {
        "Token Presence Score": float(token_presence_score),
        "Cosine Similarity Score": float(cosine_similarity_score),
        "Custom Model Score": float(custom_model_score)
    }

def get_latest_docx_from_bucket(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name)
    docx_files = [
        obj for obj in response.get('Contents', [])
        if obj['Key'].endswith('.docx')
    ]
   
    if not docx_files:
        raise Exception("No .docx files found in the bucket.")

    # Sort by LastModified descending to get latest one
    latest_file = sorted(docx_files, key=lambda x: x['LastModified'], reverse=True)[0]
    return latest_file['Key']

def main(bucket, key):
    print("Running inference for:")
    print("Bucket:", bucket)
    print("Key:", key)

    response = s3.get_object(Bucket=bucket, Key=key)
    file_stream = response['Body'].read()
    doc = Document(BytesIO(file_stream))
    resume_text = "\n".join([para.text for para in doc.paragraphs])

    job_description = "Looking for a Big data Engineer having experience in pyspark,hadoop,sql,hive,ETL,hadoop,Aws,AwsGlue,Redshift"

    result = process_resume_and_job_description(resume_text, job_description)

    output_key = key.replace(".docx", "_output.json")
    s3.put_object(Bucket=bucket, Key=output_key, Body=json.dumps(result))

    print("Inference result written to:", output_key)
    print(result)


#bucket = 'resume-upload-destination'
#key = 'updated_resume.docx'

main(bucket, key)

bucket = 'resume-upload-destination'
key = get_latest_docx_from_bucket(bucket)
print("Latest uploaded file picked:", key)
main(bucket, key)