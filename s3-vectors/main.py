import uuid
import json
import boto3
from icecream import ic
from botocore.exceptions import ClientError

VECTOR_BUCKET_NAME = "my-s3-vector-bucket-exp"
INDEX_NAME = "test-index-4"
VECTOR_DIMENSION = 1024
VECTOR_MODEL_ID = "amazon.titan-embed-text-v2:0"
TEXT_GEN_MODEL_ID = "amazon.titan-text-express-v1"

s3_vector = boto3.client("s3vectors")
bedrock_client = boto3.client("bedrock-runtime")


def list_vector_buckets():
    response = s3_vector.list_vector_buckets()
    buckets = [bucket["vectorBucketName"] for bucket in response["vectorBuckets"]]
    ic(buckets)

def get_index(index_name):
    try:
        s3_vector.get_index(
            indexName=index_name,
            vectorBucketName=VECTOR_BUCKET_NAME
        )
        return True
    except ClientError as exc:
        if "NotFoundException" in str(exc):
            ic("Index does not exist:", exc)
            return False
        else:
            raise

def create_vector_index(index_name=INDEX_NAME):
    if not get_index(index_name):
        s3_vector.create_index(
            dataType="float32",
            indexName=index_name,
            dimension=VECTOR_DIMENSION,
            distanceMetric="cosine",
            vectorBucketName=VECTOR_BUCKET_NAME
        )
        ic("✅ Index created successfully.")
    else:
        ic("ℹ️ Index already exists.")

def generate_embedding(text: str):
    response = bedrock_client.invoke_model(
        modelId=VECTOR_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

def insert_into_vector(vector_bucket_name, index_name):
    texts = [
        "Star Wars: A farm boy joins rebels to fight an evil empire in space.",
        "Jurassic Park: Scientists create dinosaurs in a theme park that goes wrong.",
        "Finding Nemo: A father fish searches the ocean to find his lost son.",
        "Interstellar: A team of explorers travels through a wormhole in space to save humanity.",
        "The Martian: An astronaut becomes stranded on Mars and must figure out how to survive."
    ]

    vectors = []
    for text in texts:
        embedding = generate_embedding(text)
        vectors.append({
            "key": str(uuid.uuid4()),
            "data": {"float32": embedding},
            "metadata": {
                "description": text,
                "genre": "scifi"
            }
        })

    s3_vector.put_vectors(
        vectorBucketName=vector_bucket_name,
        indexName=index_name,
        vectors=vectors
    )
    ic("✅ Inserted vectors into store.")

def query_vector_store(query_text, top_k=2, genre_filter="scifi"):
    embedding = generate_embedding(query_text)
    response = s3_vector.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=INDEX_NAME,
        queryVector={"float32": embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    return response["vectors"]

def search_vector_store():
    input_text = "Give me a  movie in which the father fish searches the ocean to find his lost son."
    vectors = query_vector_store(input_text)
    ic(vectors)

if __name__ == "__main__":
    list_vector_buckets()
    create_vector_index(INDEX_NAME)
    insert_into_vector(VECTOR_BUCKET_NAME, INDEX_NAME)
    search_vector_store()
