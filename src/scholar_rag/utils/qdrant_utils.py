import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorComparator, MultiVectorConfig

def get_client(path): 
    """
    Creates a qdrant client.
    """
    client = qdrant_client.QdrantClient(path=path) 
    return client

def create_collection(client, collection_name, embedding_size): 
    """
    Creates a qdrant database. 

    Args: 
        client: The qdrant client used to create the collection
        collection_name: The name of the qdrant collection.
        embedding_size: The size of the vector embeddings (axis=-1) stored.
    """
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection( 
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_size, 
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        )
    )

def query(client, query, collection_name): 
    """
    Queries the collection and returns a result. 

    Args: 
        client: The qdrant client object. 
        query: The embedded query.
    """
    result = client.query_points( 
        collection_name = collection_name,
        query = query,
        limit = 3
    )

    for i, point in enumerate(result.points, 1): 
        print(f"{i}, {point.payload['document']}")
        print(f"{i}, {point.payload['page_number']}")
        print(f"Score: {point.score:.3f}")

    return result 