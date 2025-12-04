import numpy as np
import hdbscan
import json
import logging
import DB
import warnings
from dotenv import load_dotenv
import os

# per la versione senza pca (che si presuppone sia stata già calcolata):
# toglirrr tutti i '_256' e modificare reduce_embedding con embedding

warnings.filterwarnings("ignore")
 
TABLE_NAME = 'embeddings_recursive_cluster_pca'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_cluster_column_if_not_exists(cursor, table_name):
    query = f"""
    ALTER TABLE {table_name}
    ADD COLUMN IF NOT EXISTS cluster_id_256 INTEGER DEFAULT -1;
    """
    cursor.execute(query)

def add_cluster_embedding_column_if_not_exists(cursor, table_name):
    query = f"""
    ALTER TABLE {table_name}
    ADD COLUMN IF NOT EXISTS cluster_embedding_256 VECTOR(256);
    """
    cursor.execute(query)

def compute_and_update_cluster_embeddings(cursor, conn, table_name):
    cursor.execute(f"SELECT cluster_id_256, reduced_embedding_256 FROM {table_name} WHERE cluster_id_256 >= 0")
    rows = cursor.fetchall()

    cluster_embeddings_map = {}
    for cluster_id, embedding_raw in rows:
        if isinstance(embedding_raw, str):
            embedding_vector = np.array(json.loads(embedding_raw), dtype=np.float32)
        elif isinstance(embedding_raw, (bytes, bytearray)):
            embedding_vector = np.frombuffer(embedding_raw, dtype=np.float32)
        else:
            embedding_vector = np.array(embedding_raw, dtype=np.float32)

        if embedding_vector.shape[0] != 256:
            raise ValueError(f"Embedding size mismatch: expected 256, got {embedding_vector.shape[0]}")

        cluster_embeddings_map.setdefault(cluster_id, []).append(embedding_vector)

    for cluster_id, embeddings_list in cluster_embeddings_map.items():
        stacked = np.vstack(embeddings_list)
        centroid = np.mean(stacked, axis=0)

        cursor.execute(
            f"UPDATE {table_name} SET cluster_embedding_256 = %s WHERE cluster_id_256 = %s",
            (centroid.tolist(), cluster_id)
        )
    conn.commit()

def fetch_embeddings(cursor, table_name):
    cursor.execute(f"SELECT id, reduced_embedding_256 FROM {table_name}")
    rows = cursor.fetchall()
    ids = []
    embeddings = []

    for _id, embedding_raw in rows:
        ids.append(_id)

        if isinstance(embedding_raw, str):
            embedding_vector = np.array(json.loads(embedding_raw), dtype=np.float32)
        elif isinstance(embedding_raw, (bytes, bytearray)):
            embedding_vector = np.frombuffer(embedding_raw, dtype=np.float32)
        else:
            embedding_vector = np.array(embedding_raw, dtype=np.float32)

        if embedding_vector.shape[0] != 256:
            raise ValueError(f"Embedding size mismatch for id={_id}: expected 256, got {embedding_vector.shape[0]}")

        embeddings.append(embedding_vector)

    return ids, np.vstack(embeddings)

def cluster_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels

def update_cluster_ids(cursor, conn, table_name, ids, labels):
    # Prepara i valori da aggiornare
    values = ', '.join(
        cursor.mogrify("(%s, %s)", (id_, int(label))).decode("utf-8")
        for id_, label in zip(ids, labels)
    )

    query = f"""
    UPDATE {table_name} AS t SET cluster_id_256 = v.cluster_id_256
    FROM (VALUES {values}) AS v(id, cluster_id_256)
    WHERE t.id = v.id;
    """
    cursor.execute(query)
    conn.commit()


def main_clustering(cursor, conn, table_name):
    add_cluster_column_if_not_exists(cursor, table_name)
    add_cluster_embedding_column_if_not_exists(cursor, table_name)
    conn.commit()

    ids, embeddings = fetch_embeddings(cursor, table_name)

    logger.info(f"Numero embeddings: {len(embeddings)}")
    logger.info(f"Dimensione primo embedding: {embeddings[0].shape}")
    logger.info(f"Shape totale matrice embeddings: {embeddings.shape}")

    labels = cluster_embeddings(embeddings)

    logger.info(f"Cluster labels: {labels}")
    logger.info(f"Numero cluster trovati: {len(set(labels)) - (1 if -1 in labels else 0)}")
    logger.info(f"Numero rumore (-1): {(labels == -1).sum()}")

    if set(labels) == {-1}:
        logger.warning("Tutti gli embeddings sono stati classificati come rumore. Nessun cluster valido.")
        return

    update_cluster_ids(cursor, conn, table_name, ids, labels)
    compute_and_update_cluster_embeddings(cursor, conn, table_name)

if __name__ == "__main__":
    #parametri  di connessione al DB
    HOST_NAME = os.getenv("HOST_NAME")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    USER_NAME = os.getenv("USER_NAME")
    PASSWORD = os.getenv("PASSWORD")
    PORT = os.getenv("PORT")

    # Load database connection
    cursor, conn = DB.connect_db(
        host = HOST_NAME,
        database = DATABASE_NAME,
        user = USER_NAME,
        password = PASSWORD,
        port = PORT
    )
    main_clustering(cursor, conn, table_name=TABLE_NAME)
