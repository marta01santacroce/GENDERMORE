[Paper Accepted at IRCDL 2026](https://ircdl2026.unimore.it/)

This work introduces MAGDA (Multi-document Aggregation via Global Document-level clustering Architecture),
a domain-specific RAG system that uses a clustering-based chunking and retrieval strategy to capture semantic
relations across documents in the Gender*More corpus. By aggregating inter-document information, MAGDA
improves the grounding and relevance of generated answers. We evaluate the system on a curated set of real
queries and validated responses, showing that domain-adapted RAG pipelines combined with cross-document
processing significantly enhance information access and explainability in specialized scientific digital libraries.

## 🗂️ Main files in the repository

- `chatbot_gradio.py`  
  Code for the chatbot interface developed with Gradio and based on OpenAI's LLM (GPT-4o-mini).

- `save_embeddings_character.py`  
  Saves documents into the PGVector database using a character-based splitting method.

- `save_embeddings_semantic.py`  
  Saves documents into the PGVector database using a semantic splitter.

- `save_embeddings_recursive.py`  
  Saves documents into the PGVector database using a recursive character-based splitter.
  
- `create_cluster.py`  
  Creation of inter-document clusters using the HDBSCAN algorithm.
  
- `DB.py`  
  Handles all interactions with the vector database.

- `DB_for_cluster.py`  
  Handles all interactions with the vector database with clustering enabled.

- `search_with_cluster.py`  
  Retrieval of relevant documents through semantic search on the combined_embedding in inter-document clustering and reranking.

- `search_v2.py`  
  Retrieval of relevant documents through hybrid search and reranking.

- `evaluation.py`  
  Evaluation of the RAG system using metrics such as BERTScore, BLEU, ROUGE, Recall@K, and MRR.

- `load_dataset.py`  
  Loads the dataset containing question/answer pairs.

- `update_name_pdf_and_move_folder.py`  
  Handles new PDF files: formats filenames as required by Gradio and copies them into the `pdf_files` folder.

---
## 🛢️ DataBase 
To populate the database with chunks obtained through recursive splitting:
` python save_embeddings_recursive.py path_name Y`\
NOTES:
- Parameters to pass during execution:

    a) the path of the folder containing the PDF files to store OR the path of the single PDF file to store

    b) 'Y' if you want to initialize the DB OR 'N' if you only want to add new files without initializing

- The table created/updated is named ```python TABLE_NAME = 'embeddings_recursive'``` inside the Python file. If you want to change the table name, modify the code.

- The table is created using the following function:
  ```python
  def create_table(cursor,table_name):
      cursor.execute(f"""CREATE TABLE {table_name} (id SERIAL PRIMARY KEY,content TEXT,
                                                    embedding VECTOR(1024), source TEXT,
                                                    page_number INT, language TEXT,
                                                    tsv_content tsvector DEFAULT NULL, hash_value TEXT)
                      """
                   )


## 🚀 Starting the system
To start the chatbot, run the file:

```bash
python chatbot_gradio.py
```
This script uses the following fundamental modules:
- `DB.py`
  for all operations with the vector database (PGVector).
- `save_embeddings_recursive.py`
  to save documents in the knowledge base using recursive subdivision.
- `update_name_pdf_and_move_folder.py`
  to format PDF file names and copy them correctly to the dedicated folder.

#### ▶️ How it works

1) Chatbot screen:
   - enter your query in the space below and press the button on the right to send your question
   - wait for the system to process the question to read the answer in the central screen
   - click on the links in the output to open the documents from which the answer was extracted (in some cases, it is not possible to highlight the precise extract consulted from the PDF due to non-linear formatting of the document)
   - To start a new chat, click on the button at the top left
   - To scroll from one conversation to another, on the left side of the screen, you will find a list of active chats
2) File upload screen:
   - Click on the first box to upload PDFs (we strongly recommend a maximum of 5 PDFs at a time)
   - Click on the ‘submit’ button to start the process
     ⚠️**OSS**: once you have clicked the upload button, you cannot interrupt the process. Be careful what you upload!
   - Wait for the upload to complete and view the upload output in the second block. Please be patient... the execution time varies depending on the size of the documents to be saved.
   - The ‘Clear All’ button only clears the screen **visually**. It **DOES NOT** interrupt the upload but only deletes the text on the screen.
   - While waiting for the upload, you can return to the chatbot screen and continue the conversation.
 3) In this first version, do not consider any additional buttons/actions on the screen. 


## 📂 pdf_files folder

Make sure there is a folder called pdf_files/ in the main directory of the project.

All PDF documents used by the system will be placed in this folder after uploading via the interface or after the first upload to the DB via the Python script mentioned earlier in the ‘Database’ section.
Due to its size, the `files_to_upload/` folder with the files to be initially uploaded to the database is not present in the repository.  
You can download it from Google Drive at the following link:
🔗 [Download pdf_files from Google Drive](https://drive.google.com/file/d/1Nzn8ZO0bOhIyewmZWZIPRk4Gnsp-zeiw/view?usp=drive_link)


The file `update_name_pdf_and_move_folder.py` does the following:

- Renames PDF files according to the format required by Gradio
- Automatically copies them to the pdf_files/ folder

#### ❗ Note
Pay close attention to the quality of the input data: avoid duplicate documents in the content (the system only checks that the name of the document you want to insert is not already present), define a file name that does not contain special characters (avoid '.' in the name) and *spaces* (use ‘_’ instead of spaces), make sure that the document format is PDF and that it is not corrupted (no scans/photos), give the document a meaningful name that is not too long (limited to a maximum of 200 characters). It is also recommended to upload a limited number of new files to the screen each round (about 5 at a time).
## ✅ Requirements
Make sure you have installed:

- Python 3.10 (the system has been tested with this version and therefore is not guaranteed to work with previous versions)
- The libraries specified in `requirements.txt`


## ⚙️ Environment Variable Configuration

For the project to work correctly, you need to create a `.env` file in the main directory and define:

- the OpenAI API key with the variable:

```env
OPENAI_API_KEY = your_openai_key
```

- the names for the database connection variables:


```env
HOST_NAME = host_name
DATABASE_NAME = database_name
USER_NAME = user_name
PASSWORD = password
PORT = port_number
```
To load the variables in the `.env` file, use the existing code, i.e.:
```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
# Retrieve your API key from your .env file
OPENAI_API_KEY = os.getenv(“OPENAI_API_KEY”)
# DB connection parameters
HOST_NAME = os.getenv(“HOST_NAME”)
DATABASE_NAME = os.getenv(“DATABASE_NAME”)
USER_NAME = os.getenv(“USER_NAME”)
PASSWORD = os.getenv(“PASSWORD”)
PORT = os.getenv(“PORT”)

```
## 🧩 Intra-Document Clustering and New Upload Pipeline

There are new specific scripts for uploading documents, saving embeddings, and inter-document clustering, which should be taken into consideration if you want to modify the current splitting and upload system.

- **create_cluster.py**
creates embedding clusters in the database. Modify the code if you want to use the version without PCA, as reported in the initial comment of the code itself.

- **DB_for_cluster.py**  
  Manages interactions with the vector database.

- **search_with_cluster.py**
Script for performing semantic searches on clustered chunks, using the combined vector (combined_embedding) produced during loading and clustering.

---
### Important
If you want to modify the current recursive splitting implementation in `chatbot_gradio.py`, you should consider integrating these new files and updating the `chatbot_gradio.py` code accordingly, to maintain consistency with loading, clustering, and inter-document search.


