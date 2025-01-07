from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import pdfplumber
from googlesearch import search
from sentence_transformers import SentenceTransformer
import faiss
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import pinecone
import mimetypes
from urllib.parse import urlparse
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain #need a fix: ~RunnableSequence
from langchain_core.runnables import RunnableSequence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI #need a lil fix: langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
warnings.filterwarnings("ignore")


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize models
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_model = SentenceTransformer(hf_model_name)

# Initialize OpenAI API
llm = OpenAI(openai_api_key=openai_api_key)

api = KaggleApi()
api.authenticate()

# Step 1: Search for relevant web pages

def search_web(company_name, industry_name, num_results=5):
    company_query = (
        f"{company_name} {industry_name} "
        "financial OR environmental OR technological OR ESG policies OR reports "
        "-privacy -user"
    )
    industry_query = f"{industry_name} industry standards OR benchmarks OR trends"
    
    results = []
    for query in [company_query, industry_query]:
        for url in search(query, num_results=num_results, advanced=True):
            results.append(url)
            # print(url)
            # print("\n")
    
    return results



# g_res = search_web("Google", "Technology")
# for x in g_res:
#     print(x)
#     print("\n")



# print("*"*500)
# search_web("Tesla", "Automotive")

# print("*"*500)
# search_web("Emerson", "Industrial Automation")


# # Step 2: Scrape data from web pages

def scrape_webpage(url):
    try:
        url = url.url
        path = urlparse(url).path
        file_type = mimetypes.guess_type(path)[0]
        if file_type == "application/pdf":
            # print(f"Skipping PDF: {url}")
            return ""
        
        # Scrape HTML content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Google_res = search_web("Google", "Technology")

# for url in Google_res:
#     print(scrape_webpage(url))
#     print("*"*100)
#     print("\n")


# # search_web("Tesla", "Automotive")

# # print("*"*500)
# # search_web("Emerson", "Industrial Automation")        

# # scrapped_data = scrape_webpage("https://www.tesla.com/privacy")
# # scrapped_data = scrape_webpage("https://www.google.com/policies/privacy/")        




# # # Step 3: Search and download PDFs


def extract_text_from_pdf(filepath):
    try:
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {filepath}: {e}")
        return ""

def is_valid_pdf_metadata(url, max_size_mb=10, max_pages=30):
    """
    Check the metadata of a PDF file (e.g., file size and number of pages)
    to determine if it meets the criteria for further processing.

    Args:
        url (str): URL of the PDF to check.
        max_size_mb (float): Maximum allowed file size in MB.
        max_pages (int): Maximum allowed number of pages.

    Returns:
        bool: True if the PDF is likely useful based on metadata, False otherwise.
    """
    try:
        
        response = requests.head(url, timeout=10)
        response.raise_for_status()

        # Get file size from headers (in bytes)
        file_size_mb = int(response.headers.get("Content-Length", 0)) / (1024 * 1024)

        # Use a URL parser to try to get the page count from URL (if it is included in the metadata)
        parsed_url = urlparse(url)
        # This is a heuristic: often page counts might be embedded in the URL structure or metadata.
        # You can also check for headers that provide such data.
        page_count = 0  # Set page count to 0 for this example, since we're not using a metadata extraction tool.

        # Apply basic checks for file size and page count
        return file_size_mb <= max_size_mb and page_count <= max_pages
    except Exception as e:
        print(f"Error checking metadata for {url}: {e}")
        return False

def find_and_download_pdfs(company_name, industry_name, num_results=3):
    """
    Search, download, and parse information-dense PDFs related to a company and industry.

    Args:
        company_name (str): Name of the company.
        industry_name (str): Name of the industry.
        num_results (int): Number of search results to consider.

    Returns:
        list[dict]: List of parsed PDF metadata and content.
    """
    query = f"{company_name} {industry_name} industry standards OR current technology OR AI applications OR generative AI OR future trends filetype:pdf"
    pdf_texts = []

    # Use a search engine or custom search method to gather URLs (Here, we're using a placeholder)
    search_results = search(query, num_results=num_results)  # Assuming 'search' is your search method
    
    for url in search_results:
        if url.endswith(".pdf") and is_valid_pdf_metadata(url):
            try:
                # Download the PDF
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                pdf_filename = url.split("/")[-1]
                with open(pdf_filename, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {pdf_filename}")

                # Extract and analyze PDF text
                extracted_text = extract_text_from_pdf(pdf_filename)

                # You can skip the previous information-dense check, as metadata has already been validated
                pdf_texts.append({
                    "company": company_name,
                    "industry": industry_name,
                    "filename": pdf_filename,
                    "text": extracted_text,
                })

                # os.remove(pdf_filename)  # Clean up temporary PDF file
            except Exception as e:
                print(f"Error downloading or processing PDF from {url}: {e}")

    return pdf_texts




# # Example usage
# # google_pdfs = find_and_download_pdfs("Google", "Technology")
# # tesla_pdfs = find_and_download_pdfs("Tesla", "Automotive")


# # with open("sample2.txt", "w") as f:
# #     for pdf in tesla_pdfs:
# #         f.write(pdf["text"])
# #         f.write("\n\n")
# #         f.write("*"*100)
    

# # with open("sample1.txt", "w") as f:
# #     for pdf in google_pdfs:
# #         f.write(pdf["text"])
# #         f.write("\n\n")
# #         f.write("*"*100)
    






# # extract_text_from_pdf("ai-principles-2023-progress-update.pdf")
# # Step 4: Create RAG system
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_rag_system(contents, chunk_size=1000, chunk_overlap=200):
    """
    Create a RAG (Retrieval-Augmented Generation) system for large texts.

    Args:
        contents (list): List of text documents to process.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.

    Returns:
        retriever: The RAG retriever object.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key is required for embedding generation.")

    # Validate contents
    if not contents or not all(isinstance(item, str) for item in contents):
        raise ValueError("Expected 'contents' to be a list of strings.")

    print("Splitting documents into chunks...")
    try:
        # Use RecursiveCharacterTextSplitter for smarter chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],  # Prefer splitting at paragraphs, sentences, or words
        )
        # contents is list of strings
        documents = text_splitter.create_documents(contents)

        print(f"Number of chunks created: {len(documents)}")

        # Generate embeddings
        print("Generating embeddings for documents using OpenAI...")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_documents(documents, embeddings_model)

        # Create retriever
        print("Creating retriever...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        print("RAG system successfully created.")
        return retriever

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def retrieve_and_enrich_with_subqueries(retriever, company_name, industry_name):
    """
    Retrieve and enrich content using targeted subqueries to ensure comprehensive coverage.

    Args:
        retriever: The RAG retriever object.
        company_name (str): The name of the company.
        industry_name (str): The industry the company belongs to.

    Returns:
        str: Enriched and summarized text relevant to the query.
    """
    # Define subqueries
    subqueries = [
        f"{company_name} {industry_name} general industry overview",
        f"{company_name} {industry_name} Generative AI use cases",
        f"{company_name} {industry_name} challenges faced",
        f"{company_name} {industry_name} how AI/ML can solve challenges",
    ]

    # Retrieve documents for each subquery
    print("Retrieving relevant documents for subqueries...")
    retrieved_texts = []
    for subquery in subqueries:
        print(f"Processing subquery: {subquery}")
        relevant_docs = retriever.invoke(subquery)
        retrieved_texts.extend([doc.page_content for doc in relevant_docs])

    # Combine retrieved content
    print(type(retrieved_texts))
    combined_text = " ".join(retrieved_texts)
    print(len(retrieved_texts))
    # print("COMBINED TEXT\n", combined_text)
    if not combined_text.strip():
        return ""

    # Enrich the retrieved text using a Generative AI model
    print("Using Generative AI for enrichment...")

    enrichment_prompt = """
    You are an expert in AI and ML. Based on the following text:
    ---
    {combined_text}
    ---
    Provide:
    1. Key challenges in the company or industry for Generative AI.
    2. Opportunities where AI/ML can overcome these challenges.
    3. Specific Generative AI use cases for this company or industry.
    4. Suggest some innovative AI applications for the company.    
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = PromptTemplate(input_variables=["combined_text"], template=enrichment_prompt)
    chain = LLMChain(llm=llm, prompt=prompt) 

    try:
        enriched_text = chain.invoke(input={"combined_text": combined_text})
        return enriched_text
    except Exception as e:
        print(f"Error during enrichment: {e}")
        return ""


# # Extract Insights
# Extract Insights with Subquery Enrichment
def extract_insights(retriever, company_name, industry_name):
    """
    Extract key insights for a company and industry based on the retrieved and enriched content.

    Args:
        retriever: The retriever object for the content.
        company_name (str): The name of the company.
        industry_name (str): The name of the industry.

    Returns:
        List[str]: A list of key insights for the company.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Step 1: Retrieve and enrich content with subqueries
    enriched_content = retrieve_and_enrich_with_subqueries(retriever, company_name, industry_name)

    enriched_content = enriched_content.get("text", "")
    if not enriched_content.strip():
        print("No enriched content generated from the subqueries.")
        return ["No insights could be generated due to insufficient data."]

    # Step 2: Define a prompt template for extracting insights
    prompt_template = """
    You are an AI assistant specializing in business insights. Based on the provided data:
    - Highlight key strengths and opportunities for the company ({company_name}) in the {industry_name} industry.
    - Identify risks, challenges, and market trends relevant to the company.
    - Provide actionable recommendations or innovations the company could explore.

    Use the following enriched content for your analysis:
    {enriched_content}
    """

    # Step 3: Generate insights using the enriched content
    prompt = PromptTemplate(
        input_variables=["company_name", "industry_name", "enriched_content"],
        template=prompt_template.strip()
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        response = chain.invoke(
            input={"company_name" : company_name,
            "industry_name" : industry_name,
            "enriched_content": enriched_content
            }
        )
        
        # print("RESPONSEEEEE")
        # print(response)
        response = response.get("enriched_content", "")
        insights = [line.strip() for line in response.splitlines() if line.strip()]
        return insights
    except Exception as e:
        print(f"Error during LLM chain execution: {e}")
        return ["Failed to generate insights due to an error."]


def extract_keywords_from_text(input_text):
    """
    Extracts keywords from the provided input text using an LLM.

    Args:
        input_text (str): The input text from which keywords are to be extracted.

    Returns:
        List[str]: A list of extracted keywords.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt_template = """
    You are an AI assistant that specializes in extracting relevant keywords from a given text. These keywords will be used to search for relevant datasets on Kaggle and other platforms.
    Your task is to analyze the provided content and generate a concise list of important keywords or phrases
    that will help in finding relevant datasets.

    Text to analyze:
    {input_text}

    Provide the most important 4,5 extracted keywords keeping in mind that those are helpful for relevant dataset search as a comma-separated list.
    """

    prompt = PromptTemplate(
        input_variables=["input_text"],
        template=prompt_template.strip()
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.invoke(input={"input_text": input_text})
        keywords = response.get("text", "").strip().split(",")
        return [kw.strip() for kw in keywords if kw.strip()]
    except Exception as e:
        print(f"Error during keyword extraction: {e}")
        return []


def search_datasets(keywords):
    """
    Search Kaggle datasets using the provided keywords and return top results.

    Args:
        keywords (List[str]): List of keywords to search for.

    Returns:
        Dict[str, List[Tuple[str, str, str, int]]]: A dictionary of keywords and their corresponding top datasets.
    """
    results = {}
    for keyword in keywords:
        datasets = api.dataset_list(search=keyword, sort_by="hottest")
        results[keyword] = [
            (dataset.ref, dataset.title, dataset.size, dataset.voteCount) for dataset in datasets[:5]
        ]
    return results


def display_dataset_choices(results):
    """
    Display dataset options for each keyword and allow the user to select from them.

    Args:
        results (Dict[str, List[Tuple[str, str, str, int]]]): The dictionary of datasets per keyword.

    Returns:
        List[Tuple[str, str]]: The list of selected dataset references and titles.
    """
    print("Dataset search results:")
    selected_datasets = []

    for keyword, datasets in results.items():
        print(f"\nKeyword: {keyword}")
        for i, (ref, title, size, votes) in enumerate(datasets, 1):
            print(f"  {i}. {title} (Votes: {votes}, Size: {size}) [Ref: {ref}]")

        print("\nEnter the numbers of the datasets you want to select for this keyword (comma-separated):")
        user_input = input("Your choice: ").strip()
        if user_input:
            choices = [int(choice) - 1 for choice in user_input.split(",") if choice.isdigit()]
            selected_datasets.extend([datasets[i][:2] for i in choices if 0 <= i < len(datasets)])

    return selected_datasets


def main():
    try:
        # Step 1: Gather company and industry names
        company_name = input("Enter the company name: ").strip()
        industry_name = input("Enter the industry name: ").strip()

        print("Searching for relevant web pages...")
        urls = search_web(company_name, industry_name)
        if not urls:
            print("No relevant web pages found.")
            return
        print("Found URLs:", urls)
        print("Number of links found:", len(urls))

        # Step 2: Scrape data from web pages
        print("Scraping web pages...")
        webpage_texts = [scrape_webpage(url) for url in urls]

        # Step 3: Search and download PDFs
        print("Searching for and downloading PDFs...")
        pdf_texts = find_and_download_pdfs(company_name, industry_name)

        # Combine all retrieved content
        all_texts = webpage_texts + pdf_texts
        all_texts = [text for text in all_texts if text]  # Filter out empty texts

        if not all_texts:
            print("No data could be retrieved from the web pages or PDFs.")
            return

        all_texts = [text for text in all_texts if isinstance(text, str) and text.strip()]

        # Step 4: Create RAG system
        print("Setting up the Retrieval-Augmented Generation (RAG) system using FAISS...")
        print(f"Type of all_text {type(all_texts)}")
        print(f"First 5 elements of all_texts: {all_texts[:5]}")
        retriever = create_rag_system(all_texts)

        # Step 5: Extract insights
        print("Extracting important insights...")
        insights = extract_insights(retriever, company_name, industry_name)

        print(f"\nKey insights for {company_name} in the {industry_name} industry:")
        print(insights)

        print("Extraction of keywords from the insights")
        
        keywords = extract_keywords_from_text(" ".join(insights))
        print("Extracted Keywords:", keywords)
        # Search for datasets based on keywords
        search_results = search_datasets(keywords)

    # Display choices and let the user select datasets
        selected_datasets = display_dataset_choices(search_results)

        if not selected_datasets:
            print("No datasets were selected.")
        else:
            print("\nYou selected the following datasets:")
            for ref, title in selected_datasets:
                print(f"  - {title} [Ref: {ref}]")

        # Download selected datasets
        download_path = "data/selected_datasets"
        os.makedirs(download_path, exist_ok=True)
        for ref, title in selected_datasets:
            print(f"\nDownloading dataset: {title}...")
            api.dataset_download_files(ref, path=download_path, unzip=True)

        print(f"\nDatasets downloaded to {download_path}.")
        for file in os.listdir(download_path):
            print(f"  - {file}")


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()



#31 dec working fine 
#task 1 : add feedback, hf dataset extracter agent
