from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_and_chunk(data_dir="./Dataset"):
    #load pdf
    loader = DirectoryLoader(
        path=data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()


    #chunk pdf
    #check chunkviz.up.railway to visualize the chunks

    Markdown_sep = [
    #orders (from high to low)
    "\n#{1,6}", #headings markdown
    "```\n",    #code blocks markdown
    "\n\\*\\*\\*+\n",   
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",]


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        add_start_index=True,
        strip_whitespace=True,
        separators=Markdown_sep,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages → {len(splits)} chunks")
    return splits

