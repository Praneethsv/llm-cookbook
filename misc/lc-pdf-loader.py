from langchain_community.document_loaders import UnstructuredPDFLoader

text_loader = UnstructuredPDFLoader(
    image_extractor=True,
    table_extractor=True,
    file_path="/home/sv/Downloads/Nexus-Downloads/BackUp-Downloads/Venkata_Praneeth_CV_Updated.pdf",
)

pdf_data = text_loader.load()


for page in pdf_data:

    tables = page.tables

    images = page.images

print(tables)
