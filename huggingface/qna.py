from transformers import pipeline

# # Load a question-answering pipeline
# qa = pipeline("question-answering")
# result = qa({
#     "question": "how much experience i have ?",
#     "context": """I am an experienced Python developer with expertise in backend development,
#     AI, machine learning, and big data processing. I specialize in building scalable applications
#     using frameworks like FastAPI and Django, creating efficient algorithms, and solving complex
#     problems. With over 5 years of experience, I deliver innovative solutions and drive impactful
#     results."""
# })
# print(result)




# #2from transformers import pipeline
# from PIL import Image
# import requests
# from io import BytesIO

# # Initialize the document question answering pipeline
# nlp = pipeline(
#     "document-question-answering",
#     model="impira/layoutlm-document-qa",
# )

# # Function to load and convert image
# def load_image(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content)).convert("RGBA")
#     return img

# # Input image URLs and questions
# image_urls = [
#     "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
#     "https://miro.medium.com/max/787/1*iECQRIiOGTmEFLdWkVIH2g.jpeg",
#     "https://www.accountingcoach.com/wp-content/uploads/2013/10/income-statement-example@2x.png"
# ]

# questions = [
#     "What is the invoice number?",
#     "What is the purchase amount?",
#     "What are the 2020 net sales?"
# ]

# # Iterate over the image URLs and questions
# for image_url, question in zip(image_urls, questions):
#     img = load_image(image_url)  # Load and convert image
#     answer = nlp(img, question)  # Get answer from the model
#     print(f"Question: {question}")
#     print(f"Answer: {answer[0]['answer']}")
#     print(f"Score: {answer[0]['score']}")
#     print("-" * 40)



#3
from transformers import pipeline
from PIL import Image, ImageEnhance
import pytesseract

# Initialize the document question answering pipeline
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

# Function to enhance image quality (increase contrast)
def enhance_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast to make text clearer
    return img

# Function to resize image
def resize_image(image, target_size=(1024, 1024)):
    return image.resize(target_size)

# Function to extract text using OCR to verify if the image contains readable text
def extract_text(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print(f"Extracted text: \n{text}")  # Output the text for debugging
    return text

# Input image file paths and questions
image_paths = [
    "report.JPEG", 
]

questions = [
    
    "What is Hospital name.?"
    
]

# Iterate over the local image paths and questions
for image_path, question in zip(image_paths, questions):
    img = enhance_image(image_path)  # Enhance the image to improve text clarity
    img = resize_image(img)  # Resize the image
    extracted_text = extract_text(image_path)  # Optionally extract text using OCR

    answer = nlp(img, question)  # Get answer from the model
    print(f"Question: {question}")
    print(f"Answer: {answer[0]['answer']}")
    print(f"Score: {answer[0]['score']}")
    print("-" * 40)



# #4
# from transformers import pipeline, LayoutLMv3TokenizerFast
# from PIL import Image
# import pytesseract

# # Initialize the fast tokenizer explicitly
# tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
# nlp = pipeline(
#     "document-question-answering",
#     model="microsoft/layoutlmv3-base",
#     tokenizer=tokenizer,
# )

# # Function to load and extract text and word boxes from an image using Tesseract OCR
# def extract_text_and_boxes_from_image(image_path):
#     img = Image.open(image_path)
    
#     # Use pytesseract to extract word-level information
#     ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
#     words = []
#     boxes = []
    
#     for i in range(len(ocr_data['text'])):
#         word = ocr_data['text'][i]
#         if word.strip():  # Ignore empty words
#             words.append(word)
#             # Extract word bounding boxes: [x, y, width, height]
#             box = [ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]]
#             boxes.append(box)
    
#     return words, boxes

# # Input image file paths and questions
# image_paths = [
#     "report.jpeg.JPEG", 
# ]

# questions = [
#     "What is patient name?",
#     "What is age?",
#     "What ref by?"
# ]

# # Iterate over the local image paths and questions
# for image_path in image_paths:
#     print(f"Extracted text from {image_path}:")
    
#     # Extract text and boxes using OCR
#     extracted_words, word_boxes = extract_text_and_boxes_from_image(image_path)
    
#     # Check and debug the extracted text
#     print(f"Extracted text type: {type(extracted_words)}")
#     print(f"Extracted text: {' '.join(extracted_words[:200])}...")  # Print first 200 words
    
#     # Prepare the input dictionary for the model
#     for question in questions:
#         try:
#             # Ensure word_boxes is a list of lists containing coordinates
#             word_boxes_list = [
#                 [int(coord) for coord in box] for box in word_boxes
#             ]
            
#             # Tokenize the extracted words as a list of individual words
#             inputs = tokenizer(extracted_words, return_tensors="pt", truncation=True, padding=True)
            
#             # Get the answer from the QA model (provide both context and word boxes)
#             result = nlp({
#                 "question": question,
#                 "context": ' '.join(extracted_words),
#                 "word_boxes": word_boxes_list  # Use word_boxes in the correct format
#             })
#             print(f"Question: {question}")
#             print(f"Answer: {result['answer']}")
#             print(f"Score: {result['score']}")
#         except Exception as e:
#             print(f"Error processing question: {question}")
#             print(f"Exception: {e}")
#         print("-" * 40)
