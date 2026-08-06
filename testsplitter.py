from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")

results = ocr.predict("data/uploads/cur.jpeg")

for page in results:
    print("\nExtracted Text:")
    print("-" * 50)

    for text in page["rec_texts"]:
        print(text)