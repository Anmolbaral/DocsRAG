from pypdf import PdfReader
import re, os


def load_pdf(path):
	try:
		if os.path.getsize(path) == 0:
			# print(f"File {path} is empty")
			return []
		reader = PdfReader(path)
		all_chunks= []
		for pageNum, page in enumerate(reader.pages):
			text = page.extract_text()
			if text:
				chunkTexts = create_overlap_chunks(text, chunkSize=5, overlap=2)
				for chunkIndex, chunkText in enumerate(chunkTexts):
					if path.split("/")[-2] == "resume":
						groundTruth = True
					else:
						groundTruth = False
					chunkObject = {
						"text": chunkText,
						"metadata": {
							"page": pageNum+1,
							"path": path,
							"category": path.split("/")[-2],
							"filename": path.split("/")[-1],
							"isGroundTruth": groundTruth,
							"chunkId": chunkIndex
						}	
					}
					all_chunks.append(chunkObject)
		return all_chunks
	except Exception as e:
		print(f"Error loading file {path}: {e}")
		return []

# splitting larger files into chunks of sentences
def split_into_sentences(text):
	text = re.sub(r'\n\s*\n', ' ', text)
	text = re.sub(r'\s+', ' ', text)
	text = re.sub(r'-\s+', '', text)
	text = re.split(r'[.!?]', text)
	
	clean_sentences = []

	for sentence in text:
		sentence = sentence.strip()
		if sentence:
			clean_sentences.append(sentence + '.')
	return clean_sentences

def create_sentences_chunks(sentences, chunkSize=5, overlap=2):
	chunks = []
	start = 0

	while start < len(sentences):
		end = start + chunkSize
		chunks.append(" ".join(sentences[start:end]))
		start += chunkSize - overlap

		if start >= len(sentences):
			break
	return chunks

def create_overlap_chunks(sentences, chunkSize=5, overlap=2):
	splitSentences = split_into_sentences(sentences)
	chunks = create_sentences_chunks(splitSentences, chunkSize, overlap)
	return chunks

