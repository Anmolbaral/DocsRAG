from pypdf import PdfReader
import re


def load_pdf(path):
	reader = PdfReader(path)
	all_chunks= []
	for pageNum, page in enumerate(reader.pages):
		text = page.extract_text()
		if text:
			chunkTexts = create_overlap_chunks(text, chunkSize=5, overlap=2)
			for chunkText in chunkTexts:
				chunkObject = {
					"text": chunkText,
					"metadata": {
						"page": pageNum+1,
						"path": path,
						"section": path.split("/")[-1]
					}
				}
				all_chunks.append(chunkObject)
	return all_chunks

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

