# Imports
import nltk
import spacy
import numpy as np
from spacy.en import English
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize

parser = English()

class QAChatbot:

	def __init__(self):
		self.stopwords = stopwords.words('english')
		punctuation_list =   ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
		question_type_list = ["How", "What", "Where", "Why", "Which", "Who", "When"]


		self.stopwords = self.stopwords + punctuation_list + question_type_list
		self.lemmatizer = WordNetLemmatizer()
		self.word_tokenize = word_tokenize
		self.sent_tokenize = sent_tokenize
		self.nlp = spacy.load('en')
		self.wh_answer_map = {
		    "who":     ["PERSON", "ORG"],
		    "when":    ["DATE", "TIME"],
		    "where":   ["GPE"],
		    "which":   ["PERSON", "ORG", "GPE"],
		    "what":    ["PERSON", "ORG"],
		    "default": ["PERSON", "ORG", "GPE", "TIME", "DATE"]
		}
		self.lemmatized_passage = None


	def lemmatize_passage(self, passage):
		"""Sentence tokenizes the passage and gets the lemmatized words of each sentence

		Inputs:
		passage (string) - A paragraph

		Returns: A list of list.
		"""
		self.sentences = self.sent_tokenize(passage)
		lemmatized_passage = [self.lemmatize(sentence) for sentence in self.sentences]
		return lemmatized_passage


	def lemmatize(self, sentence):
		"""Lemmatize the words in sentence. """
		tokenized_words = self.word_tokenize(sentence)
		tokenized_words_lowercase = ([word.lower() for word in tokenized_words])
		filtered_words =  [word for word in tokenized_words_lowercase if word not in self.stopwords]
		lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
		return lemmatized_words

	def get_most_similar_sentence(self, query):
		"""Gets the most similar sentence. """
		lemmatized_query = self.lemmatize(query)
		matching_words_sentences = [len(set(lemmatized_query) & set(lemmatized_sentence)) for lemmatized_sentence in self.lemmatized_passage]
		index_of_most_similar_sentence = matching_words_sentences.index(max(matching_words_sentences))
		most_similar_sentence = self.sentences[index_of_most_similar_sentence]
		return most_similar_sentence

	def extract_answer(self, query, candidate_answer_sentence):
		"""Extracts the answer from the candidate_answer_sentence based on the wh-type of the query.

		Inputs:
		query (string) - The query string or the question asked. Basically to extract the question type.
		candidate_answer_sentence (string) - Takes the most possible answer

		Returns:
		answer (string) - The exact answer in the candidate_answer_sentence
		"""
		answer_sentence = candidate_answer_sentence
		print(answer_sentence)
		words_in_query = [word.lower() for word in self.word_tokenize(query)]
		matched_wh_types = list(set(words_in_query) & set(self.wh_answer_map.keys()))
		wh_type = matched_wh_types[0] if matched_wh_types else "default"
		candidate_answer_sentence_nlp = self.nlp(candidate_answer_sentence)	
		candidate_answers = [ent.text for ent in candidate_answer_sentence_nlp.ents if ent.label_ in self.wh_answer_map[wh_type]]
		matched_query_answer_words = set(words_in_query) & set(self.word_tokenize(candidate_answer_sentence))
		shortest_word_distance_matrix = np.zeros((len(matched_query_answer_words), len(candidate_answers)))
		if shortest_word_distance_matrix: 
			for i, candidate_answer in enumerate(candidate_answers):
				for j, key_words in enumerate(matched_query_answer_words):
					shortest_word_distance_matrix[j][i] += answer_sentence.find(candidate_answer) - answer_sentence.find(key_words)
			return candidate_answers[shortest_word_distance_matrix.sum(axis=0).argmin()]
		else:
			return "Sorry couldn't find the answer"

	def respond(self, query):
		if query.startswith("@passage"):
			self.lemmatized_passage = self.lemmatize_passage(query.split("@passage")[1])
			answer = "Passage Registered. You can answer questions."
		elif self.lemmatized_passage != None:
			most_similar_sentence = self.get_most_similar_sentence(query)
			answer = self.extract_answer(query, most_similar_sentence)
		else:
			answer = "Please register the passage."
		return answer


# passage = """@passage James Bryant Conant (president, 1933–1953) reinvigorated creative scholarship to guarantee its preeminence among research institutions. He saw higher education as a vehicle of opportunity for the talented rather than an entitlement for the wealthy, so Conant devised programs to identify, recruit, and support talented youth. In 1943, he asked the faculty make a definitive statement about what general education ought to be, at the secondary as well as the college level. The resulting Report, published in 1945, was one of the most influential manifestos in the history of American education in the 20th century."""

# query = "Who lead the school back to leading research institution in 2oth century?"
# qachatbot = QAChatbot()
# print(qachatbot.respond(passage))
# print("Response", qachatbot.respond(query))