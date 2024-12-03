# https://www.sbert.net/examples/applications/retrieve_rerank/README.html
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb


import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


# We use the Bi-Encoder to encode all passages so that we can use it for semantic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     # Truncate long passages to 256 tokens
top_k = 32                          # Number of passages we want to retrieve with the Bi-Encoder

# The Bi-Encoder will retrieve 100 documents. We use a Cross-Encoder to re-rank the results to improve quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# As the dataset, we use Simple English Wikipedia. Compared to the full English Wikipedia, it only has about 170k articles. We split these articles into paragraphs and encode them with the Bi-Encoder

wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())

        # Add all paragraphs
        #passages.extend(data['paragraphs'])

        # Only add the first paragraph
        passages.append(data['paragraphs'][0])

print("Passages:", len(passages))

# We encode all passages into our vector space. This takes about 5 minutes (depending on your GPU speed)
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

# This function will search all Wikipedia articles for passages that answer the query
def search(query):
    print("Input question:", query)

    ##### Semantic Search #####
    # Encode the query using the Bi-Encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the Cross-Encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the Cross-Encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output the top 5 hits from the Bi-Encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output the top 5 re-ranked results
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))

