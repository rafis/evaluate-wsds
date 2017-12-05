from pywsd.similarity import max_similarity
from pywsd.utils import lemmatize_sentence
import nltk
from nltk.corpus import semcor
semcor.ensure_loaded()

def semcor_word_to_str(semcor_word):
    if isinstance(semcor_word, nltk.tree.Tree):
        word = "_".join(semcor_word.leaves())
    else:
        word = semcor_word[0]
    return word

def tree_to_list(tree):
    tree_list = []
    for semcor_word in tree:
        word = semcor_word_to_str(semcor_word)
        tree_list.append(word)
    return tree_list

def evaluate_algorithm(similarity_option, chunk):
    match = 0
    total = 0
    chunk_text = tree_to_list(chunk)
    surface_words, lemmas, morphy_poss = lemmatize_sentence(chunk_text, keepWordPOS=True)
    assert(len(lemmas) == len(chunk))
    for i in range(0, len(chunk)):
        semcor_word = chunk[i]
        # Skip stop-words and punctuation since neither they are in WordNet
        if not isinstance(semcor_word, nltk.tree.Tree):
            continue
        if not isinstance(semcor_word.label(), nltk.corpus.reader.wordnet.Lemma):
            # TODO: semcor_word.label() == 'such.s.00'
            continue
        # Skip named entities
        if semcor_word.label() == nltk.corpus.wordnet.lemma('group.n.01.group') and "') (NE " in semcor_word.pformat():
            continue

        context = [lemma for lemma in lemmas[max(0, i - 15):i+9]]
        lemma = lemmas[i]
        pos = morphy_poss[i]
        synset = max_similarity(context, lemma, pos=pos, option=similarity_option)

        if synset is None:
            # TODO: possibly this is bug, for example, "over-all" should be converted to "overall" before looking in WordNet database
            continue
        if synset is not None and semcor_word.label().synset() == synset:
            match += 1
        total += 1

    accuracy = match / total
    return match, total, accuracy


algorithms = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']
rough_accuracy = True
fileids = semcor.fileids()
if rough_accuracy:
    fileids = fileids[0:200]
for similarity_option in algorithms:
    match = 0
    total = 0
    for fileid in fileids:
        chunk = semcor.tagged_chunks(fileids=[fileid], tag='sem')
        match_chunk, total_chunk, accuracy_chunk = evaluate_algorithm(similarity_option, chunk)
        match += match_chunk
        total += total_chunk
        print(fileid, match_chunk, total_chunk, accuracy_chunk, match, total, match/total)
    print(similarity_option, match/total)
