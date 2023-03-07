import constants
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import os
from collections import defaultdict
from models import Token
from nltk.corpus import wordnet as wn


def write_vocab(filename, vocab_list):
    f = open(filename, 'w', encoding='utf-8')
    f.write('')
    f.write('\n')
    for v in vocab_list:
        f.write(v.strip())
        f.write('\n')
    f.write('$UNK$')


def write_vocab_edge(filename, vocab_list):
    f = open(filename, 'w', encoding='utf-8')
    f.write('')
    f.write('\n')
    for v in vocab_list:
        f.write('{} {} {}'.format(v[0].strip(), v[1].strip(), v[2].strip()))
        f.write('\n')
    f.write('$UNK$')


def process_one(doc):
    a = list()
    for sent in doc.sentences:
        deptree, root = parser.parse(sent)
        a.append(tuple([deptree, root]))
    return a


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
input_path = "data/cdr"
output_path = "data/"

# generate data for vocab files
words = []
poses = []
hypernyms = []
relations = []
edges = []

datasets = ['train', 'dev', 'test']
for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "cdr_" + dataset + ".txt"))
    raw_documents = reader.read()
    raw_entities = reader.read_entity()
    raw_relations = reader.read_relation()

    title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

    # Pre-process
    title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    # documents = data_manager.merge_documents_without_titles(title_doc_objs, abs_doc_objs)

    data_tree = defaultdict()

    data_doctree = defaultdict()

    for doc in documents:
        dep_tree = process_one(doc)
        data_tree[doc.id] = dep_tree
        for sent in doc.sentences:
            for tok in sent.tokens:
                words.append(tok.content)
                poses.append(tok.metadata['pos_tag'])
                hypernyms.append(tok.metadata['hypernym'])

    for doc_idx in data_tree:
        # print(data_tree[doc_idx])
        root_node = Token(content='$ROOT$', doc_offset=(-1, -1), sent_offset=(-1, -1))
        root_node.metadata['pos_tag'] = 'NN'
        root_node.metadata['hypernym'] = str(wn.synset('entity.n.01').offset())
        all_trees = []
        for _edges, root in data_tree[doc_idx]:
            if _edges:
                # sub_tree = DepTree(edges=edges)
                r = _edges[0][1]
                for rel, pa, ch in _edges:
                    if pa.content == root.content and pa.metadata['pos_tag'] == root.metadata['pos_tag']:
                        r = pa
                    elif ch.content == root.content and ch.metadata['pos_tag'] == root.metadata['pos_tag']:
                        r = ch
                    else:
                        pass
                all_trees.extend(_edges)
                root_edge = ('sent', root_node, r)
                all_trees.append(root_edge)

        data_doctree[doc_idx] = all_trees

    for idx in data_doctree:
        for e in data_doctree[idx]:
            rel = '(' + e[0] + ')'
            relations.append(rel)
            edges.append(tuple([e[1].content, e[2].content, rel]))
            # print(tuple([e[1].content, e[2].content, rel]))

words = sorted(list(set(words)))
poses = sorted(list(set(poses)))
hypernyms = sorted(list(set(hypernyms)))
relations = list(set(relations))
edges = list(set(edges))

words.append('$ROOT$')
relations.append('(sent)')

print("Number of words: ", len(words))
print("Number of POS tags: ", len(poses))
print("Number of hypernyms: ", len(hypernyms))
print("Number of relations: ", len(relations))

write_vocab(constants.ALL_WORDS, words)
write_vocab(constants.ALL_POSES, poses)
write_vocab(constants.ALL_SYNSETS, hypernyms)
write_vocab('data/no_dir_depend.txt', relations)
write_vocab_edge('data/all_edges.txt', edges)
