import constants
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import os


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


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
input_path = "data/chemprot"
output_path = "data/pickle"

# generate data for vocab files
words = []
poses = []
hypernyms = []
relations = []
edges = []

datasets = ['train', 'dev', 'test']
for dataset in datasets:
    print('Process dataset: ' + dataset)
    reader = BioCreativeReader(os.path.join(input_path, "cdr_data." + dataset + ".txt"))
    raw_documents = reader.read()
    raw_entities = reader.read_entity()
    raw_relations = reader.read_relation()

    title_docs, abstract_docs = data_manager.parse_documents(raw_documents)

    # Pre-process
    title_doc_objs = pre_process.process(title_docs, pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    documents = data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    # documents = data_manager.merge_documents_without_titles(title_doc_objs, abs_doc_objs)

    for doc in documents:
        for sent in doc.sentences:
            deptree = parser.parse(sent)
            if deptree:
                for edge in deptree:
                    rel = '(' + edge[0] + ')'
                    # l_rel = '(l_' + edge[0] + ')'
                    # r_rel = '(r_' + edge[0] + ')'
                    relations.append(rel)
                    # relations.append(l_rel)
                    # relations.append(r_rel)
                    edges.append(tuple([edge[1].content, edge[2].content, rel]))
            for tok in sent.tokens:
                words.append(tok.content)
                poses.append(tok.metadata['pos_tag'])
                hypernyms.append(tok.metadata['hypernym'])

words = sorted(list(set(words)))
poses = sorted(list(set(poses)))
hypernyms = sorted(list(set(hypernyms)))
relations = list(set(relations))
edges = list(set(edges))

print("Number of words: ", len(words))
print("Number of POS tags: ", len(poses))
print("Number of hypernyms: ", len(hypernyms))
print("Number of relations: ", len(relations))

write_vocab(constants.ALL_WORDS, words)
write_vocab(constants.ALL_POSES, poses)
write_vocab(constants.ALL_SYNSETS, hypernyms)
write_vocab('data/no_dir_depend', relations)
write_vocab_edge('data/all_edges.txt', edges)
