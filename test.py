from collections import defaultdict
import constants
import models
import pre_process
from data_managers import CDRDataManager as data_manager
from feature_engineering.deptree.parsers import SpacyParser
from feature_engineering.graph.dependency_graph import DepGraph
from pre_process import opt as pre_opt
from readers import BioCreativeReader
import itertools
import copy
import os
from feature_engineering.deptree.sdp import Finder
from sklearn.utils import shuffle
from feature_engineering.deptree.deptree_model import DepTree
import numpy as np
import pickle
from models import Token
from nltk.corpus import wordnet as wn


def process_one(doc):
    a = list()
    for sent in doc.sentences:
        deptree = parser.parse(sent)
        a.append(deptree)
    return a


def process_one_tree(doc):
    a = list()
    for sent in doc.sentences:
        deptree = DepGraph(sentence=sent)
        a.append(deptree)
    return a


def get_candidate(sent, entities):
    """
    :param models.Sentence sent:
    :param list of models.BioEntity entities:
    :return: list of (models.BioEntity, models.BioEntity)
    """
    chem_list = []
    dis_list = []

    min_offset = sent.doc_offset[0]
    max_offset = sent.doc_offset[1]

    for entity in entities:
        try:
            if min_offset <= entity.tokens[0].doc_offset[0] < max_offset:
                if entity.type == constants.ENTITY_TYPE_CHEMICAL:
                    chem_list.append(entity)
                elif entity.type == constants.ENTITY_TYPE_DISEASE:
                    dis_list.append(entity)
        except:
            print(entity.content)

    return list(itertools.product(chem_list, dis_list))


print('Start')
pre_config = {
    pre_opt.SEGMENTER_KEY: pre_opt.SpacySegmenter(),
    pre_opt.TOKENIZER_KEY: pre_opt.SpacyTokenizer()
}
parser = SpacyParser()
spd_finder = Finder()
input_path = "data/cdr"
output_path = "data/sdp"

datasets = ['test_mini']
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

    # Generate data
    dict_nern = defaultdict(list)
    data_tree = defaultdict()

    # generate data for vocab files:

    for doc in documents:
        raw_entity = raw_entities[doc.id]
        # for sent in doc.sentences:
        #     for tok in sent.tokens:
        #         print(tok.doc_offset)

        for r_en in raw_entity:
            entity_obj = models.BioEntity(tokens=[], ids={})
            entity_obj.content = r_en[3]
            entity_obj.type = constants.ENTITY_TYPE_CHEMICAL if r_en[4] == "Chemical" else constants.ENTITY_TYPE_DISEASE
            entity_obj.ids[constants.MESH_KEY] = r_en[5]

            for s in doc.sentences:
                if s.doc_offset[0] <= int(r_en[1]) < s.doc_offset[1]:
                    for tok in s.tokens:
                        if (int(r_en[1]) <= tok.doc_offset[0] < int(r_en[2])
                                or int(r_en[1]) < tok.doc_offset[1] <= int(r_en[2])
                                or tok.doc_offset[0] <= int(r_en[1]) < int(r_en[2]) <= tok.doc_offset[1]):
                            entity_obj.tokens.append(tok)
            if len(entity_obj.tokens) == 0:
                print(doc.id, r_en)
            dict_nern[doc.id].append(entity_obj)

        dep_tree = process_one_tree(doc)
        data_tree[doc.id] = dep_tree

    for doc_idx in data_tree:
        for tree in data_tree[doc_idx]:
            sentence = tree.sentence
            pairs = get_candidate(sentence, dict_nern[doc_idx])
            if len(pairs) == 0:
                continue

            for pair in pairs:
                chem_entity = pair[0]
                dis_entity = pair[1]

                chem_token = chem_entity.tokens[-1]
                dis_token = dis_entity.tokens[-1]

                path = tree.get_shortest_path(chem_token, dis_token)
                print(path)

        # print(data_tree[doc_idx])
        # for edges in data_tree[doc_idx]:
        #     if edges:
        #         sub_tree = DepTree(edges=edges)
        #         all_trees.extend(edges)
        #         root_edge = ('sent', root_node, sub_tree.root)
        #         all_trees.append(root_edge)
        # print('doc_tree: ', all_trees)
