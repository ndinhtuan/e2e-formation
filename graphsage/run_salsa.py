import torch
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import os
import glob
import pandas as pd
import random

from encoders import Encoder
from aggregators import MeanAggregator

import utils

DATA_ROOT = "/data/tuannd/fformation/data/Salsa"
PERSONALITY_SCORE_CSV = os.path.join(DATA_ROOT, "Annotation/PersonalityScoresSALSA.csv")

CPP_VIDEO_PATHS = sorted(glob.glob(os.path.join(DATA_ROOT, "salsa_cpp_cam*.avi")))
CPP_LABEL_ROOT = os.path.join(DATA_ROOT, "Annotation", "salsa_cpp")
CPP_FFORMATION_CSV = os.path.join(CPP_LABEL_ROOT, "fformationGT.csv")
CPP_GEOMETRY_DIR = os.path.join(CPP_LABEL_ROOT, "geometryGT")
CPP_ACCEL_FAST_DIR = os.path.join(DATA_ROOT, "Badge", "salsa_cpp", "accel_fast")
CPP_ACCEL_SLOW_DIR = os.path.join(DATA_ROOT, "Badge", "salsa_cpp", "accel_slow")

PS_VIDEO_PATHS = sorted(glob.glob(os.path.join(DATA_ROOT, "salsa_ps_cam*.avi")))
PS_LABEL_ROOT = os.path.join(DATA_ROOT, "Annotation", "salsa_ps")
PS_FFORMATION_CSV = os.path.join(PS_LABEL_ROOT, "fformationGT.csv")
PS_GEOMETRY_DIR = os.path.join(PS_LABEL_ROOT, "geometryGT")
PS_ACCEL_FAST_DIR = os.path.join(DATA_ROOT, "Badge", "salsa_ps", "accel_fast")
PS_ACCEL_SLOW_DIR = os.path.join(DATA_ROOT, "Badge", "salsa_ps", "accel_slow")

NUM_IDS = len(glob.glob(os.path.join(CPP_GEOMETRY_DIR, "*.csv")))

FFORMATION_CSV = PS_FFORMATION_CSV
GEOMETRY_DIR = PS_GEOMETRY_DIR
VIDEO_PATHS = PS_VIDEO_PATHS

class SalsaLoader(object):

    def __init__(self, num_ids):
        r"""

        Args:
            num_ids (int): number of person in one frame
        """
        
        self.num_ids = num_ids
    
    def load_salsa(self, path_fformation, path_geometries):
        r"""

        Args: 
            path_fformation (str): Path to Fformation label file
            path_geometries (str): Path to geometries folder
        Returns:
            feat_data (dict): Dictionary feature of each person (as node) in each frame, having form
                {
                    "timestamp": numpy.array([18, 4]),
                    "timestamp": numpy.array([18, 4]),
                    ...
                }
            adj_lists (dict):  Dictionary of adjacent vertices information
                {
                    "timestamp": [ {list_of_ids_for_this_group}, {list_of_ids_for_this_group}, ... ],
                    "timestamp": [ {list_of_ids_for_this_group}, {list_of_ids_for_this_group}, ... ],
                    ...
                }
        """

        fformations = self._load_fformation(path_fformation)
        geometries = self._load_geometry(path_geometries)

        feat_data = dict()
        adj_lists = dict()

        for timestamp_iter in range(len(geometries.index)):

            timestamp = geometries.index[timestamp_iter]
            _geometries = geometries.iloc[timestamp_iter]
            _fformations = fformations[timestamp]

            objects = []
            for i in range(1, self.num_ids+1):
                ground_x = _geometries[f"ground_x_{i}"]
                ground_y = _geometries[f"ground_y_{i}"]
                body_pose = _geometries[f"bodypose_{i}"]
                head_pose = _geometries[f"headpose_{i}"]
                objects.append([ground_x, ground_y, body_pose, head_pose])
            objects = np.array(objects)
            
            # pre-compute
            cos_bh = np.cos(np.abs(objects[:, 2] - objects[:, 3])) # CHECK FORMULA
            bodypose_vectors = np.stack([np.cos(objects[:, 2]), np.sin(objects[:, 2])], axis=1)
            headpose_vectors = np.stack([np.cos(objects[:, 3]), np.sin(objects[:, 3])], axis=1)
            objects = np.concatenate([
                objects, bodypose_vectors, headpose_vectors, cos_bh[:, None]
            ], axis=-1)

            feat_data[timestamp] = objects
            adj_lists[timestamp] = _fformations
        
        return feat_data, adj_lists
    
    def split(self, feat_data, adj_lists, ratio=0.6):
        r"""
        Split data into train set and test set with ratio for train

        Args: 
            ratio(float): ratio for spliting 
        Returns:
            train_feature_data: get ratio part from feat_data
            train_adj_lists : get ratio part from adj_lists
            test_feature_data: get ratio part from feat_data
            test_adj_lists : get ratio part from adj_lists
        """
        
        keys = feat_data.keys()
        list_keys = list(keys)
        random.shuffle(list_keys)

        len_data = len(keys)
        len_train = int(ratio*len_data)
        len_test = len_data - len_train
        train_keys = list_keys[:len_train]
        test_keys = list_keys[len_train:]

        train_feature_data = dict((k, feat_data[k]) for k in train_keys)
        train_adj_lists = dict((k, adj_lists[k]) for k in train_keys)
        test_feature_data = dict((k, feat_data[k]) for k in test_keys)
        test_adj_lists = dict((k, adj_lists[k]) for k in test_keys)

        return train_feature_data, train_adj_lists, test_feature_data, test_adj_lists

    def _load_fformation(self, path):
        r"""
        
        Args:
            path (str): Path to Fformation label file
        Returns:
            dict: Dictionary of the form
                {
                    "timestamp": [ {list_of_ids_for_this_group}, {list_of_ids_for_this_group}, ... ],
                    "timestamp": [ {list_of_ids_for_this_group}, {list_of_ids_for_this_group}, ... ],
                    ...
                }
        """
        with open(path) as f:
            lines = [line.strip() for line in f.readlines()]
        
        fformations = {}
        for line in lines:
            splitted_lines = line.split(",")
            timestamp = float(splitted_lines[0])
            people_ids = [int(x) for x in splitted_lines[1:]]
            
            if timestamp not in fformations:
                fformations[timestamp] = []
            fformations[timestamp].append(people_ids)

        return fformations

    def _load_geometry(self, _dir):

        paths = [
            os.path.join(_dir, "{:02d}.csv".format(person_id+1)) 
            for person_id in range(NUM_IDS)
        ]

        geometries = pd.concat([
            self._load_individual_geometry(path, i+1)
            for i, path in enumerate(paths)
        ], axis=1)

        return geometries

    def _load_individual_geometry(self, path, person_id):

        df = pd.read_csv(path, header=None, index_col=0)
        df[f"headpose_{person_id}"] = df[4] + df[5]
        df = df.drop(columns=[3, 5])
        df = df.rename(columns={
            1: f"ground_x_{person_id}", 2: f"ground_y_{person_id}", 
            4: f"bodypose_{person_id}",
            6: f"is_valid_{person_id}"
        })

        return df

class GraphFormation(nn.Module):

    def __init__(self, n_out=15):
        
        super(GraphFormation, self).__init__()
        
        self.features = nn.Embedding(18, 9)
        self.enc = None
        self.n_out = n_out

        self.distmult = nn.Parameter(torch.rand(self.n_out))
        
        # Two fully connected for link prediction
        self.two_fc = torch.nn.Sequential(
            torch.nn.Linear(30, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1),
            torch.nn.ReLU(),
        )
    
    def build_model(self, adj_lists):

        self.adj_lists = adj_lists
        self.agg1 = MeanAggregator(self.features, cuda=True)
        self.enc1 = Encoder(self.features, 9, 20, self.adj_lists, self.agg1, gcn=False, cuda=False)
        self.agg2 = MeanAggregator(lambda nodes : self.enc1(nodes).t(), cuda=False)
        self.enc2 = Encoder(lambda nodes : self.enc1(nodes).t(), self.enc1.embed_dim, self.n_out, \
                self.adj_lists, self.agg2, base_model=self.enc1, gcn=True, cuda=False)
        
        self.enc = self.enc2

    def set_features(self, feat_data):

        self.features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    def forward(self, nodes):
        
        embeds = self.enc(nodes)
        return embeds

    def predict_link(self, to_pred):
        """
        Compute for link prediction, based on GCN using CrossEntropy loss

        Args: 
            to_pred (numpy array): Array has dimesion (N, 2) including pair vertices
        """
        #return self.simple_link_prediction(to_pred)
        return self.two_fc_link_prediction(to_pred)

    def simple_link_prediction(self, to_pred):

        distmult = self.distmult.expand(to_pred.shape[0], self.distmult.shape[0])
        embed1 = self.forward(to_pred[:, 0]).T
        embed2 = self.forward(to_pred[:, 1]).T
        
        #print("Shape: ", distmult.shape, embed1.shape, embed2.shape)
        dot = (embed1*distmult*embed2).sum(dim=1)
        print(dot.shape)
        return dot
        #return nn.Sigmoid()(dot)

    def two_fc_link_prediction(self, to_pred):

        embed1 = self.forward(to_pred[:, 0]).T
        embed2 = self.forward(to_pred[:, 1]).T

        cat_embed = torch.cat((embed1, embed2), 1)
        cat_embed = self.two_fc(cat_embed)
        cat_embed = torch.squeeze(cat_embed)
        return cat_embed

def test_salsa(graph, test_feat, test_adj):

    predicts = []
    labels = []

    for key in test_feat.keys():

        feat = test_feat[key]
        adj = test_adj[key]

        all_edges, all_labels = utils.create_all_edge(adj)
        graph.set_features(feat)
        preds = graph.predict_link(all_edges)
        preds = (torch.sigmoid(preds) > 0.5).type(torch.FloatTensor)
        #print("test: ", preds)
        predicts += list(np.array(preds, dtype=np.int16))
        labels += list(np.squeeze(all_labels))
    
    print("F1 score: ", f1_score(predicts, labels))

def train_batch_salsa(graph, optimizer, batch):
    
    batch_feat, batch_adj = batch

    graph.set_features(batch_feat)
    edges, bin_edge = utils.create_bin_edge(batch_adj)
    all_edges, labels = utils.negative_sample(edges, 1, bin_edge)
    labels = torch.Tensor(labels)

    preds = graph.predict_link(all_edges)
    #print("preds: ", (torch.sigmoid(preds) > 0.5).type(torch.FloatTensor))
    loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
    #loss = torch.nn.BCELoss()(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def run_salsa(num_iters=1000, test_iters=10):

    salsa_loader = SalsaLoader(NUM_IDS)
    feat_data, adj_lists = salsa_loader.load_salsa(FFORMATION_CSV, GEOMETRY_DIR)
    train_feat, train_adj, test_feat, test_adj = salsa_loader.split(feat_data, adj_lists)

    adj_lists = dict()
    
    for i in range(18):
        adj_lists[i] = {j for j in range(18)}

    graph = GraphFormation()
    graph.build_model(adj_lists)
    #print(graph); exit()
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graph.parameters()), lr=5e-3)
    
    key_train = list(train_feat.keys())
    len_key_train = len(key_train)
    k = 0
    
    while k <= num_iters:
        key = key_train[k]
        batch = (train_feat[key], train_adj[key])
        loss = train_batch_salsa(graph, optimizer, batch)
        print("Loss: ", loss)

        k += 1
        if k >= len_key_train:
            k = 0

        if k % test_iters == 0:
            test_salsa(graph, train_feat, train_adj)

if __name__=="__main__":
    
    run_salsa()
