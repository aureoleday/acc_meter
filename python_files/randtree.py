# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:11:26 2019

@author: Administrator
"""
#node element definition
#leve farther self child#0 child#1 ... child#n

#import numpy as np
import random

class rand_tree(object):
    def __init__(self,level,nodes,max_leaf):
        if (max_leaf**level-1)/(max_leaf-1) < nodes:
            print("errro: levels and max_leaf combination are not enough to generation designated node counts\n")
        else:            
            self.max_level = level
            self.max_nodes = nodes
            self.max_leaf = max_leaf
            self.nodes = [[0,0,0]]      #[level,farther,self]
            self.av_nodes_index = []
            self.node_cnt = 1        
            print("level: %d,nodes: %d,max_leaf: %d,\n" % (level,nodes,max_leaf))    

    def reset(self):
        self.nodes = [[0,1,0]]
        self.heads = []
        self.av_nodes_index = []
        self.node_cnt = 1    
    
    def get_av_nodes_index(self):
        self.av_nodes_index = []
        if(len(self.nodes) > self.max_nodes):
            return 0
        else:
            for i in range(len(self.nodes)):
                if len(self.nodes[i]) < (self.max_leaf+3) and self.nodes[i][0]<self.max_level:
                    self.av_nodes_index.append(i)
            return len(self.av_nodes_index)        
    
    def gen_node(self):
        if self.get_av_nodes_index() > 0:
            index = random.sample(self.av_nodes_index,1)[0]
            self.nodes[index].append(self.node_cnt)
            self.nodes.append([self.nodes[index][0]+1, self.nodes[index][2], self.node_cnt])
            self.node_cnt += 1
            return 1
        else:
            return 0
            
    def gen_all(self):
        while 1:
            if self.gen_node() == 0:
                break
                

if __name__ == '__main__':
    rand_tree_inst = rand_tree(4,15,2)  #4层，最大节点数15，最大子节点数2
    rand_tree_inst.gen_all()
    print(rand_tree_inst.nodes)
    


                        
            
            
            

    

