#!/usr/bin/python3

import os
import copy
import mmh3
import numpy as np
import math
from random import shuffle
from pathlib import Path
import json

from predictor.utility import  msg2log


class BF():
    """
    Bloom filter
    For simplicity, bit array is replaced by bool array
    """

    def __init__(self, filter_size:int=None, fp_prob:float=1e-5,repository:str=None, f:object=None):
        self.size=filter_size
        self.fp_prob=fp_prob
        self.bit_arr=np.zeros((self.size),dtype=bool)
        self.max_items=self.max_number_items()
        self.item_count=0
        self.hash_number = self.get_hash_number(self.size,self.max_items)
        self.f=f

    def __str__(self):
        message= f"""
{self.__class__.__name__}
Bit Array size:                 {self.size}
Expected number inserger items: {self.max_items}
Hash function count:            {self.hash_number}
falsePositive Probability:      {self.fp_prob}
Already inserted items:         {self.item_count}
"""
        return message


    def max_number_items(self):
        n=-self.size * (math.log(2.0) * math.log(2.0))/math.log(self.fp_prob)
        return int(n)

    @classmethod
    def get_hash_number(cls, m,n):
        """
        Returns the number hash functions calculated by formula
        k =(m/n) * ln(2)
        :param m:
        :param n:
        :return:
        """
        k= (float(m)/float(n)) * math.log(2.0)
        return int(k)

    def add_item(self, item):
        """
        Add item to Bloom Filter
        :param item:
        :return:
        """

        digests=[]
        for i in range(self.hash_number):
            digest= mmh3.hash(item,i,signed=False) % self.size
            digests.append(digest)
            self.bit_arr[digest]=True

        message = "item: {} Fired bits are {}".format(item,''.join('{} '.format(i) for i in digests))
        msg2log("add_item",message,self.f)
        self.item_count+=1
        if self.item_count>self.max_number_items():
            p1=math.exp(0.6185* float(self.size)/float(self.item_count))
            message ="\nAmount of items {} is greather than max enabled {} for given falsePositive probability {}" \
                     .format(self.item_count, self.max_items,self.fp_prob,p1)
            msg2log("add_item", message,self.f)
            message="Update falsePositive probability {}".format(p1)
            msg2log("add_item",message,self.f)

    def check_item(self,item):
        """
        Check for existence of an item in the filter
        :param item:
        :return:
        """

        for i in range(self.hash_number):
            digest=mmh3.hash(item,i,signed=False) % self.size
            if self.bit_arr[digest]==False:
                return False
        return True

    def save(self,file_name:str=None):
        if file_name is None:
            return
        with open(file_name,'w') as f:
            json.dump({
                "size":self.size,
                "fp_prob": self.fp_prob,
                "max_items": self.max_items,
                "item_count": self.item_count,
                "hash_number":self.hash_number,
                "bit_arr": self.bit_arr.tolist(),

            },f,sort_keys=True, indent=4)

        pass

    def load(self,file_name:str=None):
        if file_name is None:
            return
        if not Path(file_name).exists():
            message="{} is not found".format(file_name)
            msg2log("load",message, self.f)
            return
        with open(file_name,'r') as ff:
            bf_data=json.load(ff)
        msg2log("load",bf_data,self.f)
        self.size = bf_data['size']
        self.fp_prob = bf_data['fp_prob']
        self.max_items = bf_data['max_items']
        self.item_count = bf_data['item_count']
        self.hash_number = bf_data['hash_number']
        self.bit_arr = np.array(bf_data['bit_arr'])
        msg2log("load","After deserialize:\n,{}".format(self))



if __name__=="__main__":
    pass

    # # words to be added
    # word_present = ['abound', 'abounds', 'abundance', 'abundant', 'accessable',
    #                 'bloom', 'blossom', 'bolster', 'bonny', 'bonus', 'bonuses',
    #                 'coherent', 'cohesive', 'colorful', 'comely', 'comfort',
    #                 'gems', 'generosity', 'generous', 'generously', 'genial']
    #
    # # word not added
    # word_absent = ['bluff', 'cheater', 'hate', 'war', 'humanity',
    #                'racism', 'hurt', 'nuke', 'gloomy', 'facebook',
    #                'geeksforgeeks', 'twitter']
    #
    # with open("a.txt",'w') as ff:
    #     bf=BF(filter_size=1024,fp_prob=0.05,f=ff)
    #     print(bf)
    #     for item in word_present:
    #         bf.add_item(item)
    #
    #     shuffle(word_present)
    #     shuffle(word_absent)
    #     test_words = word_present[:10] + word_absent
    #     shuffle(test_words)
    #     for word in test_words:
    #         if bf.check_item(word):
    #             if word in word_absent:
    #                 print("'{}' is a false positive!".format(word))
    #             else:
    #                 print("'{}' is probably present!".format(word))
    #         else:
    #             print("'{}' is definitely not present!".format(word))
    #     print(bf)
    #
    #     bf.save("bf.json")
    #     del bf
    #     bf1=BF(filter_size=100,fp_prob=0.01,f=ff)
    #     bf1.load("bf.json")
    #     msg2log(None,bf1,ff)
    #     for word in test_words:
    #         if bf1.check_item(word):
    #             if word in word_absent:
    #                 print("'{}' is a false positive!".format(word))
    #             else:
    #                 print("'{}' is probably present!".format(word))
    #         else:
    #             print("'{}' is definitely not present!".format(word))
    #
    #
