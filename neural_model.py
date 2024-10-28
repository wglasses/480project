import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import murmurhash3_32
import math
import random
from bitarray import bitarray
import matplotlib.pyplot as plt




random.seed(11)
def hash_function(m):
        
        seed = random.randint(0,10000000)
        def hash(key):
                return murmurhash3_32(key, seed) % m
        return hash, seed




class AdaptiveBloomFilter():
        def __init__(self, r, n):
                #n is the number of elements
                self.n = n
                self.fp_rate = 0.618 ** (r/self.n)
             
                self.r = math.ceil((-n * math.log(self.fp_rate)) / (math.log(2)**2))
             
                self.table = bitarray(int(self.r))
                self.table.setall(0)
                
             
                self.num_hash = math.ceil((self.r / self.n) * math.log(2))
                self.hash_regionsfor0_20 = list()
                self.hash_regionsfor20_50 = list()
                self.hash_regionsfor50_80 = list()
                self.hash_regionsfor80_100 = list()

                #need to come up with an explanation why we chose these numbers for num of hash functions
                for x in range(self.num_hash//2):
                              curr, _ = hash_function(self.r)
                              self.hash_regionsfor0_20.append(curr)
                for x in range(self.num_hash//4):
                              curr, _ = hash_function(self.r)
                              self.hash_regionsfor20_50.append(curr)
                for x in range(self.num_hash//16):
                                curr, _ = hash_function(self.r)
                                self.hash_regionsfor50_80.append(curr)
                for x in range(self.num_hash//32):
                                curr, _ = hash_function(self.r)
                                self.hash_regionsfor80_100.append(curr)
                
   
        def insert(self, key):
                for hash1 in self.hash_functions:
                 
                        index = hash1(key)
                        self.table[index] = 1
        
        def test(self, key):
                if key < 0.20:
                        for hash1 in self.hash_regionsfor0_20:
                                if self.table[hash1(key)] == 0:
                                        return False
                if key >= 0.20 and key < 0.50:
                        for hash1 in self.hash_regionsfor20_50:
                                if self.table[hash1(key)] == 0:
                                        return False
                if key >= 0.50 and key < 0.80:
                        for hash1 in self.hash_regionsfor50_80:
                                if self.table[hash1(key)] == 0:
                                        return False
                if key >= 0.80:
                        for hash1 in self.hash_regionsfor80_100:
                                if self.table[hash1(key)] == 0:
                                        return False    
                return True


# Define the neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Integrate Neural Network with Bloom Filter
class AdaptiveLearnedBloomFilter:
    def __init__(self, model, bloom_filter):
        self.model = model
        self.bloom_filter = bloom_filter
    
    def add(self, item):
        self.bloom_filter.add(item)
    
    def check(self, item):
        prediction = self.model.predict(np.array([item]))[0][0]
        
        return self.bloom_filter.test(prediction)



if __name__ == "__main__":

    X = np.random.rand(1000, 10)
    y = np.random.randint(2, size=1000)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    model = create_model(input_dim=X.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Create the Bloom filter
    bloom_filter = AdaptiveBloomFilter(r=1000, n=900)
    
    # Create the Adaptive Learned Bloom Filter
    ada_bf = AdaptiveLearnedBloomFilter(model, bloom_filter)
    
    # Add elements to the Bloom filter
    for item in X_train:
        ada_bf.add(item)
    
    # Check elements
    predictions = [ada_bf.check(item) for item in X_test]
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")