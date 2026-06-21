#part0 

import numpy as np

#print(np.zeros((3,4)))

#print(2*np.random.random((3,4))-1)

# w= np.random.random((4,1))
# q= np.random.random((3,4))

# print(np.dot(q,w))

#Classcode

# def perceptron_learning(I,T):
#     n= I.shape[1] #number of inputs
#     m= T.shape[1] #number of outputs
#     p= I.shape[0] #number of training examples/patterns
#     w=np.random.rand(n+1,m)
    
#     for i in range()
    
#     return w

# P=10
# I= np.random.random((P,2))
# T= np.random.randint(0,2,(10,3))

# w= perceptron_learning(I,T)
# #first column of only biases
# print(w)

#wants delta w to be called dw 

#f=1000
#old thinking: loop variance (n)

# ========================================================================================================================================
# part-1 (mostly in class)


class Perceptron:
    
    def __init__(self, n, m):
        #n= input and m=output and w=weights
        self.n = n  
        self.m = m  
        #self.w = 0.5 * (2 * np.random.random((n + 1, m)) - 1)
        
        #this creates initialised values with normal distribution (which prof prefers)
        self.w=np.random.randn(n+1,m)/100
        #print(self.w)
    
    def __str__(self):
        return f"A not so smart perceptron with {self.n} input(s) and {self.m} output(s)"


    # def test(self, I):
    #     I = np.append(I, 1)
    #     #returns a matrix with boolean values
    #     return np.dot(I, self.w) > 0
    
    def _forward(self, inputs, j):
        lj = np.append([1], inputs[j])
        oj = (np.dot(lj, self.w) >0).astype(int)
        return lj, oj
    
    def train(self,inputs, targets,iters=1000):
        p=len(inputs) #number of patterns
        for i in range(iters):
            dw=np.zeros(self.w.shape) #weight changes
            for j in range(p): #loop over patterns
                
                #"forward pass"
                lj, oj = self._forward(inputs, j)
                dj=targets[j]-oj

                dw+= np.outer(lj,dj) # delta rule
                
            self.w+=dw/p 
    
    def test(self,inputs):
        p=len(inputs)
        outputs=[]
        for j in range(p):
            _, oj = self._forward(inputs, j) #_ indicates we don't care about the first dummy variable 
            outputs.append(oj)
        return np.array(outputs)
                
        
#=====================================part-2 (in-class)=====================================
    
def show_pattern(pattern):
    for row in range(14):
        for col in range(14):
            print('*' if pattern[14*row+col] > 0.5 else ' ', end='')
        print()


def load_data(filename):

    #read entire dataset as lines of text 
    data = open(filename).read().split('\n')
    
    #strip off pattern names 
    del data[::16]
    
    #save the pattern labels (one-hot codes) before deleting them
    targets = []
    for i in range(14, len(data), 15):
        if i < len(data) and data[i].strip():
            targets.append(list(map(int, data[i].split())))
    
    #strip off pattern labels (one-hot codes)
    del data[14::15]
    
    #join individual strings into a giant list of strings 
    data = ''.join(data).split()
    
    #convert strings to floats
    data = list(map(float, data))

    #convert list of floats to numpy array (column vector conversion w/ reshape)
    inputs = np.array(data).reshape((2500, 196))
    targets = np.array(targets)
    
    return inputs, targets


#=====================================part-3=====================================

#load the data first 

train_inputs, train_targets = load_data('training.txt')
test_inputs, test_targets = load_data('test.txt')

#easy peezy classifer

train_class=train_targets[:,2].reshape(-1,1)
test_class=test_targets[:,2].reshape(-1,1)

#=====================================DRIVER=====================================


if __name__ == "__main__":
    #p= Perceptron(2,1)
    #print(p.w)
    
    #General boolean input 
    inp=[[0,0],
         [0,1],
         [1,0],
         [1,1]]
    
    #Targets for AND 
    and_tgt= [[0],
          [0],
          [0],
          [1]]
    
    or_tgt=[[0],
            [1],
            [1],
            [1]]
    xor_tgt=[[0],
             [1],
             [1],
             [0]]
    
    print("==========Part-1===========\n")
    
    # Test AND 
    print("---AND---\n")
    p_and = Perceptron(2, 1)
    print(f"Before training: {p_and.test(inp).flatten()}")
    p_and.train(inp, and_tgt, iters=100)
    print(f"After training:  {p_and.test(inp).flatten()}")
    print(f"Expected:        [0 0 0 1]")
    
    # Test OR 
    print("\n---OR---\n")
    p_or = Perceptron(2, 1)
    print(f"Before training: {p_or.test(inp).flatten()}")
    p_or.train(inp, or_tgt, iters=100)
    print(f"After training:  {p_or.test(inp).flatten()}")
    print(f"Expected:        [0 1 1 1]")
    
    # XOR (should fail)
    print("\n---XOR---\n")
    p_xor = Perceptron(2, 1)
    p_xor.train(inp, xor_tgt, iters=1000)
    print(f"After training:  {p_xor.test(inp).flatten()}")
    print(f"Expected:        [0 1 1 0]")
    print("    xxxxx FAILS xxxxx      \n")

    print("==========Part-2===========\n")
    #Some data visualization
    for digit in range(10):
        idx = digit * 250
        print(f"\nDigit {digit} (example {idx}):")
        print(f"Target: {train_targets[idx]}")
        show_pattern(train_inputs[idx])

    
        #Perceptron as classifier

    print("==========Part-3===========\n")
    

    #train it

    p=Perceptron(196,1)
    print(p)

    print("\nPERCEPTRON, ASSEMBLE!\n", end='', flush=True)
    p.train(train_inputs, train_class, iters=1000)

    #test it

    predicted=p.test(test_inputs).flatten()
    true_val=test_class.flatten()

    #Assess performance on the test set and report false-positive and false-negative rates
    performance=np.sum(predicted==true_val)/len(true_val)*100
    false_positive_rate=np.sum((predicted==1) & (true_val==0))/np.sum(true_val==0)
    false_negative_rate=np.sum((predicted==0) & (true_val==1))/np.sum(true_val==1)

    print(f"\nTest Performance: {performance:.2f}%")
    print(f"False Positive Rate: {false_positive_rate*100:.2f}%")
    print(f"False Negative Rate: {false_negative_rate*100:.2f}%\n")
    
    

