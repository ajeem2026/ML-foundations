
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from tqdm import tqdm
# ===============================================BACKPROP=========================================================================================

# part-1.1 (setting up the backprop algorithm)

class Backprop:
    
    def __init__(self, n, m,h):
        #n= input and m=output and w=weights and h=hidden units
        self.n = n  
        self.m = m
        self.h=h
        
        #this creates initialised values with normal distribution (which prof prefers)
        self.wih=np.random.randn(n+1,h)/2
        self.who=np.random.randn(h+1,m)/2
    
    def __str__(self):
        return f"A backprop network with {self.n} input(s) and {self.h} hidden unit(s) and {self.m} output(s)"
    
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def _sigmoid_derivative(self,x):
        return self._sigmoid(x)*(1-self._sigmoid(x))
    
    def _hiddenForward(self, I):
        Inp=np.c_[np.ones(len(I)), I]
        Hnet=(Inp @ self.wih)
        return Inp, Hnet
    
    def _outForward(self, I):
        Inp=np.c_[np.ones(len(I)), I]
        Onet=(Inp @ self.who)
        return Inp, Onet
    
    def _appendBias(self, I):
        return np.c_[np.ones(len(I)), I]
    
    #begin LEARNING ALGORITHM
    
    #This is the real backprop learning algorithm.  Modifies weights and returns array of RMS errors over training epochs
    
    def train(self, inputs, targets, iters=1000, eta=0.5):

        inputs=np.asarray(inputs)
        targets=np.asarray(targets)
        
        # number of patterns
        p=len(inputs)   
        
        #empty list to track RMS error over iterations
        rmserr = []

        for _ in tqdm(range(iters)):

            # zeroes_like automaticlaly mashes dimensions and dtype 
            dWih=np.zeros_like(self.wih)
            dWho=np.zeros_like(self.who)

            err=0

            for j in range(p):
                
                # keep row shape (1,n)
                x=inputs[j:j+1]   
                t=targets[j:j+1]

                # forward pass
                # (1,n+1), (1,h)
                InpH,Hnet=self._hiddenForward(x)     
                H=self._sigmoid(Hnet)
                
                # (1,h+1), (1,m)
                InpO,Onet=self._outForward(H)        
                O=self._sigmoid(Onet)

                # output delta
                delta_O = (t-O) * self._sigmoid_derivative(Onet)

                # hidden delta (discard bias row of who)
                who_no_bias=self.who[1:, :]

                delta_H=(delta_O @ who_no_bias.T) * self._sigmoid_derivative(Hnet)

                # accumulate delta w
                dWih+=InpH.T @ delta_H
                dWho+=InpO.T @ delta_O
                
                #extra cred
                err+=np.sum((t - O) ** 2)

            # batch update weights
            self.wih+=eta*(dWih / p)
            self.who+=eta*(dWho / p)

            rmserr.append(np.sqrt(err/(p * self.m)))

        return rmserr
        

    def test(self,inputs):
        p=len(inputs)
        
        #for hidden layer
        
        _,Hnet=self._hiddenForward(inputs)
        H=self._sigmoid(Hnet)
        
        #for output layer
        
        _,Onet=self._outForward(H)
        return self._sigmoid(Onet)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'wih': self.wih, 'who': self.who}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.wih = data['wih']
        self.who = data['who']
        print(f"Weights loaded from {filename}")
        
## #=====================================Helpers (mostly from previous assignment)=====================================

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

def report_results(predicted, true_val, threshold, label=""):
    tgt = (predicted > threshold).astype(int).flatten()
    true_flat = true_val.flatten()

    # true positives and true negatives
    
    # 2 called 2
    tp=np.sum((tgt == 1) & (true_flat == 1))
    # not-2 called not-2  
    tn=np.sum((tgt == 0) & (true_flat == 0))  

    # false positives and false negatives
    
    # not-2 called 2
    fp=np.sum((tgt == 1) & (true_flat == 0))  
    # 2 called not-2
    fn=np.sum((tgt == 0) & (true_flat == 1))  

    fp_rate=fp/(fp+tn)*100
    fn_rate=fn/(fn+tp)*100

    print(f"False-Positive Rate: {fp_rate:.2f}%")
    print(f"False-Negative Rate: {fn_rate:.2f}%\n")

    # confusion matrix (for part 2)
    print(f"{'':>12} {'Not Two':>10} {'Two':>6}")
    print("-" * 30)
    print(f"{'Not Two |':>12} {tn:>10} {fp:>6}")
    print(f"{'Two |':>12} {fn:>10} {tp:>6}")

        
#extra credit

def ploterr(rmserr, title):

    plt.plot(rmserr)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('RMS Error')
    plt.savefig(f'{title}_error.png')
    plt.show()
    
#=====================================p1 XOR with backprop=====================================

#Extra credit part 4: error surface visualization

#debugging with claude and had to look up how to actually do this 
def plot_error_surface(net, inputs, targets):
    
    inputs=np.asarray(inputs,dtype=float)
    targets=np.asarray(targets,dtype=float)

    # varying first two weights of wih and keeping others fixed
    w_range=np.linspace(-5, 5, 60)
    W1,W2=np.meshgrid(w_range, w_range)
    Z=np.zeros_like(W1)

    # save original weights
    orig_00=net.wih[0, 0]
    orig_01=net.wih[0, 1]

    for i in range(len(w_range)):
        for j in range(len(w_range)):
            net.wih[0,0] = W1[i, j]
            net.wih[0,1] = W2[i, j]

            out = net.test(inputs)
            Z[i,j] = np.sqrt(np.mean((targets - out) ** 2))

    # restore original weights
    net.wih[0,0]=orig_00
    net.wih[0,1]=orig_01

    # plot
    fig=plt.figure(figsize=(10, 7))
    ax=fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, Z, cmap='plasma',alpha=0.9,edgecolor='none')

    # mark the trained weights on the surface
    ax.scatter([orig_00], [orig_01],
               [np.sqrt(np.mean((targets-net.test(inputs))**2))],
               color='cyan',s=80,zorder=5,label='Trained weights')

    ax.set_xlabel('wih[0,0]')
    ax.set_ylabel('wih[0,1]')
    ax.set_zlabel('RMS Error')
    ax.set_title('XOR Error Surface (two weights varied)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
#non extra credit part 1: XOR with backprop
def part1():
    inp = [[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]]

    xor_tgt = [[0],
               [1],
               [1],
               [0]]

    print("\n=========== Part-01: XOR with Backprop ============\n")
    print("Target: [0, 1, 1, 0]\n")

    net = Backprop(n=2, h=3, m=1)
    print(f"Before training: {net.test(inp).flatten().round(4)}\n")
    
    rmserr = net.train(inp, xor_tgt, iters=10000, eta=0.5)
    results = net.test(inp).flatten()
    print(f"\nAfter training:  {results.round(4)}\n")
    ploterr(rmserr, 'XOR')

    net.save("part1_weights.pkl")
    
    plot_error_surface(net, inp, xor_tgt) 

#===================================== pt2: 2 be or not 2 be=====================================
def part2():
    
    print("\n=========== Part-02: 2-be or not 2-be ============\n")

    train_inputs,train_targets=load_data('training.txt')
    test_inputs,test_targets=load_data('test.txt')

    # index 2 in one-hoT
    train_class=train_targets[:,2].reshape(-1,1)
    test_class=test_targets[:,2].reshape(-1,1)

    net=Backprop(n=196, m=1, h=25)
    print(net)

    # training information
    print("\nTraining with the following parameters: \n")
    print(f"  Hidden units : 25")
    print(f"  Iterations   : 3000")
    print(f"  Learning rate: 0.5")
    print(f"  Threshold    : 0.5\n")
    
    # rmserr=net.train(train_inputs,train_class,iters=3000,eta=0.5)
    # ploterr(rmserr,'2-not-2')
    # net.save("part2_weights.pkl")

    # load saved weights and test on test set
    net.load("part2_weights.pkl")
    
    test_out=net.test(test_inputs)
    report_results(test_out,test_class,threshold=0.5)

#=====================================p3=====================================

#EXTRA CREDIT
#debugging with claude and had to look up how to actually do this
#Extra credit part 3
def plot_confusion_3d(conf):
    
    fig=plt.figure(figsize=(10, 7))
    ax=fig.add_subplot(111, projection='3d')

    N=len(conf)
    xpos,ypos=np.meshgrid(range(N), range(N))
    xpos=xpos.flatten()
    ypos=ypos.flatten()
    zpos=np.zeros(N * N)

    dz=np.array(conf).flatten().astype(float)
    colors=['#00c49a' if x == y else '#ff3860'
              for x, y in zip(xpos, ypos)]

    ax.bar3d(xpos,ypos,zpos,0.7,0.7,dz,
             color=colors,alpha=0.85,shade=True)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_zlabel('Count')
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_title('3D Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    #non-extra credit part 3: full digit classifier
def part3():

    print("\n=========== Part-03: Full Digit Classifier ============\n")

    train_inputs,train_targets=load_data('training.txt')
    test_inputs,test_targets=load_data('test.txt')

    net=Backprop(n=196,m=10,h=250)
    print(net)

    print("\nTraining with the following parameters: \n")
    print(f"  Hidden units : 250")
    print(f"  Iterations   : 2000")
    print(f"  Learning rate: 0.5\n")

    # #TRAINING
    # rmserr=net.train(train_inputs,train_targets,iters=2000,eta=0.5)
    # ploterr(rmserr,'digits')
    # net.save("part3_weights.pkl")

    net.load("part3_weights.pkl")

    # winner takes all!
    test_out=net.test(test_inputs)
    predicted=np.argmax(test_out,axis=1)
    true_labels=np.argmax(test_targets,axis=1)

#==============================================================
    # build confusion matrix
    print("\nConfusion Matrix:\n")
    conf = np.zeros((10, 10), dtype=int)
    for actual, pred in zip(true_labels, predicted):
        conf[actual][pred] += 1

    # print pretty matrix
    print(f"{'':>4}", end="")
    for i in range(10):
        print(f"{i:>5}", end="")
    print()
    print("-" * 54)
    for i in range(10):
        print(f"{i:>3} |", end="")
        for j in range(10):
            print(f"{conf[i][j]:>5}", end="")
        print()
#==============================================================
    
    plot_confusion_3d(conf)


if __name__ == "__main__":

    part1()
    part2()
    part3()
    print("EXTRA 2,3 and 4 plots have been uploaded to the repository as 'XOR_error.png', '2-not-2_error.png', 'digits_error.png', 'confusion3d.png' and 'XOR_error_surface.png'")



    
    

