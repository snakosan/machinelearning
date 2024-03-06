import mnist
import matplotlib.pyplot as plt 
import numpy as np
images = mnist.train_images()

def sigmoid(input):
    return 1/(1+np.exp(-1*input))
    
class Agent():
    def __init__(self, image=None, num_blocks=4):
        self.image = image
        self.num_blocks = num_blocks
        self.reduced_im = np.zeros((self.num_blocks, self.num_blocks))
        self.weights = np.random.rand(self.num_blocks, self.num_blocks)
        
    def reduce(self):
        N = int(self.image.shape[0]/self.num_blocks)
        # reduce the size of the image to block_size*block_size down from 28.28
        for x in range(self.num_blocks):
            for y in range(self.num_blocks):
                new = self.image[x * N : (x + 1) * N, y * N : (y + 1) * N]
                self.reduced_im[x][y] = np.sum(new)

    def dot(self):
        return int(np.sum(self.weights * self.reduced_im))

    def flatten(self):
        return np.reshape(self.reduced_im, (1,self.reduced_im.size))

    def forward(self):
        
        
    def plot(self):
        plt.imshow(self.reduced_im)


class Classifier():
    def __init__(self):
        pass 

    def fit(x):
        pass 


def main():
    image = images[0]
    num_blocks = 4
    block_size = int(image.shape[0]/num_blocks)
    agent = Agent(image, num_blocks)
    agent.reduce()
    reduced = agent.reduced_im
    print(agent.flatten())
    agent.plot()




if __name__ == "__main__":
    main()
