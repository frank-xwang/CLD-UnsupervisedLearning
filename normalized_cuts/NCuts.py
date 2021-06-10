
'''
reference:
[1] J. Shi and J. Malik, “Normalized cuts and image segmentation”, 
    IEEE Trans. on Pattern Analysisand Machine Intelligence, Vol 22
[2] https://github.com/SatyabratSrikumar/Normalized-Cuts-and-Image-Segmentation-Matlab-Implementation
edited by JY Wang @ 2018-6-3 SYR Unv
'''
 
import cv2
import numpy as np
from scipy.linalg.decomp import eig
from scipy import sparse
from scipy.sparse.linalg import eigs
from skimage import graph, data, io, segmentation, color
from skimage.transform import rescale, resize

class Ncut(object):
    '''
    This class is write for RGB image, so if you want to processing grayscale, some adjustment should worked on 
    F_maker, W_maker function :)
    '''
 
    def __init__(self, img):
        '''
        :param img: better no larger than 300px,300px
        '''
        self.no_rows, self.no_cols, self.channel = img.shape
        self.N = self.no_rows * self.no_cols
        self.V_nodes = self.V_node_maker(img)
        self.X = self.X_maker()
        self.F = self.F_maker(img)
        # parameter for W clculate
        self.r = 2
        self.sigma_I = 4
        self.sigma_X = 6
        # Dense W,D
        self.W = self.W_maker()
        self.D = self.D_maker()
 
    # V_nodes shape : [self.N,1,3]
    def V_node_maker(self, img):
        b,g,r = cv2.split(img)
        b = b.flatten()
        g = g.flatten()
        r = r.flatten()
        V_nodes = np.vstack((b,g))
        V_nodes = np.vstack((V_nodes,r))
        return V_nodes
 
    def X_maker(self):
        X_temp = np.arange(self.N)
        X_temp = X_temp.reshape((self.no_rows,self.no_cols))
        X_temp_rows = X_temp // self.no_rows
        X_temp_cols = (X_temp // self.no_cols).T
        X = np.zeros((self.N, 1, 2))
        X[:,:,0] = X_temp_rows.reshape(self.N,1)
        X[:,:,1] = X_temp_cols.reshape(self.N,1)
        return X
 
    def F_maker(self,img):
        if self.channel < 2:
            return self.gray_feature_maker(img)
        else:
            return self.color_img_feature_maker(img)
 
    def gray_feature_maker(self,img):
        print('need to ')
 
    def color_img_feature_maker(self,img):
        F = img.flatten().reshape((self.N,1,self.channel))
        F = F.astype('uint8')
        return F
 
    def W_maker(self):
        X = self.X.repeat(self.N,axis = 1)
        X_T = self.X.reshape((1,self.N,2)).repeat(self.N,axis = 0)
        diff_X = X - X_T
        diff_X = diff_X[:,:,0]**2 + diff_X[:,:,1]**2
 
        F = self.F.repeat(self.N,axis = 1)
        F_T = self.F.reshape((1,self.N,3)).repeat(self.N,axis = 0)
        diff_F = F - F_T
        diff_F = diff_F[:,:,0]**2 + diff_F[:,:,1]**2 + diff_F[:,:,2]**2
 
        W_map = diff_X < self.r**2 # valid map for W
 
        W = np.exp(-((diff_F / (self.sigma_I**2)) + (diff_X / (self.sigma_X**2))))
 
        return W * W_map 
 
    def D_maker(self):
        # D is a diagonal matrix using di as diagonal, di is the sum of weight of node i with all other nodes
        d_i = np.sum(self.W, axis=1)
        D = np.diag(d_i)
        return D
 
    def EigenSolver(self):
        L = self.D - self.W
        R = self.D
        lam,y = eig(L, R)
        index = np.argsort(lam)
 
        top2 = lam[index[:2]].real
        smallest_2 = y[:,index[1]]
        print('dense eigenvector: {} with shape of {}'.format(smallest_2,smallest_2.shape))
        return smallest_2.real
 
    def EigenSolver_sparse(self):
        s_D = sparse.csr_matrix(self.D)
        s_W = sparse.csr_matrix(self.W)
        s_D_nhalf = np.sqrt(s_D).power(-1)
        L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
        lam,y = eigs(L)
        index = np.argsort(lam)
 
        top2 = lam[index[:2]].real
        smallest_2 = y[:, index[1]]
        print('sparse eigenvector: {} with shape of {}'.format(smallest_2,smallest_2.shape))
        return smallest_2.real
 
 
if __name__ == '__main__':
    # This is dense eigenvector method
    # img = cv2.imread('picture/Ncut_test.png', cv2.IMREAD_COLOR)
    # cutter = Ncut(img)
    # eigenvector = cutter.EigenSolver()

    # the process is cost too much time, so I saved the results in a txt file, just ignore this part if you need't
    
    # file = open('result.txt','w')
    # for i in eigenvector:
    #     file.write(str(i))
    #     file.write(',')
    # file = open('result.txt', 'r')
    # a = file.read()
    # b = np.array(a.split(','))
 
    # This is sparse eigenvector method
    img = cv2.imread('Ncut_test.png', cv2.IMREAD_COLOR)
    # img = data.coffee()
    # img = resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
    print('Loaded image with shape of: {}'.format(img.shape))
    cutter = Ncut(img)
    eigenvector = cutter.EigenSolver_sparse()
    b = eigenvector
    b = b.reshape((img.shape[0],img.shape[1])).astype('float64')
    b = (b/b.max())*255
    cv2.imshow('eigvec',b.astype('uint8'))
    cv2.waitKey()
    print('Finished!')
