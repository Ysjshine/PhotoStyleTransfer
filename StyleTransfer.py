import tensorflow as tf 
import numpy as np 
import cv2
import random
import copy

class PhotoStyleTransfer():
    def __init__(self,content,style):
        self.content = content
        self.style = style
        self.height = content.shape[0]
        self.width = content.shape[1]

        #resize the image 
        self.HEIHT = 224
        self.WIDTH = 224

    
    def weight_and_bias_init(self,w):
        weight = tf.Variable(w[0],trainable=False)
        bias = tf.Variable(w[1],trainable=False)
        return (weight,bias)
    
    def base_conv(self,tensor,wb):
        weight,bias = self.weight_and_bias_init(wb)
        conv = tf.nn.conv2d(tensor,weight,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv,bias))
        return relu

    def base_pool(self,tensor):
        pool =  tf.nn.max_pool(tensor,[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        return pool

    
    def conv_nn(self,input,model):
        relu1_1 = self.base_conv(input,model[b'conv1_1'])
        relu1_2 = self.base_conv(relu1_1,model[b'conv1_2'])
        pool1 = self.base_pool(relu1_2)

        relu2_1 = self.base_conv(pool1,model[b'conv2_1'])
        relu2_2 = self.base_conv(relu2_1,model[b'conv2_2'])
        pool2 = self.base_pool(relu2_2)

        relu3_1 = self.base_conv(pool2,model[b'conv3_1'])
        relu3_2 = self.base_conv(relu3_1,model[b'conv3_2'])
        relu3_3 = self.base_conv(relu3_2,model[b'conv3_3'])
        pool3 = self.base_pool(relu3_3)

        relu4_1 = self.base_conv(pool3,model[b'conv4_1'])
        relu4_2 = self.base_conv(relu4_1,model[b'conv4_2'])
        relu4_3 = self.base_conv(relu4_2,model[b'conv4_3'])
        pool4 = self.base_pool(relu4_3)

        relu5_1 = self.base_conv(pool4,model[b'conv5_1'])
        # relu5_2 = self.base_conv(relu5_1,model[b'conv5_2'])
        # relu5_3 = self.base_conv(relu5_2,model[b'conv5_3'])
        #pool5 = self.base_conv(relu5_2)
        result = [relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
        return result

    def gram_matrix(self,tensor,width):
        tensor = tf.reshape(tensor,(-1,width))
        tensor1 = tf.transpose(tensor)
        gram = tf.matmul(tensor1,tensor)
        return gram

    def target(self):
        target = copy.deepcopy(self.content)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if random.randint(0,1000)%29<1:
                    target[i,j] = target[i,j]/1.5
        img = cv2.GaussianBlur(target,(3,3),0)        
        return img

    def resize(self):
        target = cv2.resize(self.target(),(self.HEIHT,self.WIDTH))
        content = cv2.resize(self.content,(self.HEIHT,self.WIDTH))
        style = cv2.resize(self.style,(self.HEIHT,self.WIDTH))

        target = tf.Variable(target,dtype=tf.float32,trainable=True)
        content = tf.Variable(content,dtype=tf.float32,trainable=False)
        style = tf.Variable(style,dtype=tf.float32,trainable=False)

        return (target,content,style)

    
    def content_loss(self,content,target):
        loss = tf.reduce_mean(tf.square(content-target))
        return loss

    def style_loss(self,style,target):
        loss = 0.0
        for i in range(len(target)):
            shape = target[i].get_shape()
            H = shape[1].value
            W = shape[2].value
            C = shape[3].value
            g1 = self.gram_matrix(target[i],C)
            g2 = self.gram_matrix(style[i],C)
            sub = C*H*W
            loss += tf.reduce_sum(tf.square((g1/sub-g2/sub)))
        return loss

    def train(self,step):
        vgg = np.load("./vgg16.npy",encoding="bytes").item()
        src = self.resize()
        content_result = self.conv_nn(tf.reshape(src[1],[1,self.HEIHT,self.WIDTH,3]),vgg)
        style_result = self.conv_nn(tf.reshape(src[2],[1,self.HEIHT,self.WIDTH,3]),vgg)
        target_result = self.conv_nn(tf.reshape(src[0],[1,self.HEIHT,self.WIDTH,3]),vgg)

        cont_loss = self.content_loss(content_result[4],target_result[4])
        sty_loss = self.style_loss(style_result,target_result)

        loss = 1000*cont_loss+sty_loss
        train_step = tf.train.AdamOptimizer(3.0).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("-------------------begin to train--------------------")
            for i in range(step):
                print("step "+str(i)+":  "+str(sess.run(loss)))
                sess.run(train_step)
            pic = sess.run(src[0])
            cv2.imwrite("./out.png",cv2.resize(pic,(self.width,self.height)))
            print("----------------------end-------------------------------")
            sess.close()

if __name__ == '__main__':
    content = cv2.imread("./content.png")
    style = cv2.imread("./style.png")
    photo = PhotoStyleTransfer(content,style)
    photo.train(100)
    cv2.imshow("content",content)
    cv2.imshow("style",style)   
    cv2.imshow("output",cv2.imread("./out.png"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
