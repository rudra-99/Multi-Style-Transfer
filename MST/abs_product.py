from libraries import *
from utilities import load_img
class AbsProduct:
    def imageList(self,img_dir:str):
        pass
    def createRepresentation(self):
        pass

class IConcreteproduct(AbsProduct):
    def __init__(self):
        self.dir_images = None
    def imageList(self,img_dir: str):
        dir_images = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
        train_images = [np.array(load_img(fname, 'style')).reshape(512, 512, 3) for fname in dir_images]
        return train_images

    def createRepresentation(self,img_dir):
        
        self.dir_images = self.imageList(img_dir)
        return self.dir_images
        

class Augment(IConcreteproduct):
    def createRepresentation(self, img_dir):
        augment_images = []
        train_images =  super().createRepresentation(img_dir)
        for images in train_images:
                theta = np.random.randint(0,360)
                shear = np.random.rand()*np.random.randint(0,2)
                img = ImageDataGenerator().apply_transform(images, transform_parameters={'theta':theta, "shear":shear})
                augment_images.append(img)
        train_images = [0.2*images for images in train_images]
        train_images = train_images + augment_images
        train_images = np.array(train_images)
        return train_images
        

class Normal(IConcreteproduct):
    def createRepresentation(self):
        train_images =  super().createRepresentation()
        return np.array(train_images)

class Vangogh(IConcreteproduct):
    def imageList(self, img_dir: str):
        return super().imageList(img_dir)
    def createRepresentation(self, augment:bool):
        img_dir = './vangogh'
        if augment:
            self.dir_images = Augment().createRepresentation(img_dir)
        else:
            self.dir_images = Normal().createRepresentation(img_dir)
        self.train_images = self.dir_images.reshape(len(self.dir_images), 512*512*3)
        self.train_images = self.train_images.T
        u, sigma, v = randomized_svd(self.train_images, n_components = 1)

        self.rep = (u.dot(sigma))
        self.artist_rep = self.rep.reshape(1,512, 512, 3)*0.8
        return self.artist_rep

class Picasso(IConcreteproduct):
    def imageList(self, img_dir: str):
        return super().imageList(img_dir)
    def createRepresentation(self, augment:bool):
        img_dir = './picasso'
        if augment:
            self.dir_images = Augment().createRepresentation(img_dir)
        else:
            self.dir_images = Normal().createRepresentation(img_dir)
        self.train_images = self.dir_images.reshape(len(self.dir_images), 512*512*3)
        self.train_images = self.train_images.T
        u, sigma, v = randomized_svd(self.train_images, n_components = 1)

        self.rep = (u.dot(sigma))
        self.artist_rep = self.rep.reshape(1,512, 512, 3)*0.8
        return self.artist_rep

