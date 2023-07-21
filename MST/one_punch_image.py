from libraries import *
# from utilities import load_img, imshow
from abs_style_factory import Abstract_Style_Factory
from abs_product import *

class ConcreteStyleFactory(Abstract_Style_Factory):
    def get_style(self,artist:str, augment:bool):
        if artist == "vangogh":
            # img_dir = '/Users/rudra_sarkar/Documents/Mtech Second Sem/Deep Learning/NST/vangogh'
            rep = Vangogh().createRepresentation(augment=augment)
           
        elif artist == "picasso":
            # img_dir = '/Users/rudra_sarkar/Documents/Mtech Second Sem/Deep Learning/NST/picasso'
            rep = Picasso().createRepresentation(augment=augment)
            
        else :
            raise NameError("Name not found!")
        return rep
    def decorate(self, artist:str, augemnt:bool, rep):
        d_rep = self.get_style(artist, augemnt)
        new_images = [d_rep, rep]
        new_images = np.array(new_images)
        new_images = new_images.reshape(len(new_images), 512*512*3)
        new_images = new_images.T
        u, sigma, v = randomized_svd(new_images, n_components = 1)

        rep = (u.dot(sigma))
        artist_rep = rep.reshape(1,512, 512, 3)*0.8
        return artist_rep

