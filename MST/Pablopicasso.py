from libraries import *
from load_image import load_img


def get_picasso_style(augment:bool):
    img_dir = '/Users/rudra_sarkar/Documents/Mtech Second Sem/Deep Learning/NST/picasso'
    
    dir_images = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
    dir_images.remove(dir_images[1])
    
    augment_images = []

    train_images = [np.array(load_img(fname)).reshape(512, 512, 3) for fname in dir_images]
    if augment:
        for images in train_images:
            theta = np.random.randint(0,360)
            shear = np.random.rand()
            img = ImageDataGenerator().apply_transform(images, transform_parameters={'theta':theta, "shear":shear})
            augment_images.append(img)
        train_images = [0.2*images for images in train_images]
        train_images = train_images + augment_images


    train_images = np.array(train_images)

    train_images = train_images.reshape(len(train_images), 512*512*3)
    train_images = train_images.T
    u, sigma, v = randomized_svd(train_images, n_components = 1)

    rep = (u.dot(sigma))
    artist_rep = rep.reshape(512, 512, 3)*0.8
  
    return artist_rep

print(get_picasso_style(True))
