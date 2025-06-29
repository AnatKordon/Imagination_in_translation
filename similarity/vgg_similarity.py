# citing:
# Shoham, A., Grosbard, I. D., Patashnik, O., Cohen-Or, D., & Yovel, G. (2024). Using deep neural networks to disentangle visual and semantic information in human perception and memory. Nature Human Behaviour, 8(4), 702-717.:
# Simonyan, K. & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings (2014) doi:10.48550/arxiv.1409.1556.
# Deng, J. et al. ImageNet: A large-scale hierarchical image database. in 2009 IEEE Conference on Computer Vision and Pattern Recognition 248–255 (IEEE, 2009). doi:10.1109/CVPR.2009.5206848.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from scipy.spatial.distance import cosine
import torch
import torchvision
import torchvision.models
from torch.utils import model_zoo as model_zoo
from torchvision import transforms
from PIL import Image
from Pathlib import Path

#change to create a class: VGG_Feature_Extractor

# load a pretrained VGG net trained on Imagenet
model_imagenet = torchvision.models.vgg16(pretrained=True)
###chagne to have a model

#  define a function that will retrieve the embeddings of a given layer
def get_embeddings_by_layer(layer_name):
    def hook(model, input, output):
        this_embedding = output.detach()
        this_embedding = this_embedding.flatten().cpu().numpy()
        embeddings[layer_name] = this_embedding
    return hook

# go over all of the layers of the network and set a forward hook to the 'get_embeddings_by_layer' function
for idx,layer in enumerate(model_imagenet.features):
    layer.register_forward_hook(get_embeddings_by_layer('Layer_'+ str(idx)))

for idx,layer in enumerate(model_imagenet.classifier):
    layer.register_forward_hook(get_embeddings_by_layer('Classifier_'+ str(idx)))

# define the preprocessing pipline for the images (we did not touch this - according to the code from git we are using)
val_transforms = transforms.Compose([
                                     transforms.Resize((224, 224)), # this will ensure all images and outputs are at the same size
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                    mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])])

def get_embeddings_imgset(model, imgs_fld, img_fns, layers):
  embeddings_per_layer = {}
  # iterate over files in
  # that directory
  for filename in img_fns:
      f = os.path.join(imgs_fld, filename)
      # checking if it is a file
      if os.path.isfile(f):
        # open the file as image
        img = Image.open(f).convert('RGB')
        # preprosses the image
        prep_img = torch.unsqueeze(val_transforms(img), 0)
        with torch.no_grad():
          output = model(prep_img)
          for l in layers:
            if l not in embeddings_per_layer:
              embeddings_per_layer[l] = []
            embeddings_per_layer[l].append(embeddings[l])
  return(embeddings_per_layer)

def get_images_embeddings(model_imagenet, layers, images_folder):
  # assign directories
  images_path = images_folder
  # get files from directory
  # images_fns = [fn for fn in os.listdir(images_path) if  fn.endswith(img_ext)]
  images_fns = [fn for fn in os.listdir(images_path)]
  results = {}
  results['images'] = get_embeddings_imgset(model_imagenet, images_path, images_fns, layers)
  results['images_fns'] = images_fns
  return(results)

def save_embeddings(embeddings_of_al_layers, file_name):
  filenames = embeddings_of_al_layers['images_fns']

  for layer_name, embeddings_list in embeddings_of_al_layers['images'].items():
      # Convert numpy arrays to lists
      embeddings_as_lists = [emb.tolist() for emb in embeddings_list]

      # Create DataFrame with embeddings
      df = pd.DataFrame(embeddings_as_lists)
      df.columns = [f'dim_{i}' for i in range(df.shape[1])]

      # Insert filenames as first column
      df.insert(0, 'filename', filenames)

      # Save CSV named by layer
      csv_filename = output_sum_rdm_root+ file_name + '_' + layer_name + '_embeddings' + '.csv'
      df.to_csv(csv_filename, index=False)

def print_rdm(df, layer, file_name):
  # Get all unique image names
  images = sorted(set(df['img1']).union(df['img2']))

  # Initialize a square distance matrix with NaNs
  distance_matrix = pd.DataFrame(index=images, columns=images, dtype=float)

  # Fill in the known cosine distances
  for _, row in df.iterrows():
      i, j, dist = row['img1'], row['img2'], row['cosine_distance']
      distance_matrix.loc[i, j] = dist
      distance_matrix.loc[j, i] = dist  # Ensure symmetry

  # Set diagonal to 0 (distance to self)
  for img in images:
      distance_matrix.loc[img, img] = 0

  plot_title = file_name + ' ' + layer

  # Plot
  sns.heatmap(distance_matrix, annot=True, cmap="plasma", vmin=0, vmax=1, square=True)
  plt.title(plot_title + " RDM")
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()


def vgg_similarity(org_img: str, gen_img, model, log_path) #how to write names properly and use Path??
    



layers = ['Layer_30','Classifier_4']
