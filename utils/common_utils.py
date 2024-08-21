import torch
import torch.nn as nn
import logging
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

logger = logging.getLogger('main')


def get_logger(    
        LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        LOG_NAME       = '',
        LOG_FILE_INFO  = 'file.log',
        LOG_FILE_DEBUG = 'file.debug'):

    log           = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.WARNING)
    log.addHandler(stream_handler)

    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='w')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_debug = logging.FileHandler(LOG_FILE_DEBUG, mode='w')
    file_handler_debug.setFormatter(log_formatter)
    file_handler_debug.setLevel(logging.DEBUG)
    log.addHandler(file_handler_debug)

    log.setLevel(logging.DEBUG)

    return log

def send_email(to: list, subject: str, body: str='', files: list=[]):
    import smtplib
    from os.path import basename
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    assert isinstance(to, list)

    msg = MIMEMultipart()
    fromaddr = "hiyouhave1newmessage@gmail.com"
    msg['From'] = fromaddr
    msg['To'] = COMMASPACE.join(to)
    # msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(body))

    for f in files:
        with open(f, 'rb') as file:
            part = MIMEApplication(file.read(), Name=basename(f))
        
        part['Content-Disposition'] = f'attachment; filename="{basename(f)}"'
        msg.attach(part)

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(fromaddr, "hwbs yvro qwch fzcd")
    server.sendmail(fromaddr, to, msg.as_string())
    server.close()

def add_hook_feature_maps(model):
    model.feature_maps = {}

    def get_activation(layer_name):
        def hook(module, input, output):
            model.feature_maps[layer_name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if 'conv' in name and isinstance(module, nn.Conv2d):
            # logger.debug('Counting zero filters (from hooks): %s/%s', torch.sum(torch.all(module.weight == 0, dim=(0,2,3))).item(), module.weight.shape[1])
            module.register_forward_hook(get_activation(name))

def plot_feature_maps(path: str, feature_maps: dict):
    """
    Plot all feature maps from a model to a single file, grouped by layer (key of the feature_maps dict).

    Args:
        path: path to the file where the feature maps will be saved
        feature_maps: dictionary with layer names as keys and tensors of feature maps as values
    """
    logger.info(f"Plotting feature maps to {path}... this may take a while")

    for layer_name, fms in feature_maps.items():
        num_fms = fms.size(1)  # Number of feature maps in this layer
        num_cols = 8  # Number of columns for plotting
        num_rows = (num_fms + num_cols - 1) // num_cols  # Number of rows required

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i in range(num_fms):
            ax = axes[i]
            ax.imshow(fms[0, i].cpu().numpy())
            ax.axis('off')
            ax.set_title(f'{layer_name} map {i+1}')

        # Turn off any remaining unused axes
        for i in range(num_fms, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'{path}_{layer_name}.png')
        plt.close(fig)

        logger.debug(f"Saved feature maps for layer {layer_name}")

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def crop_image_to_square(img, d=32):
    '''Crop image to make dimensions divisible by `d` and form a square'''

    # Make dimensions divisible by `d`
    new_width = img.size[0] - img.size[0] % d
    new_height = img.size[1] - img.size[1] % d

    # Determine size for the square
    square_size = min(new_width, new_height)

    # Calculate the bounding box for the square
    bbox = [
        int((img.size[0] - square_size) / 2), 
        int((img.size[1] - square_size) / 2),
        int((img.size[0] + square_size) / 2),
        int((img.size[1] + square_size) / 2)
    ]

    # Crop the image
    img_cropped = img.crop(bbox)
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

# def get_image_grid(images_np, nrow=8):
#     '''Creates a grid from a list of images by concatenating them.'''
#     images_torch = [torch.from_numpy(x) for x in images_np]
#     torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
#     return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos'):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10, dtype=torch.float32):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape, dtype=dtype)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(
            spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if len(img_np.shape) == 2:  # Grayscale image, no need to transpose
        ar = ar
    elif len(img_np.shape) == 3:  # Color image, need to transpose
        ar = ar.transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected number of array dimensions: {len(img_np.shape)}")

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


