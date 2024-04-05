import torch
import torch.nn.functional as F
from e2cnn import gspaces                                          
from e2cnn import nn    
import math
import numbers
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import collections
import numpy as np

class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        self.kernel = kernel
        
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

    
def rescale(x, xref):
    xrefmin, xrefmax = xref.min(), xref.max()
    x = (x- x.min())/(x.max()-x.min())*(xrefmax-xrefmin)+xrefmin
    return x
    
    
class DigitsModel(torch.nn.Module):
    """
    Recognize single digits on input source pages. 
    See https://github.com/QUVA-Lab/e2cnn/ for detailed explantions of parameters.
    Arguments:
        N (int):  Number of *discrete* rotations (see . 
        kernel_sizes (int, list): Size of the kernels.
        kernels (str, list): Specifies the e2ccn kernel types,  
        'r_{n_kernels}' defines a regular equivariant convolution.
        'c_{n_kernels}' defines a standard convolution.
        paddings (int, list): Specifies the e2ccn kernel paddings.
        final_bias (int, list): Specifies the e2ccn kernel paddings.
        invariance_rots (int, list): Specifies the considered rotations. 
            Rotations of [-1,0,1] considers rotations of the input by [-90,0,90] degrees.
        clip_max (float, optional): Specifies the value to which activation should be clipped. Default is None.
    
    """
    
    def __init__(self,N, kernel_sizes, kernels, paddings, final_bias, invariance_rots = [0], clip_max=None):
        super(DigitsModel, self).__init__()
        self.N = N
        self.kernels = kernels
        self.conv_bias = False
        self.dropout = False
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.encoder = self.rotinv_conv_block()
        self.conv_encoder = self.conv_block()
        self.final_bias = final_bias
        self.invariance_rots = invariance_rots
        self.clip_max = clip_max
        
    def normalize(self, x):
        assert len(x.shape) == 3
        return x - torch.mean(x, dim=(1,2))
       
    
    def conv_block(self):
        
        remove_last =1
        layers = []
        for st, ks, p in zip(self.kernels, self.kernel_sizes, self.paddings):
            kt, s=st.split('_')[0], int(st.split('_')[1]) # kt: kernel group type, s: n_filters
            
            if kt == 'c':
                kernels_out = s
                layer = [torch.nn.Conv2d(kernels_in, kernels_out, kernel_size=ks, stride=1, padding=p, bias= self.conv_bias),
                           torch.nn.ReLU(True)]
                
                if self.dropout==True:
                    layer+= [torch.nn.Dropout(0.15)]
               
                layers += layer
                
            kernels_in = s

        layers = layers[:-remove_last]
        return torch.nn.Sequential(*layers)
         
    def rotinv_conv_block(self):
        
        layers = []
        self._in_type = self.in_type

        for st, ks, p in zip(self.kernels, self.kernel_sizes, self.paddings):

            kt, s=st.split('_')[0], int(st.split('_')[1]) # kt: kernel group type, s: n_filters
            
            if kt =='r':
                out_type = nn.FieldType(self.r2_act, s*[self.r2_act.regular_repr])
            elif kt =='t':
                out_type = nn.FieldType(self.r2_act, s*[self.r2_act.trivial_repr])
            elif kt == 'c':
                continue
                
            layer = [nn.R2Conv(self._in_type, out_type, kernel_size=ks, padding=p, bias= self.conv_bias),
                       nn.ReLU(out_type, inplace=True)]
            
            layers += layer
            self._in_type = out_type
            
        # Skip final relu
        layers = layers[:-1] +  [nn.GroupPooling(out_type)]    
        return nn.SequentialModule(*layers)

    def conv_forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)                            
        h = self.encoder(x).tensor
        h = self.conv_encoder(h)
        h = h - self.final_bias
        return h
    
    
    def rotational_invariance_block(self, x, return_inds = False):
        assert  len(self.invariance_rots) > 1
        
        H = []
        H_rots = []

        for rot in self.invariance_rots:
            # Rotate by rot
            xrot = torch.rot90(x, rot, [2,3])
            hrot = self.conv_forward(xrot)
            
            # Rotate back to original 
            xorig = torch.rot90(xrot, -rot, [2,3])
            assert (xorig==x).all()
            h = torch.rot90(hrot, -rot, [2,3])     
            H.append(h) # original orientation
            H_rots.append(hrot) # rotated orientation
            
        H = torch.stack(H)

        # Select rotation based in sum of activation
        _, inds = torch.max(H.sum(4).sum(3).sum(2),axis=(0))

        h = torch.stack([H[ind,i] for i, ind in enumerate(inds)])
        hrot = torch.stack([H_rots[ind][i] for i, ind in enumerate(inds)])
        
        if  return_inds:
            return h, hrot, inds
        else:
            return h
            

    def forward(self, x):
        """
        Compute single digit feature maps.
        Arguments:
            x (torch.float32):  Input data.
        """

        if len(self.invariance_rots) == 1:
            assert self.invariance_rots[0]==0
            h = self.conv_forward(x)
        else:
            h = self.rotational_invariance_block(x)
        return h
    
    def forward_copy(self,x):
        if len(self.invariance_rots) == 1:
            assert self.invariance_rots[0]==0
            h = self.conv_forward(x)
            inds = [0]
            hrot = h
        else:
            # h: in original rotation
            # hrot: after rotation by inds[0]
            # x: in original rotation
            h, hrot, inds = self.rotational_invariance_block(x, return_inds=True)
            
        return h, hrot, inds
    


class E2CNNCompositionMultiDigits(DigitsModel):
    """
    Recognize digit feature compositions, e.g. sinigle digits or bigrams, on input source pages. 
    See https://github.com/QUVA-Lab/e2cnn/ for detailed explantions of parameters.
    Arguments:
        N (int):  Number of *discrete* rotations (see . 
        kernel_sizes (int, list): Size of the kernels.
        kernels (str, list): Specifies the e2ccn kernel types,  
        'r_{n_kernels}' defines a regular equivariant convolution.
        'c_{n_kernels}' defines a standard convolution.
        paddings (int, list): Specifies the e2ccn kernel paddings.
        final_bias (int, list): Specifies the e2ccn kernel paddings.
        invariance_rots (int, list): Specifies the considered rotations. 
            Rotations of [-1,0,1] considers rotations of the input by [-90,0,90] degrees.
        clip_max (float, optional): Specifies the value to which activation should be clipped. Default is None.
    
    """
    
    def __init__(self, N, kernel_sizes, kernels, paddings, final_bias, invariance_rots, clip_max):
        super(E2CNNCompositionMultiDigits, self).__init__(N, kernel_sizes, kernels, paddings, final_bias, invariance_rots, clip_max)
      
    def single_digit(self,H,number, shift):
        # Detect single digit features 

        h1 =  H[0,number[0],:,:]
        h2 = h1
        l = h1            
        return l, (h1)
    
    def single_isolated_digit(self, H, number, shift):
        # Detect isolated single digit features 

        h1 = H[0,number[0],:,:]
        
        smoothing = GaussianSmoothing(1, 13, sigma=1.)
        smoothing.cuda()
          
        h1 = F.pad(h1.unsqueeze(0).unsqueeze(0), (6, 6, 6, 6), mode='reflect')
        h1 =smoothing(h1).squeeze()
        
        # Compute 0/1 mask of surround
        smoothing = GaussianSmoothing(1, 13, sigma=2.5)
        smoothing.cuda()

        kernel = smoothing.kernel
        Hproc = -H.sum(1).unsqueeze(1)

        Hproc_pad = F.pad(Hproc, (6, 6, 6, 6), mode='reflect')
        Hproc =smoothing(Hproc_pad)

        Hproc[Hproc>0.]=1.
        Hproc[Hproc<=0] = 0.

        # Rescale to fit statistics of original map
        Hproc = rescale(Hproc, h1)
        Hproc_smooth = Hproc

        # pad right to shift over center number
        h2 = self.m1_single(Hproc_smooth) # pad to shift second number over the first
        h2 = h2[0,:,:,self.shift_large:].squeeze()

        # pad left to shift over center number
        h3 = torch.flip(self.m1_single(torch.flip(Hproc_smooth, dims=[3])),dims=[3])
        h3 = h3[0,:,:,:-self.shift_large].squeeze()
      
        h23 = torch.min(h2,h3)

        hh = torch.stack((h1,h2,h3))
        l,_ = torch.min(hh, dim=0)

        return  l, (h1,h2,h3)
    
        
    def two_digit(self, H, number, shift):
        # Detect bigram digit features 

        h1 =  H[0,number[0],:,:]
        h2 = self.m1(H) # pad to shift second number over the first
        h2 = h2[0,number[1],:,shift:]
       
        l = torch.min(h1,h2)

        return l, (h1,h2)
    
    def three_digit(self, H, number, shift):
        # Detect 3-gram digit features 

        h1 =  H[0,number[0],:,:]
        h2 = self.m1(H) # pad to shift second number over the first
        h2 = h2[0,number[1],:,shift:]
        h3 = self.m2(H)
        h3 = h3[0,number[2],:,2*shift:]

        l = torch.min(h1,h2,h3)
            
        return l, (h1,h2,h3)
    
    def forward(self, x, numbers, shifts):
        """
        Compute digit composition feature maps.
        Arguments:
            x (torch.float32):  Input data.
            numbers (int, list): List of numerical bigram features, e.g [(0,0), (0,1)].
            shifts (int, list): Pixel shifts to consider for the detection of n-gram features, e.g [8,10].
        """
    
        if len(x.shape)==3:
            x=x.unsqueeze(0)

        self.x_current = x
        
        # H: optimal alignment (should be vertically aligned numbers)
        # Horig: Rotated back to its source orientation
        Horig, H, inds = self.forward_copy(x)
        
        self.raw_h_single_sum = H.sum()
        
        #Apply clipping
        if self.clip_max:
            Horig = torch.clamp(Horig, min=None, max=self.clip_max)
            H = torch.clamp(H, min=None, max=self.clip_max)        
        
        rots = [self.invariance_rots[ind] for ind in inds]
        self.rots=rots
        
        Htest = torch.rot90(H, -rots[0], [2,3])
        assert (Htest.squeeze() == Horig.squeeze()).all()
        
        Out = []
        Out_sum = []
        R = None
        
        shifts_chosen = []        
        all_shifts_chosen = []
        
        # Loop over numbers
        shift_dict = {i:shift_ for i, shift_ in enumerate(shifts)}
        
        # Here using the rotated H since we apply shifts on standard orientations
        for number in numbers:
            Out_shifts = []

            for shift in shifts:
                
                self.m1 = torch.nn.ZeroPad2d((0,shift,0,0))
                # Increase shift for isolated digits
                self.shift_large = int(np.clip(1.5*shift, a_min=0, a_max=15))
                
                self.m1_single = torch.nn.ZeroPad2d((0,self.shift_large,0,0))
                self.m2 = torch.nn.ZeroPad2d((0,2*shift,0,0))

                if len(number)==1:
                    out,_ = self.single_isolated_digit(H, number, shift)

                elif len(number)==2:
                    out,_ = self.two_digit(H, number, shift)

                elif len(number)==3:
                    out,_ = self.three_digit(H,number, shift)
                else:
                    raise NotImplementedError('N-grams supported for N<=3.')    
                    
                Out_shifts.append(out)
                
            out = torch.stack(Out_shifts, dim=0)
            self.out=out
            
            if len(out)>1:
                # Select shift 
                out_min, shift_inds = torch.max(out, dim=0)
                all_shifts_chosen.extend(shift_inds.detach().cpu().numpy().flatten().tolist())

            else:
                out_min = out
                self.shifts_chosen = [shift]
            
            out_sum = torch.sum(out_min)

            Out.append(out_min)    
            Out_sum.append(out_sum)
            
        if len(out)>1:
            shifts_chosen = {shift_dict[int(k)]:v for k,v in collections.Counter(all_shifts_chosen).items()}
            
        self.shifts_chosen =  shifts_chosen
        
        # Rotating back to original orientation
        assert len(rots)==1
        Out = torch.stack(Out)
        Out = torch.rot90(Out, -rots[0], [1,2])
        self.Out_sum = Out_sum
        
        return Horig, Out, Out_sum, R 
    


class TableModel(E2CNNCompositionMultiDigits):
    """
    Recognize digits on source pages over multiple image scales. 
    See https://github.com/QUVA-Lab/e2cnn/ for detailed explantions of parameters.
    Arguments:
        N (int):  Number of *discrete* rotations (see . 
        kernel_sizes (int, list): Size of the kernels.
        kernels (str, list): Specifies the e2ccn kernel types,  
        'r_{n_kernels}' defines a regular equivariant convolution.
        'c_{n_kernels}' defines a standard convolution.
        paddings (int, list): Specifies the e2ccn kernel paddings.
        final_bias (int, list): Specifies the e2ccn kernel paddings.
        invariance_rots (int, list): Specifies the considered rotations. 
            Rotations of [-1,0,1] considers rotations of the input by [-90,0,90] degrees.
        clip_max (float, optional): Specifies the value to which activation should be clipped. Default is None.
    
    """
    
    def __init__(self, N, kernel_sizes, kernels, paddings, final_bias, invariance_rots, clip_max=None):
        super(TableModel, self).__init__(N, kernel_sizes, kernels, paddings, final_bias, invariance_rots, clip_max)
    
    def compute_forward_scales(self, x, numbers, shift, remove_mean, scale_sizes, scales, shift_func=None):

        """
        Compute invariant bigram feature representations over multiple image scales.
        Arguments:
            x (torch.float32):  Input data.
            numbers (int, list): List of numerical bigram features, e.g [(0,0), (0,1)].
            shift (int, list): Pixel shifts to consider for the detection of n-gram features, e.g [8,10].
            remove_mean (bool): Specifies if channels of input x should be centered.
            scale_sizes (int, list): Specifies the reference pixel height/width of each scale for input x.
            scales (float, list): Specifies the fractions of image scales considered, e.g. [0.8, 0.9, 1.0].
            shift_func (function, optional): Specifies a function to assign specific bigram shifts.
        """
        
        H = []
        H_compositions_ = []
        scale_sizes_hw = []
        self.hs_before = []
        X_scales = []
        
        if not isinstance(x,list):
            x = [x]*len(scale_sizes)
        
        self.Out_shifts = []        
        self.rot_scales = []
        self.decide_scale_sums = []
        
        shift_chosen_scales = []
        for i,refsize in enumerate(scale_sizes):
            
            float_scale = scales[i]
            x_scale = x[i]
            
            if len(x_scale.shape) ==4 and x_scale.shape[0]==1:
                x_scale = x_scale.squeeze(0)
        
            if remove_mean == True:
                x_scale = self.normalize(x_scale).unsqueeze(0)
            else:
                x_scale = x_scale[0].unsqueeze(0)
    
            if shift_func is not None:
                shift_new = shift_func(float_scale)
                print('Changing shift',shift, shift_new, float_scale)
                shift = shift_new 
        
            h, h_compositions, _, _  = self.forward(x_scale, numbers, shift) # h1.shape: ([1, 10, 1200, 800]), len(h1_pool): [n_numbers]
            
            H_compositions_.append(h_compositions)
            X_scales.append(x_scale)
            H.append(h)
            scale_sizes_hw.append((tuple(x_scale.squeeze().shape), float_scale))
            
            self.decide_scale_sums.append(self.raw_h_single_sum)
            self.rot_scales.append((refsize, self.rots, h.sum() ,h, h_compositions.squeeze().sum(0).detach().cpu().numpy()))
            shift_chosen_scales.append(self.shifts_chosen)
         
   
        # Use single digit as decision for maximal scale
        self.H_ = H

        sums = torch.stack([h_.sum() for h_ in H])   
        raw_sums = torch.stack(self.decide_scale_sums)
        idx_max = torch.argmax(raw_sums)

        h_compositions = H_compositions_[idx_max].squeeze().detach().cpu()
        self.x_scale_chosen = X_scales[idx_max].squeeze().detach().cpu().numpy(),
        self.scale_chosen = list(scale_sizes_hw[idx_max])

        self.all_shift_scales =  shift_chosen_scales
        self.shift_chosen_scales =  shift_chosen_scales[idx_max]
        self.chosen_params = (list(scale_sizes_hw[idx_max]),
                              self.rot_scales[idx_max][1], raw_sums.detach().cpu().numpy())

        h = H[idx_max]

        h_pool = None
        return  h, h_compositions, h_pool
    
