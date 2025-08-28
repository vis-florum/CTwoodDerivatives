from .imgfunctions import *
import logging

class BodyOfProps:
    '''Class to store the body of properties of a sample'''
    def __init__(self, filepath: str, slices=None):
        self.rootpath = os.path.dirname(os.path.abspath(filepath))
        self.filename = os.path.basename(filepath)
        self.fileID = self.filename.split(".")[0]
        self.filepath = filepath

        itkimg = sitk.ReadImage(self.filepath)  # is it numpy?
        if slices is not None:
            extract = sitk.ExtractImageFilter()
            extract.SetIndex([0, 0, slices[0]])
            extract.SetSize([itkimg.GetWidth(), itkimg.GetHeight(), slices[1]-slices[0]])
            itkimg = extract.Execute(itkimg)
        self.img = sitk.GetArrayFromImage(itkimg)
        self.shape = self.img.shape

        # Dimensional scaling
        self.vox2mm = itkimg.GetSpacing()
        self.vox2m = (self.vox2mm[0] / 1000, self.vox2mm[1] / 1000, self.vox2mm[2] / 1000) # ... etc.

        self.Z = np.arange(start=0, stop=self.shape[0]*self.vox2m[0], step=self.vox2m[0])
        self.Y = np.arange(start=0, stop=self.shape[1]*self.vox2m[1], step=self.vox2m[1])
        self.X = np.arange(start=0, stop=self.shape[2]*self.vox2m[2], step=self.vox2m[2])
        if slices is not None:
            self.Z = np.arange(start=slices[0]*self.vox2m[0], stop=slices[1]*self.vox2m[0], step=self.vox2m[0])
        
        # Labels and masks:
        self.mask_wood = None
        self.labels_ewlw = None
        self.labels_knots = None

        # Tensor fields:
        self.dirR = None
        self.dirT = None
        self.dirL = None
        self.C = None   # Stiffness tensor
        self.density = getDensity(self.img)
        
        
        # Statistics:
        self.t_w = None   # threshold for wood
        self.rho_mean = None
        self.rho_median = None
        self.rho_std = None
        self.r_lw = None    # ratio of LW in wood
        self.L_prop = None  # proportion of L along object / beam axis
        #self.mode_ew = None
        #self.mode_lw = None

    def coord2vox(self, z, y, x):
        '''Converts coordinates in mm to voxel coordinates'''
        return [int((z - self.Z[0])/self.vox2m[0]), int(y/self.vox2m[1]), int(x/self.vox2m[2])]
    
    def vox2coord(self, k, j, i):
        '''Converts voxel coordinates to coordinates in mm'''
        return [k*self.vox2m[0] + self.Z[0], j*self.vox2m[1], i*self.vox2m[2]]

    def findSegments(self):
        '''Create segments and labels in the object'''
        logging.debug("  Segmenting air... ")
        self.mask_wood, self.t_w, self.rho_mean, self.rho_median, self.rho_std = segmentAir(self.img)
        logging.debug("  Segmenting early-/late-wood... ")
        self.labels_ewlw, self.r_ew = findEWLW(self.img, self.mask_wood)
        #self.labels_knots = findKnots(self.img, self.mask_wood)    # TODO: implement findKnots
        

    def findFibres(self):
        '''Fibre estimations in the object'''
        logging.debug("Finding fibre coordinate system... ")
        self.dirR, self.dirT, self.dirL = getFCS(self.img, sigma=0.7, omega=1.5)
        logging.debug("Finding fibre alignment... ")
        self.L_prop = getFibreAlignment(self.dirL, self.mask_wood)


    def exportVTK(self, outdir=None, slices=None):
        '''Export the body of properties to a VTK file'''
        if outdir is None:
            filepath = os.path.join(self.rootpath, self.fileID + "_bop")
        else:
            filepath = os.path.join(outdir, self.fileID + "_bop")

        createVTK(filepath, self.img, self.mask_wood, self.labels_ewlw, self.dirR, self.dirT, self.dirL, slices=slices)

    # Orientation field, NB: Follow order L-R-T
    def FCS_intp(self, point):

        # so far only nearest neighbour...
        intpPt = self.coord2vox(*point)
        #lin_idx = np.ravel_multi_index(intpPt, BOP.shape, order='F')

        return self.dirL[intpPt], self.dirR[intpPt], self.dirT[intpPt] 
        
        
