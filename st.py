import math

import cv2
import numpy as np
import nrrd
from scipy.interpolate import RegularGridInterpolator

class ST(object):

    # assumes image is a floating-point numpy array
    def __init__(self, image):
        self.image = image
        self.lambda_u = None
        self.lambda_v = None
        self.vector_u = None
        self.vector_v = None
        self.isotropy = None
        self.linearity = None
        self.coherence = None
        # self.vector_u_interpolator_ = None
        # self.vector_v_interpolator_ = None
        self.lambda_u_interpolator_ = None
        self.lambda_v_interpolator_ = None
        self.vector_u_interpolator_ = None
        self.vector_v_interpolator_ = None
        self.grad_interpolator_ = None
        self.isotropy_interpolator_ = None
        self.linearity_interpolator_ = None
        self.coherence_interpolator_ = None

    def saveImage(self, fname):
        timage = (self.image*65535).astype(np.uint16)
        cv2.imwrite(str(fname), timage)

    # Interpolate vectors that may have an ambiguous sign
    # (such as ST eigenvectors).  When interpolating, make sure
    # that adjacent vectors are consistently aligned.
    def createVectorInterpolator(ar):

        arh = ar.copy()
        arh[arh[:,:,0]<0,:] *= -1
        arv = ar.copy()
        arv[arv[:,:,1]<0,:] *= -1
        # print(arh[499:501,469:471])
        # print(arv[499:501,469:471])
        hinterp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), arh, method='linear', bounds_error=False, fill_value=0.)
        vinterp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), arv, method='linear', bounds_error=False, fill_value=0.)
        def vectorInterpolator(pts):
            # print(pts.shape)
            rh = hinterp(pts)
            dh = (rh*rh).sum(axis=-1)
            rv = vinterp(pts)
            dv = (rv*rv).sum(axis=-1)
            ro = rh.copy()
            # print(ro.shape, dv.shape, (dv>dh).shape, ro[dv>dh].shape)
            b = (dv>dh)
            '''
            if b.sum() > 0:
                # print(b)
                # print(b0)
                print(rh[b][0])
                print(dh[b][0])
                print(rv[b][0])
                print(dv[b][0])
            '''
            # if b.sum() > 0:
            #     print(b.sum())
            # print(b.shape)
            # print(b)
            # print(ro[b])
            # b = b[:,:,np.newaxis]
            # print(b.shape)
            # print(ro[b].shape)
            ro[b] = rv[b]
            '''
            if b.sum() > 0:
                print(b.sum())
                c = np.logical_and(b, (np.abs(ro[:,:,0]) > np.abs(ro[:,:,1])))
                if c.sum() > 0:
                    print(pts[c])
                    print(rh[c])
                    print(dh[c])
                    print(rv[c])
                    print(dv[c])
                    print(ro[c])
                    print
            '''
            '''
            c = pts[:,:,0] > 499
            c = np.logical_and(c, pts[:,:,0] < 501)
            c = np.logical_and(c, pts[:,:,1] < 470)
            # c = np.logical_and(c, pts[pts[:,:,0] < 501])
            c = np.logical_and(c, (np.abs(ro[:,:,1] > .5*np.abs(ro[:,:,0]))))
            if c.sum() > 0:
                print(pts[c])
                print(ro[c])
                print(rh[c])
                print(rv[c])
            '''

            return ro

        return vectorInterpolator

    def saveEigens(self, fname):
        if self.lambda_u is None:
            print("saveEigens: eigenvalues not computed yet")
            return
        lu = self.lambda_u
        lv = self.lambda_v
        vu = self.vector_u
        vv = self.vector_v
        grad = self.grad
        # print(lu.shape, lu[np.newaxis,:,:].shape)
        # print(vu.shape)
        st_all = np.concatenate((lu[:,:,np.newaxis], lv[:,:,np.newaxis], vu, vv, grad), axis=2)
        # turn off the default gzip compression
        header = {"encoding": "raw",}
        nrrd.write(str(fname), st_all, header, index_order='C')

    # class function
    def createInterpolator(ar):
        interp = RegularGridInterpolator((np.arange(ar.shape[0]), np.arange(ar.shape[1])), ar, method='linear', bounds_error=False, fill_value=0.)
        return interp

    @property
    def vector_u_interpolator(self):
        if self.vector_u_interpolator_ is None:
            print("create vui")
            self.vector_u_interpolator_ = ST.createVectorInterpolator(self.vector_u)
        return self.vector_u_interpolator_

    @vector_u_interpolator.setter
    def vector_u_interpolator(self, nv):
        # if nv is not None:
        #     raise ValueError("Value must be None")
        print("replace vui")
        self.vector_u_interpolator_ = nv

    @property
    def vector_v_interpolator(self):
        if self.vector_v_interpolator_ is None:
            print("create vvi")
            self.vector_v_interpolator_ = ST.createVectorInterpolator(self.vector_v)
        return self.vector_v_interpolator_

    @vector_v_interpolator.setter
    def vector_v_interpolator(self, nv):
        # if nv is not None:
        #     raise ValueError("Value must be None")
        print("replace vvi")
        self.vector_v_interpolator_ = nv

    @property
    def linearity_interpolator(self):
        if self.linearity_interpolator_ is None:
            self.linearity_interpolator_ = ST.createInterpolator(self.linearity)
        return self.linearity_interpolator_

    def loadOrCreateEigens(self, fname):
        self.lambda_u = None
        print("loading eigens")
        self.loadEigens(fname)
        if self.lambda_u is None:
            print("calculating eigens")
            self.computeEigens()
            print("saving eigens")
            self.saveEigens(fname)

    def computeEigens(self):
        tif = self.image
        sigma0 = 1.  # value used by Hale
        sigma0 = 2.
        ksize = int(math.floor(.5+6*sigma0+1))
        hksize = ksize//2
        kernel = cv2.getGaussianKernel(ksize, sigma0)
        dkernel = kernel.copy()
        for i in range(ksize):
            x = i - hksize
            factor = x/(sigma0*sigma0)
            dkernel[i] *= factor
        # this gives a slightly wrong answer when tif is constant
        # (should give 0, but gives -1.5e-9)
        # gx = cv2.sepFilter2D(tif, -1, dkernel, kernel)
        # this is equivalent, and gives zero when it should
        gx = cv2.sepFilter2D(tif.transpose(), -1, kernel, dkernel).transpose()
        gy = cv2.sepFilter2D(tif, -1, kernel, dkernel)
        grad = np.concatenate((gx, gy)).reshape(2,gx.shape[0],gx.shape[1]).transpose(1,2,0)

        # gaussian blur
        # OpenCV function is several times faster than
        # numpy equivalent
        sigma1 = 8. # value used by Hale
        # sigma1 = 16.
        # gx2 = gaussian_filter(gx*gx, sigma1)
        gx2 = cv2.GaussianBlur(gx*gx, (0, 0), sigma1)
        # gy2 = gaussian_filter(gy*gy, sigma1)
        gy2 = cv2.GaussianBlur(gy*gy, (0, 0), sigma1)
        # gxy = gaussian_filter(gx*gy, sigma1)
        gxy = cv2.GaussianBlur(gx*gy, (0, 0), sigma1)

        gar = np.array(((gx2,gxy),(gxy,gy2)))
        gar = gar.transpose(2,3,0,1)

        # Explicitly calculate eigenvalue of 2x2 matrix instead of
        # using the numpy linalg eigh function; 
        # the explicit method is about 10 times faster.
        # See https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
        # for a derivation
        ad = gx2+gy2
        sq = np.sqrt((gx2-gy2)**2+4*gxy**2)
        lu = .5*(ad+sq)
        lv = .5*(ad-sq)
        # lv should never be < 0, but numerical issues
        # apparently sometimes cause it to happen
        lv[lv<0]=0
        
        # End of explicit calculation of eigenvalues
        
        # both eigenvalues are non-negative, and lu >= lv.
        # lu is zero if gx2, gy2, and gxy are all zero

        # if lu is 0., set lu and lv to 1.; this will give
        # the correct values for isotropy (1.0) and linearity (0.0)
        # for this case

        lu0 = (lu==0)
        # lu0 = (lu<.002)
        # print("lu==0", lu0.sum())
        lu[lu0] = 1.
        lv[lu0] = 1.

        isotropy = lv/lu
        linearity = (lu-lv)/lu
        coherence = ((lu-lv)/(lu+lv))**2

        # explicitly calculate normalized eigenvectors

        # eigenvector u
        # eigenvector u is perpendicular to the layering.
        # The y component of eigenvector u is always >= 0.
        # This can lead to trouble when the layering is
        # nearly parallel to the y axis, because then eigenvector u
        # is nearly parallel to the x axis.  Small changes in the
        # slope can lead to a change in sign of u's x component,
        # in order to keep the y component sign >= 0.  The sign
        # change in x will cause trouble if the u eigenvector is
        # linearly interpolated, because when the x component changes
        # sign, linear interpolation will send the x component to zero.
        # Yikes!
        vu = np.concatenate((gxy, lu-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vu[lu0,:] = 0.
        vulen = np.sqrt((vu*vu).sum(axis=2))
        vulen[vulen==0] = 1
        vu /= vulen[:,:,np.newaxis]
        # print("vu", vu.shape, vu.dtype)
        
        # eigenvector v
        # eigenvector v is parallel to the layering,
        # and perpendicular to eigenvector u
        '''
        # vv = np.concatenate((gxy, lv-gx2)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vv = np.concatenate((lv-gy2, gxy)).reshape(2,gxy.shape[0],gxy.shape[1]).transpose(1,2,0)
        vv[lu0,:] = 0.
        vvlen = np.sqrt((vv*vv).sum(axis=2))
        vvlen[vvlen==0] = 1
        vv /= vvlen[:,:,np.newaxis]
        # At this point, the y component of eigenvector v is always <= 0.
        # But it would be better (more consistent) for the cross
        # product of eigenvectors u and v to always have a consistent
        # sign.  This can be ensured by making sure the x component
        # of eigenvector v is always >= 0.
        vv[vv[:,:,0]<0] *= -1
        '''
        # Compute vv as the perpendicular to vu instead of doing
        # a separate numerical calculation for vv as an eigenvector.
        # This provides
        # more stability in the case of precision problems
        vv = vu[:,:,::-1].copy()
        vv[:,:,0] *= -1

        '''
        vuvv = np.abs((vu*vv).sum(axis=2))
        print("vuvv", vuvv.max())
        bb = vuvv > .1
        # bb = np.logical_and(bb, gxy > .0000001)
        print("bads", bb.sum())
        # print("test", (gxy > .0000001).sum())
        # In theory, vuvv (the dot product of u and v eigenvectors)
        # should always be zero.  Sometimes, because of numerical
        # instability caused by extremely low values of gxy,
        # the dot product can be around 1.  
        print("vuvv gxy", vuvv[np.abs(gxy) > 1.e-18].max())
        if bb.sum() > 0:
            print(gx2[bb][0])
            print(gxy[bb][0])
            print(gy2[bb][0])
            print(lu[bb][0])
            print(lv[bb][0])
        '''

        self.lambda_u = lu
        self.lambda_v = lv
        self.vector_u = vu
        self.vector_v = vv
        self.grad = grad
        self.isotropy = isotropy
        self.linearity = linearity
        self.coherence = coherence

        '''
        print(lu[499:501,469:471])
        print(lv[499:501,469:471])
        print(vu[499:501,469:471])
        print(vv[499:501,469:471])
        print(gxy[499:501,469:471])
        print(gx2[499:501,469:471])
        print(gy2[499:501,469:471])
        '''

        '''
        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        self.vector_u_interpolator = ST.createVectorInterpolator(self.vector_u)
        self.vector_v_interpolator = ST.createVectorInterpolator(self.vector_v)
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)
        '''

    def loadEigens(self, fname):
        try:
            data, data_header = nrrd.read(str(fname), index_order='C')
        except Exception as e:
            print("Error while loading",fname,e)
            return
        print("eigens loaded")

        self.lambda_u = data[:,:,0]
        self.lambda_v = data[:,:,1]
        self.vector_u = data[:,:,2:4]
        self.vector_v = data[:,:,4:6]
        self.grad = data[:,:,6:8]
        lu = self.lambda_u
        lv = self.lambda_v
        self.isotropy = lv/lu
        self.linearity = (lu-lv)/lu
        self.coherence = ((lu-lv)/(lu+lv))**2

        # print("lambda_u", self.lambda_u.shape, self.lambda_u.dtype)
        # print("vector_u", self.vector_u.shape, self.vector_u.dtype)

        '''
        print("creating interpolators")
        self.lambda_u_interpolator = ST.createInterpolator(self.lambda_u)
        self.lambda_v_interpolator = ST.createInterpolator(self.lambda_v)
        # TODO: vector_u can abruptly change sign in areas of
        # near-vertical layers, in whichy case linear interpolation
        # of vector_u is invalid.  To avoid this, vector_u (and vector_v) 
        # should be computed, instead of interpolated, at each interpolation
        # point, from interpolated lambda_u and lambda_v.  Or do
        # the lambdas need to be computed instead of interpolated
        # as well?
        # self.vector_u_interpolator = ST.createVectorInterpolator(self.vector_u)
        # self.vector_v_interpolator = ST.createVectorInterpolator(self.vector_v)
        # self.vector_u_interpolator_ = None
        # self.vector_v_interpolator_ = None
        self.grad_interpolator = ST.createInterpolator(self.grad)
        self.isotropy_interpolator = ST.createInterpolator(self.isotropy)
        self.linearity_interpolator = ST.createInterpolator(self.linearity)
        self.coherence_interpolator = ST.createInterpolator(self.coherence)
        print("interpolators created")
        '''
