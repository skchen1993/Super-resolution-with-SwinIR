Put high resolution trimed training image here
We have to make sure that each dimension of HR image is bigger than 144,
because SwinIR cut out 48x48 image patch form image for training.
Besides, in order to do the x3 downscale for LR image, we have to trim the
HR training image size to mutiple of 3.

