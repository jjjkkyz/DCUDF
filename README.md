# DCUDF
Implementation of "Robust Zero Level-Set Extraction from Unsigned Distance Fields Based on Double Covering"

We now release main code of our algorithm. 
You can use our code in dcudf folder to extract mesh from unsigned distance fields.


# Install
    # we use torch to calculate gridient
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

    # some referenced package
    pip install open3d trimesh matplotlib scipy scikit-image

    # use mini-cut 
    pip install PyMaxflow

    # if use 3D IoU to seperate watertight double cover mesh
    pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

    # to use our train and test code, we need pyhocon to init config
    pip install pyhocon==0.3.59

# Usage
    from dcudf.mesh_extraction import dcudf

    query_fun = lambda pts: udf_network.udf(pts)
    resolution = 256
    threshold = 0.005

    # we have a lot default parameters, see source code for details.
    extractor = dcudf(query_fun, resolution, threshold)
    mesh = extractor.optimize()

# Demo
    # to run our demo
    python evaluate.py --conf confs/cloth.conf --gpu 0 --dataname 564

# Acknowledgement
This code base is built upon [CAPUDF](https://github.com/junshengzhou/CAP-UDF). 
Thanks for their remarkable job !