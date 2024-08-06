# DCUDF: Robust Zero Level-Set Extraction from Unsigned Distance Fields Based on Double Covering (SIGGRAPH ASIA 2023)
## [<a href="https://lcs.ios.ac.cn/~houf/pages/dcudf/index.html" target="_blank">Project Page</a>]  [<a href="https://arxiv.org/abs/2310.03431" target="_blank">Arxiv</a>]

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
    
    # for complex models or nnon-manifold models such as car, sences, etc. Please disable cut postprocess.
    extractor = dcudf(query_fun, resolution, threshold, is_cut=False)
    
    # for low resolution, please decrease laplacian weight.
    extractor = dcudf(query_fun, 64, threshold, laplacian_weight=500)
    
    #Details in shown in code, please read it.
    
    # for high quality distance field, you can also decrease laplacian weights.
    # for high resolution, you can use as low r as possible to extract mesh.
    # Email us if you have any problem about our hyper-parameters.
   

    mesh = extractor.optimize()

# Demo
    # to run our demo
    python evaluate.py --conf confs/cloth.conf --gpu 0 --dataname 564

# Acknowledgement
This code base is built upon [CAPUDF](https://github.com/junshengzhou/CAP-UDF). 
Thanks for their remarkable job !

## Citation
```
@article{Hou2023DCUDF,
	author = {Hou, Fei and Chen, Xuhui and Wang, Wencheng and Qin, Hong and He, Ying},
	title = {Robust Zero Level-Set Extraction from Unsigned Distance Fields Based on Double Covering},
	year = {2023},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {42},
	number = {6},
	issn = {0730-0301},
	doi = {10.1145/3618314},
	journal = {ACM Trans. Graph.},
	month = {dec},
	articleno = {245},
	numpages = {15},
}
```
