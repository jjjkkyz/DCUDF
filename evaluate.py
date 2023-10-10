import time

import torch
import os
import trimesh
import numpy as np
import argparse
from pyhocon import ConfigFactory

from models.fields import UDFNetwork
from dcudf.mesh_extraction import dcudf

def test(args):
    s = time.time()
    args.dir_name = args.dataname
    torch.cuda.set_device(args.gpu)
    conf_path = args.conf
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    device = torch.device('cuda')
    conf = ConfigFactory.parse_string(conf_text)
    udf_network = UDFNetwork(**conf['model.udf_network']).to(device)
    checkpoint_name = conf.get_string('evaluate.load_ckpt')
    base_exp_dir = conf['general.base_exp_dir'] + args.dataname
    checkpoint = torch.load(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name),
                            map_location=device)
    print(os.path.join(base_exp_dir, 'checkpoints', checkpoint_name))
    udf_network.load_state_dict(checkpoint['udf_network_fine'])

    mesh_path = os.path.join(conf['dataset'].data_dir, "input", "{}.ply".format(args.dataname))
    mesh = trimesh.load_mesh(mesh_path)
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)
    object_bbox_min = np.array(mesh.bounds[0])-0.05
    object_bbox_max = np.array(mesh.bounds[1])+0.05
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    conf = ConfigFactory.parse_string(conf_text)

    evaluator = dcudf(lambda pts: udf_network.udf(pts), conf.get_int('evaluate.resolution'), conf.get_float('evaluate.threshold'),
                      max_iter=conf.get_int("evaluate.max_iter"), normal_step=conf.get_int("evaluate.normal_step"),
                      laplacian_weight=conf.get_int("evaluate.laplacian_weight"),
                      bound_min=object_bbox_min, bound_max=object_bbox_max,
                      is_cut=conf.get_int("evaluate.is_cut"), region_rate=conf.get_int("evaluate.region_rate"),
                      max_batch=conf.get_int("evaluate.max_batch"), learning_rate=conf.get_float("evaluate.learning_rate"),
                      warm_up_end=conf.get_int("evaluate.warm_up_end"), report_freq=conf.get_int("evaluate.report_freq"),
                      watertight_separate=conf.get_int("evaluate.watertight_separate"))

    mesh = evaluator.optimize()
    base_exp_dir = os.path.join(conf['general.base_exp_dir'], args.dataname)
    mesh_out_dir = os.path.join(base_exp_dir, "mesh")
    os.makedirs(mesh_out_dir,exist_ok=True)
    mesh.export(mesh_out_dir + '/' + '{}_{}.ply'.format(args.dataname,str(conf.get_float('evaluate.threshold'))))
    t = time.time()
    return t-s


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str)
    args = parser.parse_args()
    print(test(args))