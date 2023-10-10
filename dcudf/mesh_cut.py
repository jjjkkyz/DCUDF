import numpy as np
import trimesh
import os
import time
from scipy.spatial import cKDTree

import maxflow
import open3d as o3d
# random.seed(4)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def compute_neighbor(face_neighbor, idx, point_num, rate):
    size = int(point_num/rate)
    mask_wait = np.zeros(point_num, dtype=bool)
    # mask_wait[:] = False
    mask_wait[idx] = True
    mask_wait[face_neighbor[idx]] = True
    wait_list = face_neighbor[idx]
    neighbor_list = wait_list
    flag = True
    count = 0
    while len(neighbor_list) < size:
        if len(wait_list) == 0:
            return None
        k_point = wait_list.pop()
        for k in face_neighbor[k_point]:
            if not mask_wait[k]:
                wait_list.append(k)
                flag=False
                neighbor_list.append(k)

        if flag:
            count += 1
        else:
            count = 0
        if count == 20:
            return None
    # print(len(wait_list))
    # neighbor_list.union(set(wait_list))
    neighbor_list = set(neighbor_list)
    return neighbor_list


def detect_manifold(out_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(out_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(out_mesh.faces)
    non_manifold = np.asarray(mesh.get_non_manifold_vertices())
    return non_manifold


def mesh_cut(mesh,region_rate=20,exp_weight=200):
    # remove small item
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=len(mesh.vertices)//5)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    points = np.sum(np.array(mesh.triangles), axis=1)
    ptree = cKDTree(points)
    face_neighbor = []
    for i in range(len(points)):
        face_neighbor.append([])
    for d in mesh.face_adjacency:
        face_neighbor[d[0]].append(d[1])
        face_neighbor[d[1]].append(d[0])
    try_time = 0
    ite_num = 0
    while True:
        if ite_num > 5:
            if try_time>20:
                print("cut failed")
                break
            ite_num=0
            region_rate *=1.5
            print("decrease region {}".format(region_rate))
            try_time +=1
        # select a seed
        seed = np.random.choice(points.shape[0], 1, replace=False)[0]
        df, near_idx = ptree.query(points[seed], 50)
        near_idx = near_idx-1
        # growing to a region
        seed_neighbor = compute_neighbor(
            face_neighbor, seed, len(points),region_rate)
        if seed_neighbor is None:
            continue
        seed_neighbor.add(seed)
        idx_flag = False
        for idx in near_idx:

            if idx not in seed_neighbor:
                # so this point may be a sink.
                idx_flag=True
                sink_neighbor = compute_neighbor(
                    face_neighbor, idx, len(points),region_rate)
                if sink_neighbor is None:
                    continue
                sink_neighbor.add(idx)
                if len(sink_neighbor.intersection(seed_neighbor)) > 0:
                    break
                # create graph, use face angle as weight
                weight = mesh.face_adjacency_angles[:, np.newaxis].copy()

                weight = weight.max() - weight
                weight = np.exp(exp_weight*weight)

                edges = np.concatenate((mesh.face_adjacency, weight), axis=1)
                new_idx = []
                for i in range(edges.shape[0]):
                    # edge  in seed range
                    if edges[i][0] in seed_neighbor and edges[i][1] in seed_neighbor:
                        continue
                    # edge in sink range
                    elif edges[i][0] in sink_neighbor and edges[i][1] in sink_neighbor:
                        continue
                    # start from seed range
                    elif edges[i][0] in seed_neighbor:
                        edges[i][0] = seed
                    elif edges[i][0] in sink_neighbor:
                        edges[i][0] = idx
                    elif edges[i][1] in seed_neighbor:
                        edges[i][1] = seed
                    elif edges[i][1] in sink_neighbor:
                        edges[i][1] = idx
                    new_idx.append(i)
                edges = edges[new_idx]
                edges[:, 2] = edges[:, 2] + 1

                count = 0
                ori_g = [-1]*len(points)
                g_ori = [-1]*len(points)
                for e in edges:
                    if(e[0] == seed or e[0] == idx):
                        if(ori_g[int(e[1])] == -1):
                            ori_g[int(e[1])] = count
                            g_ori[count] = e[1]
                            count = count+1
                    elif(e[1] == seed or e[1] == idx):
                        if(ori_g[int(e[0])] == -1):
                            ori_g[int(e[0])] = count
                            g_ori[count] = e[0]
                            count = count+1
                    else:
                        if(ori_g[int(e[0])] == -1):
                            ori_g[int(e[0])] = count
                            g_ori[count] = e[0]
                            count = count+1
                        if(ori_g[int(e[1])] == -1):
                            ori_g[int(e[1])] = count
                            g_ori[count] = e[1]
                            count = count+1
                # graph contrain idx -1, just remove it
                g_ori.remove(-1.0)
                g = maxflow.Graph[float]()
                nodes = g.add_nodes(len(g_ori))
                for e in edges:
                    if(e[0] == seed):
                        g.add_tedge(nodes[ori_g[int(e[1])]], e[2], 0)
                    elif(e[0] == idx):
                        g.add_tedge(nodes[ori_g[int(e[1])]], 0, e[2])
                    elif(e[1] == seed):
                        g.add_tedge(nodes[ori_g[int(e[0])]], e[2], 0)
                    elif(e[1] == idx):
                        g.add_tedge(nodes[ori_g[int(e[0])]], 0, e[2])
                    else:
                        g.add_edge(nodes[ori_g[int(e[0])]],
                                   nodes[ori_g[int(e[1])]], e[2], e[2])
                # run mini cut
                flow = g.maxflow()
                # gather two parts
                seg = g.get_grid_segments(nodes)
                partition = (set(), set())
                for id in range(seg.shape[0]):
                    if g_ori[id]<0:
                        continue
                    if(seg[id]):
                        partition[1].add(g_ori[id])
                    else:
                        partition[0].add(g_ori[id])
                rate = abs(len(partition[0])-len(partition[1]))/(len(partition[0])+len(partition[1]))
                if rate > 0.15:
                    ite_num += 1
                    print("not a OK cut, rate is {}".format(rate))
                    break

                seed_list = set(np.array(
                    list(partition[0].union(seed_neighbor)), dtype=int).tolist())
                sink_list = set(np.array(
                    list(partition[1].union(sink_neighbor)), dtype=int).tolist())

                out_mesh_1 = trimesh.base.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces[list(seed_list)], process=False)
                out_mesh_2 = trimesh.base.Trimesh(
                        vertices=mesh.vertices, faces=mesh.faces[list(sink_list)], process=False)
                if len(seed_list) > len(sink_list):
                    out_mesh = out_mesh_1
                else:
                    out_mesh = out_mesh_2
                while True:
                    # cut may lead to non-manifold vertices when model contains too many noises
                    # tightly re-edit cut result will fix it
                    re = detect_manifold(out_mesh)
                    if re.shape[0]>0:
                        for idx in re:
                            add_face = mesh.vertex_faces[idx].tolist()
                            if len(seed_list) > len(sink_list):
                                seed_list = seed_list.union(set(add_face))
                                sink_list = sink_list.difference(set(add_face))
                                if -1 in seed_list:
                                    seed_list.remove(-1)
                            else:
                                sink_list = sink_list.union(set(add_face))
                                seed_list = seed_list.difference(set(add_face))
                                if -1 in sink_list:
                                    sink_list.remove(-1)


                        out_mesh_1 = trimesh.base.Trimesh(
                            vertices=mesh.vertices, faces=mesh.faces[list(seed_list)], process=False)
                        out_mesh_2 = trimesh.base.Trimesh(
                            vertices=mesh.vertices, faces=mesh.faces[list(sink_list)], process=False)

                        if len(seed_list) > len(sink_list):
                            out_mesh = out_mesh_1
                        else:
                            out_mesh = out_mesh_2
                    else:
                        break
                out_mesh_1.remove_unreferenced_vertices()
                out_mesh_2.remove_unreferenced_vertices()
                return (out_mesh_1,out_mesh_2)


        if not idx_flag:
            ite_num += 1
            print("near region may too small")
        # print("no sink not in seed neighbor")
    return


if __name__ == "__main__":
    # root = "postprocess/wait_for_cut"
    # for path in os.listdir(root):
    #     s = time.time()
    #     mesh_name = os.path.join(root, path)
    #     mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
    #     out = cut_mesh_v2(mesh)
    #     out[0].export(
    #         "postprocess/cut_result/{}-0_merge.ply".format(path[:-4]))
    #     out[1].export(
    #         "postprocess/cut_result/{}-1_merge.ply".format(path[:-4]))
    #     e = time.time()
    #     print('total ', e-s)
    root = "experiment/out_cloth_batch"
    with open("cut_time.csv", "w") as f:
        for path in os.listdir(root):
            print(path)
            s = time.time()
            mesh_name = os.path.join(root, path,"mesh","{}_399_Optimize_0.005.ply".format(path))
            mesh = as_mesh(trimesh.load_mesh(mesh_name, process=False))
            out = mesh_cut(mesh)
            if out is not None:
                out[0].export(
                     os.path.join(root, path,"mesh","{}-0_merge.ply".format(path)))
                out[1].export(
                     os.path.join(root, path,"mesh","{}-1_merge.ply".format(path)))
            e = time.time()
            print('total ', e - s)
            f.write(path)
            f.write(",")
            f.write(str(e - s))
            f.write("\n")
            f.flush()
