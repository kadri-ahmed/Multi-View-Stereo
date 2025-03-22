import time
import pymeshlab


def save_poisson_mesh(mesh_file_path, depth=13, max_faces=10_000_000):

    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file_path)
    print("loaded", mesh_file_path)

    # compute normals
    start = time.time()
    ms.compute_normal_for_point_clouds()
    print("computed normals")

    # run poisson
    ms.generate_surface_reconstruction_screened_poisson(depth=depth)
    end_poisson = time.time()
    print(f"finish poisson in {end_poisson - start} seconds")

    # save output
    parts = mesh_file_path.split(".")
    out_file_path = ".".join(parts[:-1])
    suffix = parts[-1]
    out_file_path_poisson = f"{out_file_path}_poisson_meshlab_depth_{depth}.{suffix}"
    ms.save_current_mesh(out_file_path_poisson)
    print("saved poisson mesh", out_file_path_poisson)

    # quadric edge collapse to max faces
    start_quadric = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
    end_quadric = time.time()
    print(f"finish quadric decimation in {end_quadric - start_quadric} seconds")

    # save output
    out_file_path_quadric = f"{out_file_path}_poisson_meshlab_depth_{depth}_quadric_{max_faces}.{suffix}"
    ms.save_current_mesh(out_file_path_quadric)
    print("saved quadric decimated mesh", out_file_path_quadric)

    return out_file_path_poisson


save_poisson_mesh('/home/aysu/Desktop/poisson_reconstructions/Multiview_ICP_ptCloud.ply')
