# python 1_vlm_demo.py \            # vlm inference
#     --demo_path ./demo \          # inputted image path
#     --save_part_ply True \        # save the geometry of parts 
#     --remove_bg False \           # Set this to false for RGBA images and true otherwise.
#     --ckpt ./pretrain/vlm        # ckpt path
    
python 2_decoder.py             # decoder inference

# python 3_split.py               # split the mesh

# python 4_simready_gen.py        # convert to URDF & XML
#     --voxel_define 32           # voxel resolution
#     --basepath ./test_demo      # results path
#     --process 0                 # use postprocess
#     --fixed_base 0              # fix the basement of object or not
#     --deformable 0              # introduce deformable parts or not