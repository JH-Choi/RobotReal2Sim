# Download YCB assets using ManiSkill2
# pip install mani-skill2
# python -m mani_skill2.utils.download_asset ycb
# mv data/mani_skill2_ycb/* /home/choi/.maniskill/data/assets/mani_skill2_ycb/



# cd examples/maniskill
# python gsworld_rand_action_tabletop.py --robot_uids xarm6_uf_gripper \
#     --scene_cfg_name xarm6_align --record_dir ./exp_log/xarm6_align \
#     --ep_len 10 --env_id AlignXArmEnv-v1


cd gsworld/mani_skill/examples/motionplanning
# xarm6
python xarm6/run_with_gs.py -e AlignXArmEnv-v1 -n 1 --vis --scene_cfg_name xarm6_align