import argparse
import json
import os
import os.path as osp
import imageio
from decord import VideoReader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frame_root',       required=True)
    p.add_argument('--save_path',        required=True)
    p.add_argument('--video2clip_json',  required=True)
    p.add_argument('--clip_txt_path',    required=True)
    p.add_argument('--clip_folder_map',  required=True)
    p.add_argument('--low_idx',          type=int, default=0)
    p.add_argument('--high_idx',         type=int, default=-1)
    p.add_argument('--gpus',             type=int, default=8)
    p.add_argument('--num_machines',     type=int, default=1,
                   help='Total parallel machines/instances')
    p.add_argument('--machine_rank',     type=int, default=0,
                   help='This machine’s rank (0 to num_machines-1)')
    return p.parse_args()

def load_map(path):
    m = {}
    with open(path) as f:
        for L in f:
            line = L.strip()
            if not line: continue
            folder, clip = line.split('/', 1)
            m[clip] = line
    return m

def process_video(vid, clip_list, clip_map, args, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    out_dir = osp.join(args.save_path, vid)
    done_marker = osp.join(out_dir, '.done')
    if osp.exists(done_marker):
        return  # already done

    os.makedirs(out_dir, exist_ok=True)
    for clip in clip_list:
        txt = osp.join(args.clip_txt_path, clip + '.txt')
        if not osp.exists(txt):
            continue
        lines = [l.strip() for l in open(txt).read().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        ts = [int(l.split()[0]) for l in lines[1:]]
        if ts[-1] <= ts[0]:
            continue

        fps = 1e6 / (ts[1] - ts[0])
        if clip not in clip_map:
            continue
        frames_dir = osp.join(args.frame_root, clip_map[clip])
        if not osp.isdir(frames_dir):
            continue

        frame_files = sorted(
            osp.join(frames_dir, x)
            for x in os.listdir(frames_dir)
            if x.lower().endswith(('.png', '.jpg'))
        )
        if not frame_files:
            continue

        out_mp4 = osp.join(out_dir, clip + '.mp4')
        if osp.exists(out_mp4):
            continue

        frames = [imageio.imread(fp) for fp in frame_files]
        imageio.mimsave(out_mp4, frames, fps=fps)

        vr = VideoReader(out_mp4)
        assert len(vr) == len(frame_files)

    # mark done
    open(done_marker, 'w').close()

def main():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    video2clips = json.load(open(args.video2clip_json))
    clip_map    = load_map(args.clip_folder_map)

    vids = list(video2clips.keys())
    # apply index slicing
    if args.high_idx != -1:
        vids = vids[args.low_idx:args.high_idx]
    else:
        vids = vids[args.low_idx:]

    # further shard across machines
    vids = [v for idx, v in enumerate(vids)
            if idx % args.num_machines == args.machine_rank]

    # assign each video a GPU in round-robin
    assignments = [
        (vid, video2clips[vid], clip_map, i % args.gpus)
        for i, vid in enumerate(vids)
    ]

    with ProcessPoolExecutor(max_workers=min(len(assignments), args.gpus)) as exe:
        futures = {
            exe.submit(process_video, vid, clips, clip_map, args, gpu): vid
            for vid, clips, clip_map, gpu in assignments
        }
        for _ in tqdm(as_completed(futures),
                      total=len(futures), desc='videos'):
            pass

if __name__ == '__main__':
    main()
