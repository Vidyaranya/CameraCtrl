import argparse
import json
import os
import os.path as osp
import subprocess
import tempfile
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frame_root', required=True,
                   help='Root under which clip folders live (via clip_folder_map)')
    p.add_argument('--save_path', required=True)
    p.add_argument('--video2clip_json', required=True)
    p.add_argument('--clip_txt_path', required=True)
    p.add_argument('--clip_folder_map', required=True,
                   help='Each line "folder/clipname" mapping to frame_root/folder/clipname')
    p.add_argument('--low_idx', type=int, default=0)
    p.add_argument('--high_idx', type=int, default=-1)
    return p.parse_args()

def run_cmd(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def ffmpeg_frames_to_video(frames_dir, fps, output_path, file_list):
    # Write a temp concat list (absolute paths) so ffmpeg can read arbitrary names
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        list_path = f.name
        for fname in file_list:
            f.write(f"file '{osp.join(frames_dir, fname)}'\n")
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-r', str(fps),
        '-i', list_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '30',
        output_path
    ]
    run_cmd(cmd)
    os.remove(list_path)

def load_map(path):
    m = {}
    with open(path) as f:
        for L in f:
            line = L.strip()
            if not line: 
                continue
            folder, clip = line.split('/', 1)
            m[clip] = line
    return m

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    video2clips = json.load(open(args.video2clip_json))
    clip_map = load_map(args.clip_folder_map)

    keys = list(video2clips.keys())
    if args.high_idx != -1:
        keys = keys[args.low_idx:args.high_idx]
    else:
        keys = keys[args.low_idx:]

    for vid in tqdm(keys, desc='videos'):
        out_dir = osp.join(args.save_path, vid)
        os.makedirs(out_dir, exist_ok=True)

        for clip in video2clips[vid]:
            # load timestamps
            txt = osp.join(args.clip_txt_path, clip + '.txt')
            if not osp.exists(txt):
                continue
            lines = [l for l in open(txt).read().splitlines() if l.strip()]
            if len(lines) < 2:
                continue
            ts = [int(l.split()[0]) for l in lines[1:]]
            if ts[-1] <= ts[0]:
                continue

            fps = 1e6 / (ts[1] - ts[0])

            # locate frames dir via map
            if clip not in clip_map:
                continue
            frames_dir = osp.join(args.frame_root, clip_map[clip])
            if not osp.isdir(frames_dir):
                continue

            # gather all image files
            files = sorted(f for f in os.listdir(frames_dir)
                           if f.lower().endswith(('.png','.jpg')))
            if not files:
                continue

            out_mp4 = osp.join(out_dir, clip + '.mp4')
            if osp.exists(out_mp4):
                continue

            try:
                ffmpeg_frames_to_video(frames_dir, fps, out_mp4, files)
            except subprocess.CalledProcessError:
                pass
