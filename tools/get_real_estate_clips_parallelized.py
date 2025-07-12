import argparse
import json
import os
import os.path as osp
import subprocess
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--frame_root', default=None,
                        help='If set, frames→video instead of raw→clips')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--video2clip_json', required=True)
    parser.add_argument('--clip_txt_path', required=True)
    parser.add_argument('--low_idx', type=int, default=0)
    parser.add_argument('--high_idx', type=int, default=-1)
    return parser.parse_args()

def ffmpeg_extract_clip(input_path, start_s, duration_s, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-ss', f'{start_s:.6f}',
        '-i', input_path,
        '-t', f'{duration_s:.6f}',
        '-c:v', 'h264_nvenc',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ffmpeg_frames_to_video(frames_dir, fps, output_path):
    # Assumes frames named 000001.png, 000002.png, … zero-padded equally
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-framerate', str(fps),
        '-i', osp.join(frames_dir, '%*.png'),  # adjust padding to your filenames
        '-c:v', 'h264_nvenc',
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    with open(args.video2clip_json) as f:
        video2clips_all = json.load(f)

    keys = list(video2clips_all.keys())
    if args.high_idx != -1:
        keys = keys[args.low_idx:args.high_idx]
    else:
        keys = keys[args.low_idx:]

    for vid in tqdm(keys, desc='videos'):
        out_dir = osp.join(args.save_path, vid)
        os.makedirs(out_dir, exist_ok=True)

        clips = video2clips_all[vid]
        if args.frame_root:
            for clip in tqdm(clips, desc=f'  frames→{vid}'):
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt): continue
                lines = open(txt).read().splitlines()[1:]
                ts = [int(l.split()[0]) for l in lines]
                if len(ts) < 2 or ts[-1] <= ts[0]: continue

                fps = 1e6 / (ts[1] - ts[0])
                ffmpeg_frames_to_video(
                    osp.join(args.frame_root, clip),
                    fps,
                    osp.join(out_dir, clip + '.mp4')
                )

        else:
            vid_path = osp.join(args.video_root, vid + '.mp4')
            if not osp.exists(vid_path): continue

            for clip in tqdm(clips, desc=f'  raw→{vid}'):
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt): continue
                lines = open(txt).read().splitlines()[1:]
                ts = [int(l.split()[0]) for l in lines]
                if ts[-1] <= ts[0]: continue

                start_s = ts[0] / 1e6
                duration_s = (ts[-1] - ts[0]) / 1e6
                ffmpeg_extract_clip(
                    vid_path, start_s, duration_s,
                    osp.join(out_dir, clip + '.mp4')
                )
