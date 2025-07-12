import argparse
import json
import os
import os.path as osp
import subprocess
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--frame_root', default=None, help='If set, convert frames to video. Else, extract clips from videos.')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--video2clip_json', required=True)
    parser.add_argument('--clip_txt_path', required=True)
    parser.add_argument('--clip_folder_map', required=True, help='Mapping file: each line "folder/clipname"')
    parser.add_argument('--low_idx', type=int, default=0)
    parser.add_argument('--high_idx', type=int, default=-1)
    parser.add_argument('--use_nvenc', action='store_true', help='Use NVENC GPU acceleration if FFmpeg supports it.')
    return parser.parse_args()

def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed:", " ".join(cmd))
        print(e.stderr)
        raise

def ffmpeg_extract_clip(input_path, start_s, duration_s, output_path, use_nvenc=False):
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_s:.6f}',
        '-i', input_path,
        '-t', f'{duration_s:.6f}',
        '-c:a', 'copy'
    ]
    if use_nvenc:
        cmd += ['-c:v', 'h264_nvenc']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    cmd.append(output_path)
    run_cmd(cmd)

def ffmpeg_frames_to_video_from_list(frames_dir, fps, output_path, file_list, use_nvenc=False):
    txt_list = osp.join(frames_dir, "frames.txt")
    with open(txt_list, "w") as f:
        for fname in file_list:
            f.write(f"file '{fname}'\n")
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-r', str(fps),
        '-i', txt_list
    ]
    if use_nvenc:
        cmd += ['-c:v', 'h264_nvenc']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    cmd.append(output_path)
    run_cmd(cmd)

def load_clip_folder_map(map_txt_path):
    mapping = {}
    with open(map_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            folder, clip = line.split('/', 1)
            mapping[clip] = line
    return mapping

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)

    with open(args.video2clip_json) as f:
        video2clips_all = json.load(f)

    clip_map = load_clip_folder_map(args.clip_folder_map)

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
            for clip in clips:
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt):
                    continue
                with open(txt) as f:
                    lines = [l.strip() for l in f if l.strip()]
                if len(lines) < 2:
                    continue
                ts = [int(l.split()[0]) for l in lines[1:]]
                if len(ts) < 2 or ts[-1] <= ts[0]:
                    continue

                fps = 1e6 / (ts[1] - ts[0])
                if clip not in clip_map:
                    continue
                frames_dir = osp.join(args.frame_root, clip_map[clip])
                if not osp.exists(frames_dir):
                    continue
                file_list = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))])
                if not file_list:
                    continue

                output_mp4 = osp.join(out_dir, clip + '.mp4')
                if osp.exists(output_mp4):
                    continue

                try:
                    ffmpeg_frames_to_video_from_list(frames_dir, fps, output_mp4, file_list, use_nvenc=args.use_nvenc)
                except Exception:
                    continue
        else:
            vid_path = osp.join(args.video_root, vid + '.mp4')
            if not osp.exists(vid_path):
                continue
            for clip in clips:
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt):
                    continue
                with open(txt) as f:
                    lines = [l.strip() for l in f if l.strip()]
                if len(lines) < 2:
                    continue
                ts = [int(l.split()[0]) for l in lines[1:]]
                if len(ts) < 2 or ts[-1] <= ts[0]:
                    continue
                start_s = ts[0] / 1e6
                duration_s = (ts[-1] - ts[0]) / 1e6

                output_mp4 = osp.join(out_dir, clip + '.mp4')
                if osp.exists(output_mp4):
                    continue

                try:
                    ffmpeg_extract_clip(vid_path, start_s, duration_s, output_mp4, use_nvenc=args.use_nvenc)
                except Exception:
                    continue
