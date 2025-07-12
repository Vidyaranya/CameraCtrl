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
    parser.add_argument('--low_idx', type=int, default=0)
    parser.add_argument('--high_idx', type=int, default=-1)
    parser.add_argument('--use_nvenc', action='store_true', help='Use NVENC GPU acceleration if FFmpeg supports it.')
    return parser.parse_args()

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("FFmpeg command failed!")
        print("Command:", " ".join(cmd))
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise

def ffmpeg_extract_clip(input_path, start_s, duration_s, output_path, use_nvenc=False):
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_s:.6f}',
        '-i', input_path,
        '-t', f'{duration_s:.6f}',
        '-c:a', 'copy'
    ]
    # Video encoding: GPU or CPU
    if use_nvenc:
        cmd += ['-c:v', 'h264_nvenc']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    cmd.append(output_path)
    run_cmd(cmd)

def ffmpeg_frames_to_video(frames_dir, fps, output_path, pattern='%06d.png', use_nvenc=False):
    # Detect pattern automatically if possible (default is 6 digits, adjust if needed)
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', osp.join(frames_dir, pattern)
    ]
    if use_nvenc:
        cmd += ['-c:v', 'h264_nvenc']
    else:
        cmd += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
    cmd.append(output_path)
    run_cmd(cmd)

def detect_pattern(frames_dir):
    """Auto-detect pattern like %06d.png or %04d.png based on frame filenames."""
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        return None
    files.sort()
    sample = files[0]
    import re
    m = re.match(r'(\D*)(\d+)(\.\w+)', sample)
    if not m:
        # fallback
        return files[0]
    prefix, digits, suffix = m.groups()
    pattern = f"{prefix}%0{len(digits)}d{suffix}"
    return pattern

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
            # Frame directories to videos
            for clip in tqdm(clips, desc=f'  frames→{vid}'):
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt): continue
                lines = open(txt).read().splitlines()[1:]
                ts = [int(l.split()[0]) for l in lines if l.strip()]
                if len(ts) < 2 or ts[-1] <= ts[0]: continue

                fps = 1e6 / (ts[1] - ts[0])
                frames_dir = osp.join(args.frame_root, clip)
                if not osp.exists(frames_dir): continue

                # Auto-detect frame filename pattern
                pattern = detect_pattern(frames_dir)
                if not pattern:
                    print(f"  No frames found in {frames_dir}")
                    continue

                output_mp4 = osp.join(out_dir, clip + '.mp4')
                if osp.exists(output_mp4): continue

                try:
                    ffmpeg_frames_to_video(frames_dir, fps, output_mp4, pattern=pattern, use_nvenc=args.use_nvenc)
                except Exception as e:
                    print(f"Failed to make video for {clip}: {e}")

        else:
            # Video clips extraction
            vid_path = osp.join(args.video_root, vid + '.mp4')
            if not osp.exists(vid_path): continue

            for clip in tqdm(clips, desc=f'  raw→{vid}'):
                txt = osp.join(args.clip_txt_path, clip + '.txt')
                if not osp.exists(txt): continue
                lines = open(txt).read().splitlines()[1:]
                ts = [int(l.split()[0]) for l in lines if l.strip()]
                if len(ts) < 2 or ts[-1] <= ts[0]: continue

                start_s = ts[0] / 1e6
                duration_s = (ts[-1] - ts[0]) / 1e6

                output_mp4 = osp.join(out_dir, clip + '.mp4')
                if osp.exists(output_mp4): continue

                try:
                    ffmpeg_extract_clip(vid_path, start_s, duration_s, output_mp4, use_nvenc=args.use_nvenc)
                except Exception as e:
                    print(f"Failed to extract clip {clip} from {vid}: {e}")
