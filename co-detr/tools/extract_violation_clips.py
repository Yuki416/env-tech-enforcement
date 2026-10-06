import os
import re
import subprocess

markdown_path = '/path/to/repo/co-detr/影片違規時間戳.md'
hdd_dir = '/path/to/repo/hdd2_storage/時段分類彙整_供下載'
output_dir = '/path/to/repo/co-detr/violation_clips'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def find_file(filename):
    # searching globally could be slow, let's use linux find command
    try:
        res = subprocess.check_output(['find', hdd_dir, '-type', 'f', '-name', filename])
        paths = res.decode('utf-8').strip().split('\n')
        if paths and paths[0]:
            return paths[0]
    except Exception as e:
        pass
    return None

with open(markdown_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line.startswith('|') and not '影片名稱' in line and not '---' in line and not ':----' in line:
        parts = line.split('|')
        if len(parts) >= 3:
            raw_filename = parts[1].strip()
            # Clean up escape characters
            filename = raw_filename.replace('\\_', '_')
            if not filename.endswith('.avi') and filename != '':
                filename += '.avi'
                
            raw_times = parts[2].strip()
            # Find all MM:SS formatted times
            times = re.findall(r'\d+:\d+', raw_times)
            
            if filename and times:
                input_file = find_file(filename)
                if not input_file:
                    print(f"Could not find video file: {filename}")
                    continue
                
                for timestamp in times:
                    # Clean filename to output, replace : with _
                    time_clean = timestamp.replace(':', 'm') + 's'
                    filename_stem = filename.rsplit('.', 1)[0]
                    output_file = os.path.join(output_dir, f"{filename_stem}_{time_clean}.avi")
                    
                    if not os.path.exists(output_file):
                        print(f"Extracting {timestamp} from {filename} -> {output_file}")
                        # -ss before -i for fast seeking
                        cmd = ['ffmpeg', '-y', '-ss', timestamp, '-i', input_file, '-t', '10', '-c:v', 'copy', '-c:a', 'copy', output_file]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"Generated: {output_file}")
                    else:
                        print(f"Already exists: {output_file}")

print("Done generating violation clips!")
