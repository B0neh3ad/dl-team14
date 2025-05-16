import os
import json
from PIL import Image, ImageDraw

# 설정
input_dir = '../dataset/'  
output_base_dir = '../dataset/output_images'  
cell_size = 20  
color_map = {
    0: "#000000", 1: "#ff0000", 2: "#00ff00", 3: "#ffff00",
    4: "#0000ff", 5: "#ff00ff", 6: "#00ffff", 7: "#ffffff",
    8: "#ff5555", 9: "#55ff55",
}

os.makedirs(output_base_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith('.json'):
        continue

    # JSON 파일명(확장자 제거)을 폴더명으로 사용
    rule_name = os.path.splitext(fname)[0]
    rule_dir = os.path.join(output_base_dir, rule_name)
    os.makedirs(rule_dir, exist_ok=True)

    with open(os.path.join(input_dir, fname), 'r') as f:
        rules = json.load(f)

    for idx, rule in enumerate(rules):
        in_mat  = rule['input']
        out_mat = rule['output']
        h = max(len(in_mat), len(out_mat))
        w_in  = len(in_mat[0])  if in_mat  else 0
        w_out = len(out_mat[0]) if out_mat else 0
        gap = 2  # input/output 사이 띄울 칸 수

        # 캔버스 생성
        img = Image.new('RGB', ((w_in + gap + w_out)*cell_size, h*cell_size))
        draw = ImageDraw.Draw(img)

        # 매트릭스 그리기 함수
        def draw_mat(mat, x_off):
            for i, row in enumerate(mat):
                for j, v in enumerate(row):
                    x0 = (j + x_off)*cell_size
                    y0 = i*cell_size
                    draw.rectangle([x0, y0, x0+cell_size, y0+cell_size],
                                   fill=color_map.get(v, "#000000"))

        draw_mat(in_mat, 0)
        draw_mat(out_mat, w_in + gap)

        # 여러 이미지가 한 폴더에 들어가도록 파일명에 인덱스 붙이기
        img.save(os.path.join(rule_dir, f"combined_{idx}.png"))
        print(f"Rule {rule_name} - Image {idx} saved.")

print("모든 rule 폴더 안에 다수의 combined_*.png 파일이 생성되었어.")
