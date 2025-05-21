import os
import re
import sys

def check_solver_functions_against_images(output_images_dir, tasksolver_path):
    """
    tasksolver.py 파일에서 모든 solve_XXXXXXXX 함수를 추출하고,
    XXXXXXXX 부분이 output_images 디렉토리에 파일로 존재하는지 확인합니다.
    
    Args:
        output_images_dir: 이미지 파일들이 있는 디렉토리 경로
        tasksolver_path: tasksolver.py 파일의 경로
    """
    # tasksolver.py 파일 내용을 읽어옵니다
    try:
        with open(tasksolver_path, 'r', encoding='utf-8') as f:
            tasksolver_content = f.read()
    except Exception as e:
        print(f"tasksolver.py 파일을 읽는 중 오류가 발생했습니다: {e}")
        return
    
    # output_images 디렉토리의 파일 목록을 가져옵니다
    try:
        image_files = os.listdir(output_images_dir)
    except Exception as e:
        print(f"이미지 디렉토리를 읽는 중 오류가 발생했습니다: {e}")
        return
    
    # solve_XXXXXXXX 패턴을 찾기 위한 정규 표현식
    pattern = r'def\s+solve_([0-9a-f]{8})'
    solver_functions = re.findall(pattern, tasksolver_content)
    
    if not solver_functions:
        print("tasksolver.py 파일에서 solve_XXXXXXXX 형태의 함수를 찾을 수 없습니다.")
        return
    
    print(f"\ntasksolver.py의 함수들이 {output_images_dir} 디렉토리에 파일로 존재하는지 확인합니다:\n")
    print(f"{'함수 (solve_XXXXXXXX)':<30} {'파일 존재 여부':<15}")
    print("-" * 45)
    
    # 각 함수에 대해 해당 16진수 이름을 가진 파일이 있는지 확인합니다
    for hex_id in sorted(solver_functions):
        # 파일 이름에 해당 16진수가 포함되어 있는지 확인
        file_exists = any(hex_id in file_name for file_name in image_files)
        result = "✓" if file_exists else "X"
        print(f"solve_{hex_id:<22} {result:<15}")
    
    # 통계 출력
    found_count = sum(1 for hex_id in solver_functions if any(hex_id in file_name for file_name in image_files))
    print(f"\n총 {len(solver_functions)}개의 함수 중 {found_count}개가 이미지 파일과 일치합니다.")
    print(f"일치율: {found_count/len(solver_functions)*100:.2f}%")

if __name__ == "__main__":
    # 명령줄 인수를 확인합니다
    if len(sys.argv) != 3:
        print("사용법: python script.py <output_images_디렉토리> <tasksolver.py_경로>")
        print("예시: python script.py ./dataset/output_images ./skeleton/tasksolver/tasksolver.py")
        sys.exit(1)
    
    output_images_dir = sys.argv[1]
    tasksolver_path = sys.argv[2]
    
    # 경로가 존재하는지 확인합니다
    if not os.path.isdir(output_images_dir):
        print(f"오류: '{output_images_dir}'는 유효한 디렉토리가 아닙니다.")
        sys.exit(1)
    
    if not os.path.isfile(tasksolver_path):
        print(f"오류: '{tasksolver_path}'는 유효한 파일이 아닙니다.")
        sys.exit(1)
    
    check_solver_functions_against_images(output_images_dir, tasksolver_path)