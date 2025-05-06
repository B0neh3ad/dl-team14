# How to set

로컬 환경에서 작업하고 싶다면 practice server와 python 환경을 맞춰주어야 합니다. 이 절에서는 venv를 이용해 환경을 동기화하는 방법을 설명합니다.
> 아래의 shell 명령어는 모두 프로젝트의 root directory에 위치해있다는 가정 하에 작성되었습니다.

## 1. Create virtual env
기본적으로, 아래의 명령어로 python 가상환경을 생성 & 활성화합니다.
```shell
# create venv
python -m venv venv

# activate venv
source venv/bin/activate
```

이때 python, pip, conda의 버전은 아래에 명시된 practice server에서의 버전과 같아야 합니다.
|항목|버전|
|--|--|
|`python`|3.11.1|
|`conda`|25.1.0|
|`pip`|24.3.1|

즉, terminal에서 버전 체크를 하면 아래와 같이 나와야 합니다.
```shell
$ python --version
Python 3.11.11
$ conda --version
conda 25.1.0
$ pip --version
pip 24.3.1 from /home/student/workspace/venv/lib/python3.11/site-packages/pip (python 3.11)
```

만약 버전이 안 맞는다면 아래의 명령어들을 활용하여 꼭 맞추고 가도록 합시다.
```shell
# create venv using python 3.11.11
conda create -n venv python=3.11.11
# or 
{python 3.11.11의 path} -m venv myenv

# 아래의 두 명령어를 실행하기 전 venv 활성화하기
source venv/bin/activate

# install conda 25.1.0
conda install conda=25.1.0

# install pip 24.3.1
pip install --upgrade pip==24.3.1
```

## 2. Install required packages

### 1. Install torch manually
`requirements.txt`에서 pytorh 관련 패키지들은 주석 처리되어 있습니다. 이들은 `pip install -r requirements.txt`로는 명시된 버전을 자동 설치할 수 없습니다. Cuda 버전이 특정되어 있기 때문입니다.

따라서 가상환경 하에서 아래의 명령어를 입력하여 설치합니다.

```shell
# torch, torchvision, and torchaudio
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install other packages

이제 아래의 명령어를 실행하여 나머지 패키지들을 한 번에 설치합니다.
```shell
cd skeleton/artifacts
pip install -r requirements.txt
```

## 3. If you updated dependencies...
새로운 패키지를 설치했다면 꼭 `skeleton/artifacts/requirements.txt` 파일에 이를 반영하도록 합시다.

```shell
cd skeleton/artifacts
pip list --format=freeze > requirements.txt
```

이때, 2-1에서 수동설치했던 패키지들(`libmambapy`, `menuinst`, `torch`, `torchvision`, `torchaudio`)에 대한 주석이 해제되었다면 꼭 다시 해줘야 합니다.