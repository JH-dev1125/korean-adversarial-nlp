오늘 연구 프로젝트를 처음 실행할 수 있도록 기본 환경을 세팅했어.

VS Code에서 korean-adversarial-nlp 폴더를 열고, 그 안에 .venv라는 파이썬 가상환경을 만들었어.

가상환경은 각자 컴퓨터에서 따로 쓰는 실행환경이라 GitHub에 올리지 않도록 .gitignore 설정도 확인했어.

Git 상태를 확인했고, 원격 저장소와 내 컴퓨터의 프로젝트를 git pull로 최신 상태로 맞췄어.

필요한 파이썬 패키지 중 datasets==2.19.0으로 K-MHaS 데이터셋이 정상적으로 불러와지는 것도 확인했어.

scripts/check_data.py 파일을 만들어 Hugging Face의 jeanlee/kmhas_korean_hate_speech 데이터셋을 불러오도록 했어.

check_data.py의 내용은 아래와 같아.

from pathlib import Path
from datasets import load_dataset


def main():
    raw_dir = Path("data/raw/kmhas")
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("jeanlee/kmhas_korean_hate_speech")
    ds.save_to_disk(str(raw_dir))

    print("데이터셋 구조:")
    print(ds)

    print("\n컬럼:")
    print(ds["train"].column_names)

    print("\n첫 번째 샘플:")
    print(ds["train"][0])

    print("\n데이터 개수:")
    for split in ds.keys():
        print(split, len(ds[split]))

    print(f"\n저장 위치: {raw_dir}")


if __name__ == "__main__":
    main()

이 스크립트는 K-MHaS 데이터를 불러온 뒤 프로젝트 안의 data/raw/kmhas 폴더에 저장하고, 데이터 구조·컬럼·첫 번째 샘플·split별 개수를 출력해.

확인된 K-MHaS 데이터셋은 train 78,977개, validation 8,776개, test 21,939개로 나뉘어 있었어.

각 데이터는 text와 label 컬럼으로 구성되어 있고, label은 하나의 문장에 여러 라벨이 붙을 수 있는 멀티라벨 형식이야.

다음 단계는 이 데이터를 전처리해서 혐오/비혐오 라벨과 욕설 포함 여부 라벨을 만든 뒤, KLUE-BERT 학습 코드와 공격 변형 코드를 구현하는 거야.