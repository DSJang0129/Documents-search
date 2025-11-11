# 🔍 nVent 문서 검색 시스템

AI 기반 하이브리드 문서 검색 시스템 (시맨틱 검색 + 키워드 검색 + 자동 번역)

## 📋 기능

- ✅ **다국어 검색**: 한글로 검색해도 영어 문서 찾기
- ✅ **하이브리드 검색**: 시맨틱 유사도 + 키워드 매칭
- ✅ **자동 번역**: 영어 문서를 한글로 번역
- ✅ **다양한 파일 지원**: PDF, DOCX, TXT, MD
- ✅ **정확한 위치 정보**: 페이지/블록 번호 표시

## 🚀 Streamlit Cloud 배포 방법

### 1단계: GitHub 저장소 생성

1. GitHub (https://github.com) 로그인
2. "New repository" 클릭
3. 저장소 이름: `nvent-document-search`
4. Public 선택
5. "Create repository" 클릭

### 2단계: 파일 업로드

다음 파일들을 저장소에 업로드:

```
nvent-document-search/
├── nvent_semantic_search.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

### 3단계: Streamlit Cloud 배포

1. Streamlit Cloud (https://streamlit.io/cloud) 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소 선택: `본인계정/nvent-document-search`
5. Main file: `nvent_semantic_search.py`
6. "Deploy!" 클릭
7. 5-10분 대기 → 완성!

### 4단계: URL 공유

배포 완료 후 생성된 URL을 공유:
```
https://your-app-name.streamlit.app
```

## 💻 로컬 실행 방법

### 필수 조건
- Python 3.8 이상

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/your-account/nvent-document-search.git
cd nvent-document-search

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. 실행
streamlit run nvent_semantic_search.py
```

브라우저에서 자동으로 열립니다 (기본: http://localhost:8501)

## 📚 사용 방법

### 1. 문서 업로드
- 왼쪽 사이드바 "PDF, DOCX, TXT 파일 업로드"
- 여러 파일 동시 업로드 가능

### 2. 검색
- 검색어 입력창에 키워드 또는 질문 입력
- "검색 실행" 버튼 클릭
- 유사도 임계값 조정 가능 (기본: 0.55)

### 3. 결과 확인
- 문서별로 그룹화된 결과 표시
- 페이지/블록 위치 정보 제공
- 관련 텍스트 하이라이트

### 4. 번역
- "✨ 한국어로 번역" 버튼 클릭
- 영어 원문을 한글로 자동 번역

## ⚙️ 고급 설정

### 검색 파라미터
- **유사도 임계값**: 0.0 ~ 1.0 (기본: 0.55)
  - 높을수록 정확한 결과만 표시
  - 낮을수록 더 많은 결과 표시
- **표시할 문서 그룹 수**: 1 ~ 50 (기본: 10)

### 번역 라이브러리
- deep-translator (추천, 안정적)
- googletrans (대안)

## 🔧 기술 스택

- **Frontend**: Streamlit
- **ML Model**: Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
- **검색 알고리즘**: Cosine Similarity
- **번역**: deep-translator / googletrans
- **문서 처리**: PyPDF2, python-docx

## 📊 성능

- **지원 파일 크기**: 최대 200MB
- **동시 문서 처리**: 제한 없음
- **검색 속도**: 문서 100개 기준 1-2초
- **임베딩 생성**: 문서당 5-10초

## ⚠️ 제한사항

### Streamlit Cloud 무료 플랜
- CPU: 1 코어
- RAM: 1GB
- 앱 개수: 1개
- 공개 앱만 가능

### 권장 사항
- 문서 개수: 100개 이하 권장
- 파일 크기: 개당 10MB 이하 권장
- 동시 사용자: 5명 이하 권장

## 🐛 문제 해결

### 모델 로드 실패
```
에러: sentence-transformers 모델 로드 실패
해결: requirements.txt 확인 및 재설치
```

### 번역 오류
```
에러: 번역 API 서버 차단
해결: 1. 잠시 후 재시도
     2. 다른 번역 라이브러리 선택
```

### 메모리 부족
```
에러: Out of memory
해결: 1. 파일 개수 줄이기
     2. 문서 크기 줄이기
     3. 유료 플랜 고려
```

## 📞 지원

문제 발생 시:
1. GitHub Issues에 문제 등록
2. 에러 메시지 스크린샷 첨부
3. 사용 환경 정보 제공 (OS, 브라우저 등)

## 📝 라이선스

이 프로젝트는 사내 사용을 위한 것입니다.

## 🔄 업데이트

### 최신 버전
- v1.0.0 (2024-01-15)
  - 초기 배포
  - 하이브리드 검색 구현
  - 자동 번역 기능 추가

---

Made with ❤️ for nVent Team