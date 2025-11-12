# nvent_semantic_search.py (번역 에러 수정 버전)
import streamlit as st
from io import BytesIO
import os
import time
import numpy as np
from collections import defaultdict
import re
from typing import Dict, Any, List, Set, Union

# --- 1. 환경 설정 및 라이브러리 로드 ---

# 텍스트 추출 라이브러리
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
    st.warning("⚠️ PyPDF2 라이브러리를 설치해야 PDF 파일을 처리할 수 있습니다.\n```pip install pypdf2```")

try:
    import docx
except ImportError:
    docx = None
    st.warning("⚠️ python-docx 라이브러리를 설치해야 DOCX 파일을 처리할 수 있습니다.\n```pip install python-docx```")

# 시맨틱 검색 라이브러리
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 모델 로드 (캐싱을 통해 재실행 시 로딩 시간 단축)
    @st.cache_resource
    def load_model():
        # 다국어 지원 및 좋은 성능의 모델 선택
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    MODEL = load_model()
    SEMANTIC_LIBRARIES_LOADED = True
except ImportError:
    SEMANTIC_LIBRARIES_LOADED = False
    MODEL = None
    st.error("🚨 시맨틱 검색 필수 라이브러리 설치 필요:\n```pip install sentence-transformers scikit-learn```")

# --- ⭐ 수정: 무료 번역 라이브러리 로드 및 상태 저장 ---
TRANSLATOR_OBJECTS = {}
AVAILABLE_TRANSLATORS = {}

# 1차 시도: deep-translator (가장 안정적)
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    
    def get_deep_translator_object():
        return DeepGoogleTranslator(source='en', target='ko')
        
    TRANSLATOR_OBJECTS['deep_translator'] = get_deep_translator_object
    AVAILABLE_TRANSLATORS['Google Translate (deep-translator)'] = 'deep_translator'
except ImportError:
    pass

# 2차 시도: googletrans
if not AVAILABLE_TRANSLATORS:
    try:
        from googletrans import Translator
        
        def get_googletrans_object():
            return Translator()
            
        TRANSLATOR_OBJECTS['googletrans'] = get_googletrans_object
        AVAILABLE_TRANSLATORS['Google Translate (googletrans)'] = 'googletrans'
    except ImportError:
        pass

if not AVAILABLE_TRANSLATORS:
    st.warning("⚠️ 무료 번역 라이브러리 설치 필요:\n```pip install deep-translator``` 또는 ```pip install googletrans==4.0.0-rc1```")
    FREE_TRANSLATOR_LOADED = False
else:
    FREE_TRANSLATOR_LOADED = True
# --- ⭐ 수정 끝 ---


# --- 2. 설정 ---
st.set_page_config("문서 검색기", layout="wide")
st.title("🔎 문서 검색")

# session state 초기화
if "docs" not in st.session_state:
    st.session_state["docs"] = []
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
# ⭐ 추가: 현재 선택된 번역 라이브러리 상태 저장
if "selected_translator_key" not in st.session_state:
    # 기본값 설정
    st.session_state["selected_translator_key"] = list(AVAILABLE_TRANSLATORS.values())[0] if AVAILABLE_TRANSLATORS else "없음"
    
CHUNK_LENGTH = 300 # 청크 길이 (토큰 대신 글자 수)
OVERLAP_LENGTH = 50 # 청크 오버랩
SIMILARITY_THRESHOLD = 0.55 # 임계값 설정
# 정확히 일치하는 경우, 시맨틱 유사도(최대 1.0)를 초과하는 점수를 부여하여 항상 상위 정렬되도록 함
LEXICAL_OVERRIDE_SCORE = 1.01 

# --- 3. 유틸리티 함수 ---

def clean_text(text: str) -> str:
    """텍스트 정리: 공백, 특수문자 등 정리"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_pdf_pages_data(uploaded_file: BytesIO) -> List[Dict[str, Any]]:
    """PDF 파일에서 텍스트 추출 및 페이지별 데이터 반환"""
    if not PdfReader:
        return []
    try:
        reader = PdfReader(uploaded_file)
        pages_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text) > 20: 
                pages_data.append({
                    "text": cleaned_text,
                    "page_num": i + 1
                })
        return pages_data
    except Exception as e:
        st.error(f"PDF 처리 오류: {e}")
        return []

def get_text_from_docx(uploaded_file: BytesIO) -> str:
    """DOCX 파일에서 텍스트 추출"""
    if not docx:
        return ""
    try:
        document = docx.Document(uploaded_file)
        text = "\n".join([p.text for p in document.paragraphs])
        return clean_text(text)
    except Exception as e:
        st.error(f"DOCX 처리 오류: {e}")
        return ""

def get_document_chunks(text: str, chunk_length: int=CHUNK_LENGTH, overlap: int=OVERLAP_LENGTH) -> List[str]:
    """텍스트를 겹치는 부분과 함께 청크로 분할"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_length
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_length - overlap)
        if start < 0: 
            start = 0 
    return chunks

def process_file(uploaded_file: BytesIO):
    """업로드된 파일 처리 및 청크 생성 (위치 정보 포함)"""
    if uploaded_file.name in [doc['name'] for doc in st.session_state["docs"]]:
        st.warning(f"'{uploaded_file.name}' 파일은 이미 로드되었습니다.")
        return

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    chunks_with_location = []
    raw_text_size = 0

    if file_extension == '.pdf':
        pages_data = get_pdf_pages_data(uploaded_file) 
        if not pages_data:
            st.error("PDF에서 텍스트를 추출하지 못했거나 내용이 너무 짧습니다.")
            return

        for page_data in pages_data:
            page_text = page_data['text']
            page_chunks = get_document_chunks(page_text)
            raw_text_size += len(page_text)

            for chunk_text in page_chunks:
                chunks_with_location.append({
                    "text": chunk_text,
                    "location": f"페이지 {page_data['page_num']}" 
                })

    elif file_extension in ['.docx']:
        raw_text = get_text_from_docx(uploaded_file)
        if not raw_text: return
        
        raw_text_size = len(raw_text)
        base_chunks = get_document_chunks(raw_text)
        for i, chunk_text in enumerate(base_chunks):
            chunks_with_location.append({
                "text": chunk_text,
                "location": f"블록 {i + 1}"
            })
            
    elif file_extension in ['.txt', '.md']:
        try:
            # ⭐ 인코딩 오류가 발생할 수 있으므로, 'errors='ignore'' 추가 (텍스트 파일 로딩 안전성 확보)
            raw_text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        except Exception as e:
            st.error(f"텍스트 파일 디코딩 오류: {e}. 파일 인코딩을 확인해주세요.")
            return
            
        if not raw_text:
            st.error("텍스트 파일 내용이 비어 있습니다.")
            return
            
        raw_text = clean_text(raw_text)
        raw_text_size = len(raw_text)
        base_chunks = get_document_chunks(raw_text)
        for i, chunk_text in enumerate(base_chunks):
            chunks_with_location.append({
                "text": chunk_text,
                "location": f"블록 {i + 1}"
            })
    
    if not chunks_with_location:
        st.warning("파일에서 유효한 텍스트 청크를 생성할 수 없습니다.")
        return
    
    if raw_text_size < 50:
        st.warning("파일 내용이 너무 짧아 인덱싱을 건너뜁니다.")
        return
        
    # 2. 임베딩 생성
    with st.spinner(f"'{uploaded_file.name}' 임베딩 생성 중..."):
        chunk_texts = [chunk["text"] for chunk in chunks_with_location]
        embeddings = MODEL.encode(chunk_texts)
    
    # 3. 문서 상태 저장
    doc_data = {
        "name": uploaded_file.name,
        "size": raw_text_size,
        "chunks": []
    }
    for i in range(len(chunks_with_location)):
        doc_data["chunks"].append({
            "text": chunks_with_location[i]["text"],
            "embedding": embeddings[i],
            "location": chunks_with_location[i]["location"] 
        })
    st.session_state["docs"].append(doc_data)
    st.success(f"✅ '{uploaded_file.name}' 로드 및 인덱싱 완료 ({len(chunks_with_location)} 청크)")

def get_related_queries(query: str) -> List[str]:
    """
    쿼리를 기반으로 관련 영어 번역 및 한국어/영어 동의어를 반환합니다.
    (하드코딩된 동의어는 실제 API를 사용하는 경우 제거해야 합니다.)
    """
    query_lower = query.lower()
    queries_to_run = [query] 
    
    # --- 핵심 키워드/동의어 확장 (Hard-coded Mocking) ---
    if '재질' in query_lower or '소재' in query_lower:
        queries_to_run.extend(["material specifications", "composition", "alloy", "durability", "내구성", "합금", "규격"])
    
    if '케이블' in query_lower or '배선' in query_lower:
        queries_to_run.extend(["cable management solutions", "wiring diagram", "cabling", "전선", "접속"])
        
    if '인클로저' in query_lower or '함체' in query_lower:
        queries_to_run.extend(["enclosure product standards", "housing", "protection rating", "NEMA", "IP rating", "보호 등급"])

    if '전력' in query_lower or '배전' in query_lower:
        queries_to_run.extend(["power distribution systems", "circuit breaker", "차단기", "변압기", "transformer"])

    if 'safety' in query_lower or '안전' in query_lower:
        queries_to_run.extend(["safety regulations", "compliance", "위험", "규정 준수"])
    
    return list(set(queries_to_run))

def highlight_text(text: str, queries_to_highlight: List[str]) -> str:
    """텍스트 내에서 주어진 쿼리들을 <mark> 태그로 하이라이트합니다."""
    highlighted_text = text
    queries_to_highlight.sort(key=len, reverse=True) 
    
    for q_text in queries_to_highlight:
        q_text_stripped = clean_text(q_text)
        if not q_text_stripped: continue

        # 정규식 패턴을 사용하여 대소문자를 구분하지 않고 정확히 일치하는 단어를 찾아 교체
        pattern = re.compile(re.escape(q_text_stripped), re.IGNORECASE)
        
        def replace_func(match):
            # 원본 텍스트의 대소문자를 유지하면서 하이라이트
            return f"<mark>{match.group(0)}</mark>"
            
        highlighted_text = pattern.sub(replace_func, highlighted_text)
    
    return highlighted_text

def translate_text_free(text_to_translate: str, translator_key: str) -> str:
    """선택된 무료 번역 라이브러리를 사용하여 텍스트를 한국어로 번역합니다."""
    
    if not FREE_TRANSLATOR_LOADED or translator_key == "없음":
        return "번역 라이브러리가 로드되지 않았습니다. `deep-translator` 또는 `googletrans`를 설치해주세요."
    
    try:
        if translator_key == "deep_translator":
            # deep-translator 사용 (안정적)
            get_translator = TRANSLATOR_OBJECTS['deep_translator']
            translator = get_translator()
            
            # 텍스트가 너무 길면 분할
            if len(text_to_translate) > 4500:
                sentences = text_to_translate.split('. ')
                translated_parts = []
                current_chunk = ""
                
                for sent in sentences:
                    if len(current_chunk) + len(sent) < 4500:
                        current_chunk += sent + ". "
                    else:
                        if current_chunk:
                            translated_parts.append(translator.translate(current_chunk))
                            time.sleep(0.3)
                        current_chunk = sent + ". "
                
                if current_chunk:
                    translated_parts.append(translator.translate(current_chunk))
                
                return " ".join(translated_parts)
            else:
                return translator.translate(text_to_translate)
            
        elif translator_key == "googletrans":
            # googletrans 사용
            get_translator = TRANSLATOR_OBJECTS['googletrans']
            translator = get_translator()
            
            translation = translator.translate(text_to_translate, dest='ko')
            time.sleep(0.5)
            
            if not translation or not translation.text:
                return f"⚠️ 번역 실패: googletrans가 결과를 반환하지 않았습니다."
                
            return translation.text

        else:
            return f"알 수 없는 번역 라이브러리 타입: {translator_key}"
            
    except Exception as e:
        error_name = e.__class__.__name__
        if 'JSONDecodeError' in error_name or 'HTTPError' in error_name:
             return f"🚨 번역 오류 발생: 서버 차단 또는 API 변경으로 인해 번역에 실패했습니다. 잠시 후 다시 시도하거나 다른 번역 라이브러리를 설치해 보세요. ({error_name})"
        else:
            return f"🚨 번역 오류 발생: {error_name} - {e}"


# --- 4. Streamlit UI 구성 ---

# 사이드바: 파일 업로드 및 문서 목록
with st.sidebar:
    st.header("📄 문서 관리")
    
    # --- ⭐ 번역 라이브러리 선택 드롭다운 추가 ---
    st.subheader("🌐 번역 라이브러리 설정")
    
    if FREE_TRANSLATOR_LOADED:
        # 드롭다운에서 보여줄 레이블 목록
        translator_labels = list(AVAILABLE_TRANSLATORS.keys())
        
        # 기본값을 현재 세션 상태에 저장된 key에 해당하는 label로 설정
        default_index = 0
        current_key = st.session_state["selected_translator_key"]
        
        # 'key'를 'label'로 변환
        key_to_label = {v: k for k, v in AVAILABLE_TRANSLATORS.items()}
        current_label = key_to_label.get(current_key, translator_labels[0] if translator_labels else "없음")
        
        # 현재 레이블의 인덱스 찾기
        try:
            default_index = translator_labels.index(current_label)
        except ValueError:
            default_index = 0

        selected_label = st.selectbox(
            "사용할 번역 라이브러리 선택",
            options=translator_labels,
            index=default_index,
            key="translator_select_widget" 
        )
        
        # 선택된 레이블에 해당하는 내부 키(deep_translator 또는 googletrans)를 찾아 상태에 저장
        st.session_state["selected_translator_key"] = AVAILABLE_TRANSLATORS[selected_label]
        
    else:
        st.session_state["selected_translator_key"] = "없음"
        st.warning("사용 가능한 번역 라이브러리가 없습니다.")
    # --- ⭐ 번역 라이브러리 선택 끝 ---
    
    st.markdown("---")
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT 파일 업로드", 
        type=['pdf', 'docx', 'txt', 'md'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            process_file(uploaded_file)
            
    # 로드된 문서 목록
    st.markdown("### 📚 로드된 문서 목록")
    if st.session_state["docs"]:
        for doc in st.session_state["docs"]:
            chunk_count = len(doc['chunks'])
            st.caption(f"**{doc['name']}** ({chunk_count} 청크)")
        
        if st.button("모든 문서 제거"):
            st.session_state["docs"] = []
            st.session_state["search_history"] = []
            # 번역 결과 상태도 제거
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("translation_")]
            for k in keys_to_delete:
                del st.session_state[k]
            st.rerun()
    else:
        st.info("아직 로드된 문서가 없습니다.")

# --- 5. 시맨틱 검색 로직 ---

st.markdown("---")

if not SEMANTIC_LIBRARIES_LOADED:
    st.warning("시맨틱 검색 라이브러리가 로드되지 않아 검색을 수행할 수 없습니다.")
elif not st.session_state["docs"]:
    st.info("문서를 업로드하면 검색을 시작할 수 있습니다.")
else:
    query = st.text_input(
        "🔍 검색어 입력 (문서 내용과 관련된 질문이나 키워드를 입력하세요)",
        key="query_input"
    )
    
    col_settings, col_n_results, col_execute = st.columns([1, 1, 3])
    
    with col_settings:
        current_threshold = st.slider(
            "유사도 임계값", 
            min_value=0.0, 
            max_value=1.0, 
            value=SIMILARITY_THRESHOLD, 
            step=0.05,
            key="similarity_threshold_slider"
        )
    
    with col_n_results:
        N_RESULTS = st.number_input(
            "표시할 최대 문서 그룹 수",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="n_results_input"
        )

    with col_execute:
        st.write("") 
        search_button = st.button("검색 실행", type="primary", key="search_execute_button")

    # ⭐ 검색 버튼 클릭 시에만 검색 로직 실행
    if search_button and st.session_state["query_input"]:
        query = st.session_state["query_input"]
        
        if query not in st.session_state["search_history"]:
            st.session_state["search_history"].append(query)
            
        with st.spinner(f"'{query}'에 대한 문서를 확장 검색 중..."):
            
            # --- 0. 검색 쿼리 목록 및 임베딩 준비 ---
            queries_to_run = get_related_queries(query)
            query_embeddings = {}
            for q_text in queries_to_run:
                try:
                    query_embeddings[q_text] = MODEL.encode(q_text) 
                except Exception as e:
                    st.error(f"'{q_text}' 임베딩 생성 오류: {e}")
                    st.stop()
            
            display_queries = ", ".join(queries_to_run)
            all_chunks_results = []
            
            # --- 2. 모든 청크에 대해 하이브리드 검색 수행 ---
            for doc in st.session_state["docs"]:
                for chunk in doc.get("chunks", []):
                    
                    max_sim = -1.0
                    best_q_text = ""
                    is_lexical_override = False
                    
                    # 2-1. 시맨틱 유사도 계산
                    for q_text, q_embedding in query_embeddings.items():
                        sim = cosine_similarity(
                            q_embedding.reshape(1, -1),
                            chunk["embedding"].reshape(1, -1)
                        )[0][0]
                        
                        if sim > max_sim:
                            max_sim = sim
                            best_q_text = q_text
                    
                    # 2-2. 키워드 일치 확인 (Lexical Check)
                    chunk_lower = clean_text(chunk["text"].lower())
                    matched_queries_for_override = [
                        q for q in queries_to_run 
                        if clean_text(q.lower()) in chunk_lower
                    ]
                    is_exact_match = len(matched_queries_for_override) > 0
                    
                    # 2-3. 결과 포함 결정 및 점수 부여
                    if is_exact_match:
                        max_sim = LEXICAL_OVERRIDE_SCORE 
                        best_q_text = ", ".join(matched_queries_for_override) 
                        is_lexical_override = True
                    elif max_sim >= current_threshold:
                        pass 
                    else:
                        continue
                    
                    all_chunks_results.append({
                        "doc_name": doc["name"],
                        "text": chunk["text"],
                        "similarity": max_sim,
                        "location": chunk["location"],
                        "best_query_text": best_q_text, 
                        "is_lexical_override": is_lexical_override, 
                        "all_search_queries": queries_to_run 
                    })
                            
            # 3. 유사도 순으로 정렬 (개별 청크 기준)
            all_chunks_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 4. 결과 그룹화 (파일명 기준)
            grouped_results: Dict[str, List[Dict[str, Union[str, float, bool, List[str]]]]] = defaultdict(list)
            for r in all_chunks_results:
                group_key = r['doc_name']
                grouped_results[group_key].append(r)
                
            sorted_groups = []
            for doc_name, chunks in grouped_results.items():
                max_group_sim = max(c['similarity'] for c in chunks)
                group_is_lexical_override = any(c['is_lexical_override'] for c in chunks)
                
                sorted_groups.append({
                    "doc_name": doc_name,
                    "max_sim": max_group_sim,
                    "is_lexical_override": group_is_lexical_override,
                    "chunks": chunks
                })
            
            # 최종 정렬: 정확도 일치 여부 -> 최고 유사도 순
            sorted_groups.sort(key=lambda x: (x["is_lexical_override"], x["max_sim"]), reverse=True)
            
            # 검색 결과를 세션 상태에 저장
            st.session_state["last_search_results"] = sorted_groups
            st.session_state["search_performed"] = True
    
    # --- 6. 검색 결과 출력 ---
    if st.session_state.get("search_performed", False):
        sorted_groups = st.session_state["last_search_results"]

        st.subheader(f"총 {len(sorted_groups)}개의 문서 그룹 (임계값: {current_threshold}, 정확도 우선 정렬)")
        st.markdown(f"**ℹ️ 확장 검색 정보:** 원본 쿼리: '{st.session_state['query_input']}'. 검색에 사용된 확장 쿼리/동의어: **{', '.join(get_related_queries(st.session_state['query_input']))}**")
        
        if not sorted_groups:
            st.info("검색 결과가 없습니다. 임계값을 낮추거나 다른 검색어를 사용해 보세요.")
        
        for idx, group in enumerate(sorted_groups[:int(N_RESULTS)]):
            group['chunks'].sort(key=lambda x: x["similarity"], reverse=True)

            match_tag = " [⭐️ 정확히 일치]" if group['is_lexical_override'] else ""
            
            expander_title = (
                f"✨ 최고 유사도: {group['max_sim']:.3f}{match_tag} | 파일명: **{group['doc_name']}**"
            )

            with st.expander(expander_title, expanded=False):
                
                for chunk_idx, r in enumerate(group['chunks']):
                    
                    # ⭐ 번역 결과를 저장할 고유 키 정의
                    translation_state_key = f"translation_{group['doc_name']}_{idx}_{chunk_idx}"
                    
                    # --- 원문 스니펫 하이라이팅 처리 ---
                    highlighted_text = highlight_text(r['text'], r['all_search_queries'])
                    
                    st.markdown("---")
                    
                    # 청크 제목 (위치 정보 포함) 및 기여 쿼리 표시
                    chunk_header = (
                        f"**{r['location']}** | 청크 {chunk_idx + 1} (유사도: {r['similarity']:.3f})"
                    )
                    st.markdown(f"{chunk_header} | 기여 쿼리: *{r['best_query_text']}*")
                    
                    # 원문 스니펫 표시
                    st.markdown("#### 📖 원문") 
                    st.markdown(highlighted_text, unsafe_allow_html=True) 
                    
                    # 번역 섹션
                    st.markdown("#### 🌐 원문 번역")
                    
                    col_btn, col_placeholder = st.columns([1, 4])
                    
                    # 1. 번역 버튼: 버튼 클릭 시 번역 수행 및 상태 저장
                    def handle_translation_wrapper(text_to_translate, key):
                        # st.session_state에서 필요한 값을 읽어옴
                        selected_key = st.session_state["selected_translator_key"]
                        # 번역 함수 호출
                        with st.spinner(f"무료 번역 라이브러리 ({selected_key})를 사용하여 번역 중..."):
                            translated_text = translate_text_free(text_to_translate, selected_key)
                            # 상태 업데이트
                            st.session_state[key] = translated_text
                        
                    # 버튼을 클릭하면 위 핸들러를 호출하도록 설정
                    if col_btn.button(
                        f"✨ 한국어로 번역 ({st.session_state.get('selected_translator_key', '없음')})", 
                        key=f"translate_btn_{translation_state_key}",
                        on_click=handle_translation_wrapper,
                        args=(r['text'], translation_state_key)
                    ):
                        pass
                        
                    # 2. 번역 결과 표시: 세션 상태에 저장된 결과 표시
                    if translation_state_key in st.session_state:
                        col_placeholder.info(st.session_state[translation_state_key])
                    else:
                        col_placeholder.caption("번역을 보려면 '한국어로 번역' 버튼을 누르세요.")
                
        st.markdown("---")

# --- 검색 히스토리 (사이드바 유지) ---
if st.session_state.get("search_history"):
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🕐 최근 검색")
        for h in reversed(st.session_state["search_history"][-5:]):
            st.caption(h)

