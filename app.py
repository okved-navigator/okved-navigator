import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ОКВЭД-Навигатор", page_icon="🔍", layout="centered")

@st.cache_resource
def load_classifier(csv_path='okved_codes.csv'):
    data = []
    # Пробуем разные кодировки и разные разделители
    for enc in ['cp1251', 'utf-8']:
        for sep in [',', ';', '\t']:
            try:
                # Читаем файл
                df_raw = pd.read_csv(csv_path, encoding=enc, sep=sep, header=None, on_bad_lines='skip')
                data = []
                for index, row in df_raw.iterrows():
                    # Склеиваем всю строку в один текст
                    text = " ".join([str(x) for x in row.values if pd.notna(x)])
                    
                    # Ищем последний код по шаблону XX.XX (или XX.XX.X)
                    import re
                    match = re.search(r'([0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?)\s*$', text)
                    if match:
                        code = match.group(1).strip()
                        desc = text[:match.start()].strip()
                        # Берем только 4-значные коды (например 10.71)
                        if len(code) == 5:
                            data.append({'description': desc, 'code': code})
                
                if len(data) > 100: # Если нашли больше 100 кодов, значит формат подошел, останавливаемся
                    break
            except Exception as e:
                continue
        if len(data) > 100:
            break

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(), None, None
        
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    texts = [f"passage: Вид деятельности: {desc}" for desc in df['description']]
    embeddings = model.encode(texts, show_progress_bar=False)
    return df, model, embeddings

def classify(text, df, model, embeddings, top_k=3):
    query = f"query: Деятельность: {text}"
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    
    best_score = similarities[top_indices[0]]
    for idx in top_indices:
        raw_score = similarities[idx]
        norm_conf = (raw_score / best_score) * 99 if best_score > 0 else 0
        
        results.append({
            'code': df.iloc[idx]['code'],
            'description': df.iloc[idx]['description'],
            'confidence': norm_conf
        })
    return results

st.title("ОКВЭД-Навигатор")
st.caption("ИИ-помощник для подбора кодов ОКВЭД по описанию деятельности")

with st.spinner("Загрузка модели и справочника..."):
    df, model, embeddings = load_classifier()

if df.empty:
    st.error("Не удалось загрузить коды. Проверьте файл okved_codes.csv")
else:
    st.write(f"Загружено кодов: {len(df)}")
    
    text = st.text_area("Опишите вашу деятельность простыми словами:", height=100,
                        placeholder="Например: я пеку торты на заказ")

    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.selectbox("Сколько вариантов показать", [1, 3, 5], index=1)
    
    if st.button("Подобрать код", type="primary"):
        if not text.strip():
            st.warning("Введите описание деятельности")
        else:
            with st.spinner("Подбираю код..."):
                results = classify(text, df, model, embeddings, top_k=top_k)
            st.subheader("Результаты")
            for i, res in enumerate(results, 1):
                conf = int(res['confidence'])
                st.markdown(f"**{i}. Код: `{res['code']}`** — уверенность {conf}%")
                st.write(f"Описание: {res['description']}")
                st.divider()

st.divider()
st.caption("Прототип на основе эмбеддингов. Работает локально, без внешних API.")